"""Orchestrate the full 4-stage RL training + evaluation pipeline.

Ties together base pretraining, RFT, RLVR, PRM training, agent-loop RL, and
agentic evaluation into a single pipeline run.  Each stage loads the previous
stage's checkpoint, trains, saves, and evaluates.

Stages:
  0  base_pretrain   — train.py: train the base MoE model on code/prose
  1  rft             — rft.py: rejection sampling → SFT on verified solutions
  2  rlvr            — rlvr.py: GRPO with code-verifier reward
  3  prm             — prm.py: train the Process Reward Model head
  4  agent_rl        — agent_rl.py: agent-loop-as-rollout RL
  5  eval            — eval/agent_eval.py: final agentic evaluation

Usage:
  python pipeline.py --profile smol-fat --stages all
  python pipeline.py --profile dev --stages 0 1 2 -n 10
  python pipeline.py --profile smol-fat --stages 4 --resume

Resume: each stage saves training/{profile}/{stage}/done, so re-running
--stages all or --stages 3 skips completed stages unless --force is given.
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    "base_pretrain",
    "rft",
    "rlvr",
    "prm",
    "agent_rl",
    "eval",
]

# Default arguments for each stage (merged with user overrides)
STAGE_DEFAULTS: dict[str, list[str]] = {
    "base_pretrain": [
        # Run train.py with defaults — profile and data source are set
        # dynamically.  The checkpoint lands at model.pt which the next stage
        # picks up.
    ],
    "rft": [
        "--n-samples", "32",
        "--max-problems", "100",
    ],
    "rlvr": [
        "--group-size", "8",
        "--n-steps", "200",
    ],
    "prm": [
        "--n-problems", "200",
        "--n-steps", "100",
    ],
    "agent_rl": [
        "--group-size", "4",
        "--n-steps", "50",
    ],
    "eval": [
        "--limit", "10",
    ],
}


def stage_dir(profile: str, stage: str) -> str:
    """Output directory for a stage."""
    return os.path.join(PROJ_ROOT, "training", profile, stage)


def _log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def _run_cmd(cmd: list[str], cwd: str, env: dict) -> int:
    """Run a command, stream output, return rc."""
    _log(f"  $ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  wrote {path}")


def run_stage(
    profile: str,
    stage: str,
    resume: bool,
    force: bool,
    dry_run: bool,
    n_problems: int | None,
    extra_args: dict[str, str],
) -> bool:
    """Run a single pipeline stage.  Returns True on success."""
    stage_key = STAGES.index(stage)
    out_dir = stage_dir(profile, stage)
    done_flag = os.path.join(out_dir, "done")
    result_path = os.path.join(out_dir, "result.json")

    # Skip if already done and not forced
    if os.path.exists(done_flag) and not force:
        _log(f"[{stage}] already complete (remove {done_flag} to re-run)")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # Dry-run: print what would be run and return
    if dry_run:
        _log(f"[DRY-RUN] Would run stage '{stage}' with:")
        _log(f"  profile={profile}, resume={resume}, force={force}")
        _log(f"  output={out_dir}")
        return True

    # Build env with profile and HF token
    env = os.environ.copy()
    env["CHATON_PROFILE"] = profile
    env["CHATON_RESUME"] = "1" if resume else "0"
    if os.path.exists(os.path.join(PROJ_ROOT, "gpus.md")):
        with open(os.path.join(PROJ_ROOT, "gpus.md")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Support both "export KEY=VALUE" and "KEY = VALUE" formats
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        env[parts[0].strip()] = parts[1].strip()

    # Determine checkpoint source
    if stage == "base_pretrain":
        ckpt_path = None  # starts fresh
        pretrained_path = os.path.join(out_dir, "model.pt")
        env["CHATON_CKPT_PATH"] = os.path.join(out_dir, "checkpoint.pt")
    else:
        prev_stage = STAGES[stage_key - 1]
        prev_dir = stage_dir(profile, prev_stage)
        ckpt_path = os.path.join(prev_dir, "model.pt")
        pretrained_path = ckpt_path
        env["CHATON_CKPT_PATH"] = os.path.join(out_dir, "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            _log(f"[{stage}] ERROR: no checkpoint from {prev_stage} at {ckpt_path}")
            return False
        _log(f"[{stage}] loading checkpoint from {prev_stage}")

    # Build stage-specific command
    cmd: list[str] = []
    if stage == "base_pretrain":
        cmd = [sys.executable, "train.py"]
        if n_problems:
            env["CHATON_CODE_MAX_DOCS"] = str(n_problems * 1000)

    elif stage == "rft":
        cmd = [
            sys.executable, "rft.py", "collect",
            "--n-samples", str(extra_args.get("n_samples", 32)),
            "--max-problems", str(n_problems or extra_args.get("max_problems", 100)),
            "--output", os.path.join(out_dir, "rft_data.json"),
        ]
        _log(f"[rft] collecting samples from {pretrained_path}")
        rc = _run_cmd(cmd, PROJ_ROOT, env)
        if rc != 0:
            return False
        cmd = [
            sys.executable, "rft.py", "train",
            "--data", os.path.join(out_dir, "rft_data.json"),
            "--output", pretrained_path,
        ]

    elif stage == "rlvr":
        cmd = [
            sys.executable, "rlvr.py",
            "--ckpt", pretrained_path,
            "--output", pretrained_path,
            "--group-size", str(extra_args.get("group_size", 8)),
            "--n-steps", str(extra_args.get("n_steps", 200)),
            "--source", "humaneval",
            "--limit", str(n_problems or 10),
        ]

    elif stage == "prm":
        # Collect step-level training data using the RLVR checkpoint
        cmd = [
            sys.executable, "prm.py", "collect",
            "--n-problems", str(n_problems or extra_args.get("n_problems", 200)),
            "--output", os.path.join(out_dir, "prm_data.json"),
            "--label-mode", extra_args.get("label_mode", os.environ.get("CHATON_PRM_LABEL_MODE", "mc")),
        ]
        _log(f"[prm] collecting step labels from {pretrained_path}")
        rc = _run_cmd(cmd, PROJ_ROOT, env)
        if rc != 0:
            return False
        cmd = [
            sys.executable, "prm.py", "train",
            "--data", os.path.join(out_dir, "prm_data.json"),
            "--ckpt", pretrained_path,
            "--output", pretrained_path,
            "--n-steps", str(extra_args.get("n_steps", 100)),
        ]

    elif stage == "agent_rl":
        cmd = [
            sys.executable, "agent_rl.py",
            "--ckpt", pretrained_path,
            "--output", pretrained_path,
            "--group-size", str(extra_args.get("group_size", 4)),
            "--n-steps", str(extra_args.get("n_steps", 50)),
            "--source", "humaneval",
            "--limit", str(n_problems or 5),
        ]

    elif stage == "eval":
        cmd = [
            sys.executable, "eval/agent_eval.py",
            "--ckpt", pretrained_path,
            "--limit", str(n_problems or extra_args.get("limit", 10)),
            "--source", "humaneval",
            "--output", result_path,
        ]

    _log(f"[{stage}] running...")
    t0 = time.time()
    rc = _run_cmd(cmd, PROJ_ROOT, env)
    elapsed = time.time() - t0
    _log(f"[{stage}] {'✓' if rc == 0 else '✗'} ({elapsed:.1f}s, rc={rc})")

    if rc != 0:
        return False

    # Save result metadata
    result = {
        "stage": stage,
        "profile": profile,
        "rc": rc,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(result_path, result)

    # Mark done
    with open(done_flag, "w") as f:
        f.write(f"completed {stage} at {result['timestamp']}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate the 4-stage RL training pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profile", default="smol-fat",
        choices=["dev", "smol-fat", "fat"],
        help="Model profile to train (default: smol-fat)",
    )
    parser.add_argument(
        "--stages", nargs="+", default=["all"],
        help="Stages to run, e.g. '0 1' or 'all' (default: all)",
    )
    parser.add_argument(
        "-n", "--n-problems", type=int, default=None,
        help="Override problem/document count for data-limited stages",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Enable checkpoint resume within each stage",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run and exit",
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Quick sanity: dev profile, tiny data, 1 step per stage",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run completed stages (ignore done flags)",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="Print available stages and exit",
    )
    args, extra = parser.parse_known_args()

    if args.list_stages:
        print("Pipeline stages:")
        for i, name in enumerate(STAGES):
            print(f"  {i}: {name}")
        return

    # --- sanity mode: dev profile, tiny data, 1 step per stage -----------
    if args.sanity:
        args.profile = "dev"
        args.force = True
        if args.n_problems is None:
            args.n_problems = 1
        os.environ["CHATON_CODE_MAX_DOCS"] = "10"
        os.environ["CHATON_CODE_MAX_TOKENS"] = "10000"
        os.environ["CHATON_MAX_ITERS"] = "2"
        os.environ["CHATON_EVAL_INTERVAL"] = "1"
        os.environ["CHATON_CODE_BLOCK"] = "128"
        os.environ["CHATON_PRM_LABEL_MODE"] = "exec"
        _log(f"SANITY MODE: profile=dev, max_docs=10, max_iters=2, prm_label=exec")

    # Resolve stage list
    if args.stages == ["all"]:
        stage_names = STAGES
    else:
        stage_names = []
        for s in args.stages:
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < len(STAGES):
                    stage_names.append(STAGES[idx])
                else:
                    print(f"Invalid stage index {idx}; valid: 0-{len(STAGES)-1}")
                    sys.exit(1)
            elif s in STAGES:
                stage_names.append(s)
            else:
                print(f"Unknown stage '{s}'; use --list-stages")
                sys.exit(1)

    _log(f"Pipeline: profile={args.profile}, stages={stage_names}")
    _log(f"Resume={args.resume}, force={args.force}")

    # Parse extra args into a dict keyed by stage name
    extra_dict: dict[str, dict] = {s: {} for s in stage_names}
    # Apply defaults
    for s in stage_names:
        for default in STAGE_DEFAULTS.get(s, []):
            if default.startswith("--"):
                key = default.lstrip("-").replace("-", "_")
                extra_dict[s].setdefault(key, True)
            elif "=" in default:
                k, v = default.split("=", 1)
                extra_dict[s][k.lstrip("-").replace("-", "_")] = v

    success = True
    for stage in stage_names:
        success &= run_stage(
            profile=args.profile,
            stage=stage,
            resume=args.resume,
            force=args.force,
            dry_run=args.dry_run,
            n_problems=args.n_problems,
            extra_args=extra_dict.get(stage, {}),
        )
        if not success:
            _log(f"Pipeline FAILED at stage '{stage}'")
            sys.exit(1)

    _log(f"Pipeline complete ✓")
    _log(f"Output: {stage_dir(args.profile, 'eval')}")


if __name__ == "__main__":
    main()