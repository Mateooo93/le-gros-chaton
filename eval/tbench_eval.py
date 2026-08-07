"""Terminal-Bench 2.0 evaluation harness for Le Gros Chaton.

The official Terminal-Bench 2.0 harness is **Harbor** (the successor of the
v1 ``tb`` CLI — ``pip install terminal-bench`` was the v1 tool; TB 2.0 is
run with ``pip install harbor`` / ``harbor run -d terminal-bench@2.0``).
This module wraps Harbor:

  - ``--list``                lists all TB 2.0 tasks + count (cached dataset)
  - ``--run`` / ``--dry-run`` runs tasks through our custom Harbor agent
                              (``eval.tb_agent.LeGrosChatonTBAgent``) inside
                              the official Docker sandboxes, then appends a
                              pass/fail report to ``benchmark_results.jsonl``

Full-eval commands (leaderboard methodology)
--------------------------------------------
Leaderboard protocol (tbench.ai/leaderboard/terminal-bench/2.0):
same agent, containerized sandboxes, 100-turn cap, task timeouts/resource
limits unmodified. Baseline for Qwen3.5-9B on the leaderboard: 9.2% (rank
138). Run the whole benchmark with:

    # BASE model via any OpenAI-compatible server (recommended):
    python eval/tbench_eval.py --run \
        --model-server http://<host>:8000 --model-name Qwen/Qwen3.5-9B \
        --model-api-key <key> \
        --label "Qwen3.5-9B-baseline" --adapter base

    # Same, but with the trajectory-SFT adapter served as a merged model:
    python eval/tbench_eval.py --run \
        --model-server http://<host>:8000 --model-name le-gros-chaton-traj-sft \
        --label "le-gros-chaton-traj-sft" --adapter traj_sft

    # FINAL RLVR model:
    python eval/tbench_eval.py --run \
        --model-server http://<host>:8000 --model-name le-gros-chaton-rlvr \
        --label "le-gros-chaton-rlvr" --adapter rlvr

    # Local GPU box with the weights (no server):
    python eval/tbench_eval.py --run \
        --local-model Qwen/Qwen3.5-9B --local-ckpt mateo0093/le-gros-chaton-qwen \
        --label "le-gros-chaton-traj-sft" --adapter traj_sft

    # Verify the harness (sandbox + agent loop) without any model:
    python eval/tbench_eval.py --dry-run

Cost / time estimate per task
-----------------------------
- Model inference (9B): ~1-4k tokens in + ~1k tokens out per turn, 5-40
  turns/task. On an L4 (24GB, ~$0.80/hr Modal): ~3-6 min/task model time
  -> roughly **$0.05-0.10/task**. Full 89-task run on one L4: ~5-9 hrs,
  **~$5-8**; on a faster A10G/A100 or a 2x-faster server, roughly halved.
- Sandbox: Harbor reuses cached Docker images; first task pulls the base
  image (28-150MB for text tasks, more for big-image tasks like
  install-windows-3.11). No per-task GPU cost in the sandbox.
- Wall-clock: ~5-20 min/task wall time depending on model speed; the task
  agent timeout is 900s (unmodified, per leaderboard rules).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

DATASET = "terminal-bench@2.0"
CACHE_DIR = PROJ_ROOT / "eval" / ".tb_cache"
RESULTS_FILE = PROJ_ROOT / "benchmark_results.jsonl"
AGENT_IMPORT = "eval.tb_agent:LeGrosChatonTBAgent"

PILOT_TASKS = [
    "fix-git",
    "overfull-hbox",
    "log-summary-date-ranges",
    "regex-log",
    "count-dataset-tokens",
]


# --------------------------------------------------------------------------
# Task listing
# --------------------------------------------------------------------------


def check_harbor() -> str | None:
    """Return the harbor binary path, or None with a documented hint."""
    import shutil
    found = shutil.which("harbor")
    if found:
        return found
    # Not on PATH: look next to the running interpreter (venv bin dir).
    venv_bin = Path(sys.executable).parent / "harbor"
    return str(venv_bin) if venv_bin.is_file() else None


def list_tasks(force_refresh: bool = False) -> list[str]:
    """Download (or reuse) the TB 2.0 dataset and return sorted task names."""
    harbor = check_harbor()
    if harbor is None:
        print(
            "[tbench] ERROR: `harbor` CLI not found. Install with:\n"
            "    uv pip install --python .venv/bin/python harbor\n"
            "Harbor is the official Terminal-Bench 2.0 harness (the v1 `tb` "
            "CLI / `pip install terminal-bench` is not used for TB 2.0)."
        )
        raise SystemExit(2)
    dataset_dir = CACHE_DIR / DATASET.replace("@", "__")
    tasks_dir = dataset_dir / "terminal-bench"
    if not tasks_dir.is_dir() or force_refresh:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tbench] Downloading dataset {DATASET} (cached at {dataset_dir})...")
        subprocess.run(
            [harbor, "dataset", "download", DATASET, "-o", str(dataset_dir)],
            check=True, timeout=600,
        )
    if not tasks_dir.is_dir():
        print(f"[tbench] ERROR: dataset download did not produce {tasks_dir}")
        raise SystemExit(2)
    tasks = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())
    return tasks


def cmd_list(args) -> None:
    tasks = list_tasks(force_refresh=args.refresh)
    print(f"[tbench] Terminal-Bench 2.0: {len(tasks)} tasks available")
    print(f"[tbench] Dataset: {DATASET}")
    if args.all:
        for t in tasks:
            print(f"  {t}")
    else:
        print("[tbench] Sample task ids:")
        for t in tasks[:15]:
            print(f"  {t}")
        print(f"  ... ({len(tasks) - 15} more; use --all to print every task)")
    print("[tbench] Pilot subset:", ", ".join(PILOT_TASKS))


# --------------------------------------------------------------------------
# Harbor run
# --------------------------------------------------------------------------


def select_tasks(tasks: list[str], include: list[str], exclude: list[str],
                 n_tasks: int | None) -> list[str]:
    """Apply include/exclude glob filters + n_tasks cap."""
    import fnmatch
    selected = tasks
    if include:
        selected = [t for t in selected
                    if any(fnmatch.fnmatch(t, pat) for pat in include)]
    if exclude:
        selected = [t for t in selected
                    if not any(fnmatch.fnmatch(t, pat) for pat in exclude)]
    if n_tasks is not None:
        selected = selected[:n_tasks]
    return selected


def build_harbor_cmd(args, selected: list[str]) -> list[str]:
    harbor = check_harbor()
    cmd = [
        harbor, "run", "-d", DATASET,
        "-a", AGENT_IMPORT,
        "-m", args.model_name or "qwen/qwen3.5-9b",
        "--jobs-dir", str(args.jobs_dir),
        "-y",
        "--quiet",
    ]
    if args.n_concurrent:
        cmd += ["-n", str(args.n_concurrent)]
    if args.attempts:
        cmd += ["-k", str(args.attempts)]
    # Select exactly the requested tasks (explicit allowlist via globs).
    for pat in selected:
        cmd += ["-i", pat]

    ak = []
    if args.model_server:
        ak.append(f"model_server_url={args.model_server}")
    if args.model_api_key:
        ak.append(f"model_api_key={args.model_api_key}")
    if args.hf_inference:
        ak.append("hf_inference=true")
    if args.local_model:
        ak.append(f"local_model={args.local_model}")
    if args.local_ckpt:
        ak.append(f"local_ckpt={args.local_ckpt}")
    if args.mock:
        ak.append("mock=true")
    if args.four_bit:
        ak.append("four_bit=true")
    if args.temperature:
        ak.append(f"temperature={args.temperature}")
    if args.max_turns:
        ak.append(f"max_turns={args.max_turns}")
    if args.server_ctx_limit:
        ak.append(f"server_ctx_limit={args.server_ctx_limit}")
    for kv in ak:
        cmd += ["--ak", kv]
    return cmd


def run_trials(args, selected: list[str]) -> dict:
    cmd = build_harbor_cmd(args, selected)
    print(f"[tbench] Running {len(selected)} task(s): {', '.join(selected)}")
    print(f"[tbench] CMD: {' '.join(cmd)}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJ_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_TOKEN", "")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[tbench] WARNING: harbor exited {proc.returncode} after {dt:.0f}s")
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    else:
        print(f"[tbench] harbor finished in {dt:.0f}s")
    return {"returncode": proc.returncode, "seconds": dt}


# --------------------------------------------------------------------------
# Results parsing
# --------------------------------------------------------------------------


def find_result_files(jobs_dir: Path) -> list[Path]:
    """Trial-level result.json files (job-level summary files are one level up)."""
    return sorted(jobs_dir.rglob("*/result.json"))


def parse_trial(path: Path) -> dict:
    """Extract a compact row from a Harbor trial result.json."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"file": str(path), "parse_error": str(exc)}
    task_name = data.get("task_name")
    if not task_name:
        # Job-level summary file (no trial payload); not a task result.
        return None
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    passed = _rewards_passed(rewards)
    row = {
        "task_id": task_name,
        "trial_name": data.get("trial_name"),
        "passed": passed,
        "rewards": rewards,
        "error": (data.get("exception_info") or {}).get("exception_message"),
        "seconds": _trial_seconds(data),
        "agent_meta": (data.get("agent_result") or {}).get("metadata") or {},
    }
    return row


def _rewards_passed(rewards: dict) -> bool:
    if not rewards:
        return False
    if "passed" in rewards:
        return bool(rewards["passed"])
    if "success" in rewards:
        return bool(rewards["success"])
    # Generic: any positive reward counts as a pass (test.sh-based verifiers).
    return any(v > 0 for v in rewards.values() if isinstance(v, (int, float)))


def _trial_seconds(data: dict) -> float | None:
    agent_exec = data.get("agent_execution") or {}
    started = agent_exec.get("started_at")
    finished = agent_exec.get("finished_at")
    if started and finished:
        from datetime import datetime
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        try:
            return (datetime.fromisoformat(finished) -
                    datetime.fromisoformat(started)).total_seconds()
        except ValueError:
            pass
    return None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def append_results(rows: list[dict], args, results_file: Path) -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    with open(results_file, "a") as f:
        for row in rows:
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "benchmark": "terminal-bench-2.0",
                "task_id": row.get("task_id"),
                "model": args.model_name or args.local_model or
                         (args.label or "unknown"),
                "adapter": args.adapter,
                "passed": bool(row.get("passed")),
                "rewards": row.get("rewards"),
                "error": row.get("error"),
                "seconds": row.get("seconds"),
                "turns": row.get("agent_meta", {}).get("turns"),
                "tool_calls": row.get("agent_meta", {}).get("tool_calls"),
                "output_excerpt": _excerpt(row),
            }
            f.write(json.dumps(record) + "\n")
    print(f"[tbench] Appended {len(rows)} result(s) to {results_file}")


def _excerpt(row: dict) -> str:
    meta = row.get("agent_meta") or {}
    note = (meta.get("finish_note") or "")[:300]
    err = (row.get("error") or "")[:200]
    return note or err or ""


def summarize(rows: list[dict], total_expected: int) -> None:
    n_pass = sum(1 for r in rows if r.get("passed"))
    n_total = len(rows)
    print("\n" + "=" * 60)
    print(f"Terminal-Bench 2.0 pilot: {n_pass}/{n_total} passed "
          f"({100 * n_pass / max(n_total, 1):.1f}%)")
    print("=" * 60)
    for r in sorted(rows, key=lambda x: str(x.get("task_id"))):
        status = "PASS" if r.get("passed") else "FAIL"
        extra = r.get("error") or ""
        print(f"  [{status}] {r.get('task_id')}"
              + (f"  ({r.get('seconds', 0):.0f}s)" if r.get("seconds") else "")
              + (f"  error={extra[:80]}" if extra else ""))
    if n_total < total_expected:
        print(f"[tbench] NOTE: {total_expected - n_total} trial(s) produced no "
              f"result row (crashed before verification).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal-Bench 2.0 eval for Le Gros Chaton (via Harbor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true",
                        help="List available TB 2.0 tasks and exit")
    parser.add_argument("--run", action="store_true",
                        help="Run tasks (default when no other mode given)")
    parser.add_argument("--all", action="store_true",
                        help="With --list: print every task name")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download of the dataset cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the harness with a mock model (no GPU/API "
                             "needed) to verify sandbox + loop paths")

    sel = parser.add_argument_group("task selection")
    sel.add_argument("--tasks", default=None,
                     help="Comma-separated task ids (overrides --n-tasks)")
    sel.add_argument("--include", action="append", default=[],
                     help="Glob of task names to include (repeatable)")
    sel.add_argument("--exclude", action="append", default=[],
                     help="Glob of task names to exclude (repeatable)")
    sel.add_argument("--n-tasks", type=int, default=None,
                     help="Max number of tasks (default: pilot=5 when --tasks "
                          "unset)")

    model = parser.add_argument_group("model backend")
    model.add_argument("--model-server", default=None,
                       help="OpenAI-compatible base URL (e.g. Modal vLLM "
                            "endpoint). Preferred path.")
    model.add_argument("--model-api-key", default=None,
                       help="Bearer key for --model-server")
    model.add_argument("--model-name", default=None,
                       help="Model name sent to the server / recorded")
    model.add_argument("--hf-inference", action="store_true",
                       help="Use the HF Inference API (router.huggingface.co) "
                            "with HF_TOKEN — zero-infra backend for the "
                            "base-model pilot")
    model.add_argument("--local-model", default=None,
                       help="HF model id for in-process generation (GPU box)")
    model.add_argument("--local-ckpt", default=None,
                       help="HF LoRA adapter / checkpoint for --local-model")
    model.add_argument("--four-bit", action="store_true",
                       help="4-bit load for --local-model")
    model.add_argument("--mock", action="store_true",
                       help="Use the scripted mock model (implies --dry-run)")

    run = parser.add_argument_group("run options")
    run.add_argument("--label", default=None,
                     help="Model label recorded in results (defaults to "
                          "--model-name / --local-model)")
    run.add_argument("--adapter", default="base",
                     help="Adapter tag recorded in results (base|traj_sft|rlvr)")
    run.add_argument("--temperature", type=float, default=None)
    run.add_argument("--max-turns", type=int, default=None)
    run.add_argument("--server-ctx-limit", type=int, default=None,
                     help="History budget in tokens for the serving backend "
                          "(default: 14000)")
    run.add_argument("--n-concurrent", type=int, default=1)
    run.add_argument("--attempts", type=int, default=None,
                     help="Attempts per task (default: 1; leaderboard uses 5)")
    run.add_argument("--jobs-dir", default=str(PROJ_ROOT / "eval" / ".tb_jobs"))
    run.add_argument("--results-file", default=None,
                     help="Results JSONL path (default: benchmark_results.jsonl; "
                          "dry-run writes to eval/.tb_dryrun_results.jsonl)")
    args = parser.parse_args()

    if args.list:
        cmd_list(args)
        return

    if not args.mock and not args.dry_run and not args.model_server \
            and not args.local_model and not args.hf_inference:
        print("[tbench] No model backend: use --model-server, --local-model, "
              "--hf-inference, or --mock/--dry-run (see --help).")
        raise SystemExit(2)

    if args.hf_inference:
        args.model_server = "https://router.huggingface.co"
        args.model_api_key = args.model_api_key or os.environ.get("HF_TOKEN", "")
        if not args.model_api_key:
            print("[tbench] --hf-inference requires HF_TOKEN in the "
                  "environment (or --model-api-key).")
            raise SystemExit(2)
        args.model_name = args.model_name or "Qwen/Qwen3.5-9B"
        if not args.label:
            args.label = args.model_name

    tasks = list_tasks(force_refresh=args.refresh)

    if args.tasks:
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in selected if t not in tasks]
        if missing:
            print(f"[tbench] Unknown task id(s): {', '.join(missing)}")
            raise SystemExit(2)
    elif args.include or args.exclude:
        selected = select_tasks(tasks, args.include, args.exclude, args.n_tasks)
    else:
        n = args.n_tasks or 5
        selected = PILOT_TASKS if all(t in tasks for t in PILOT_TASKS) \
            else tasks[:n]

    if args.dry_run:
        args.mock = True
        selected = selected[: min(len(selected), 1)]
        print("[tbench] DRY-RUN: mock model, 1 task")

    # Fresh per-run subdir so result parsing only sees this run's trials.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    jobs_dir = Path(args.jobs_dir) / run_id
    jobs_dir.mkdir(parents=True, exist_ok=True)
    args.jobs_dir = str(jobs_dir)

    if args.results_file is None:
        if args.dry_run:
            args.results_file = str(PROJ_ROOT / "eval" / ".tb_dryrun_results.jsonl")
        else:
            args.results_file = str(RESULTS_FILE)
    results_file = Path(args.results_file)

    run_trials(args, selected)

    rows = [r for r in (parse_trial(p)
                         for p in find_result_files(Path(args.jobs_dir)))
            if r is not None]
    if rows:
        append_results(rows, args, results_file)
    summarize(rows, len(selected))


if __name__ == "__main__":
    main()
