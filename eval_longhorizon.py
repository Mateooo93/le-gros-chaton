"""Long-horizon agentic eval — measures sustained-progress capability.

Research (Long-Horizon-Terminal-Bench): agents fail on long tasks because they
can't sustain progress, verify completion, or stay within budget — not because
individual steps are wrong. HumanEval-style single-shot evals miss this.

This eval runs the SWEAgent on multi-step bug-fix repos (like the trajectory
generator's, but harder) and measures:
  - Resolution rate (did it fix the bug?)
  - Turns used (efficiency; too many = looping)
  - Tool-call repeats (action looping — the #1 small-model failure mode)
  - Context-growth behavior (did it use prune / stay within budget?)
  - Self-verification (did it run tests before finish?)

Usage:
    python eval_longhorizon.py --n 10 --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent
    python eval_longhorizon.py --n 20 --use-4bit --only-failures  # loop analysis
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from agent_swe import SWEAgent  # noqa: E402
from eval_qwen import load_qwen  # noqa: E402
from gen_trajectories import _buggy_versions, make_repo, verify_repo  # noqa: E402


def analyze_trace(trace: list[dict], issue: str) -> dict:
    """Extract loop/turn/verification metrics from a full trace."""
    calls = []
    for m in trace:
        if m.get("role") != "assistant":
            continue
        for cm in re.finditer(r'```(\w+)\s*\n(.*?)```', m.get("content", ""), re.DOTALL):
            calls.append((cm.group(1), cm.group(2).strip()))
    # Action looping: repeated identical tool calls
    seen = set()
    dup = 0
    for c in calls:
        if c in seen:
            dup += 1
        else:
            seen.add(c)
    tool_names = [c[0] for c in calls]
    # Self-verification: ran tests at least once
    ran_tests = "run_test" in tool_names
    # Used prune (context management)
    used_prune = "prune" in tool_names
    # Premature stop: finished without ever running tests
    finished = any(c[0] == "finish" for c in calls)
    verified_before_finish = ran_tests and finished
    return {
        "tool_calls": len(calls),
        "unique_calls": len(seen),
        "repeats": dup,
        "loop_rate": round(dup / max(1, len(calls)), 3),
        "ran_tests": ran_tests,
        "used_prune": used_prune,
        "finished": finished,
        "verified_before_finish": verified_before_finish,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--out", default="longhorizon_results.json")
    parser.add_argument("--only-failures", action="store_true",
                        help="Print per-task detail for failed runs (loop analysis)")
    args = parser.parse_args()

    print("[eval] Loading model...")
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.use_4bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    templates = _buggy_versions()
    results = []
    work = tempfile.mkdtemp(prefix="chaton_eval_")

    for i in range(args.n):
        tpl = templates[i % len(templates)]
        repo_dir = os.path.join(work, f"eval_{i}")
        make_repo(tpl, repo_dir)
        agent = SWEAgent(model, tokenizer, repo_dir, device=device, tdd=False)
        t0 = time.time()
        result = agent.run(tpl["issue"], instance_id=f"{tpl['id']}_{i}")
        dt = time.time() - t0
        verified, n_pass, n_total = verify_repo(repo_dir, tpl["test"])
        metrics = analyze_trace(result.get("trace", []), tpl["issue"])
        entry = {
            "instance_id": result["instance_id"],
            "task": tpl["id"],
            "verified": verified,
            "n_pass": n_pass,
            "n_total": n_total,
            "turns": result["turns"],
            "seconds": round(dt, 1),
            **metrics,
        }
        results.append(entry)
        print(f"[eval] {i+1}/{args.n} | {tpl['id']} | verified={verified} "
              f"({n_pass}/{n_total}) | turns={result['turns']} | "
              f"loops={metrics['repeats']} | tests_ran={metrics['ran_tests']} | {dt:.0f}s")
        if args.only_failures and not verified:
            print(f"  ! FAILED — {json.dumps(metrics)}")
        subprocess.run(["rm", "-rf", repo_dir])

    n = len(results)
    n_ok = sum(1 for r in results if r["verified"])
    avg_turns = sum(r["turns"] for r in results) / max(1, n)
    avg_repeats = sum(r["repeats"] for r in results) / max(1, n)
    avg_loop_rate = sum(r["loop_rate"] for r in results) / max(1, n)
    pct_tests = 100 * sum(1 for r in results if r["ran_tests"]) / max(1, n)
    pct_prune = 100 * sum(1 for r in results if r["used_prune"]) / max(1, n)
    summary = {
        "n": n,
        "resolution_rate": round(n_ok / max(1, n), 3),
        "avg_turns": round(avg_turns, 1),
        "avg_loop_repeats": round(avg_repeats, 1),
        "avg_loop_rate": round(avg_loop_rate, 3),
        "pct_ran_tests": round(pct_tests, 1),
        "pct_used_prune": round(pct_prune, 1),
        "verified": n_ok,
    }
    out_path = os.path.join(PROJ_ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n=== SUMMARY ({args.n} long-horizon tasks) ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")

    # Interpret: these are the numbers that move on Terminal-Bench-style evals
    print("\n--- Long-horizon health check ---")
    print(f"Resolution: {summary['resolution_rate']*100:.0f}% "
          f"(target: rising across training)")
    print(f"Avg turns: {avg_turns} (high + many loops = action looping)")
    print(f"Avg loop repeats: {avg_repeats} / tool call")
    print(f"Tests run before finish: {pct_tests:.0f}% (self-verification)")
    print(f"Used prune tool: {pct_prune:.0f}% (context management)")


if __name__ == "__main__":
    main()
