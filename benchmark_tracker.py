"""Benchmark results tracker — see if training is actually improving the model.

Persists evaluation results (HumanEval, SWE-bench, tool-calls, voting) in a
JSONL registry and shows the trend across runs. Without this, we can't answer
"did SFT/RLVR actually help?".

Usage:
    python benchmark_tracker.py --add humaneval --pass-rate 35.2 --model qwen_sft
    python benchmark_tracker.py --add swebench --pass-rate 18.5 --n-samples 16 --model qwen_rlvr
    python benchmark_tracker.py --list
    python benchmark_tracker.py --trend humaneval
    python benchmark_tracker.py --compare 20260731_sft 20260801_rlvr
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

TRACKER = os.path.join(PROJ_ROOT, "benchmark_results.jsonl")


def load_all() -> list[dict]:
    if not os.path.exists(TRACKER):
        return []
    with open(TRACKER) as f:
        return [json.loads(line) for line in f if line.strip()]


def add_result(benchmark: str, pass_rate: float, model: str = "",
               n_samples: int = 1, extra: dict | None = None):
    results = load_all()
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "benchmark": benchmark,
        "pass_rate": pass_rate,
        "model": model,
        "n_samples": n_samples,
    }
    if extra:
        entry.update(extra)
    with open(TRACKER, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[tracker] Recorded {benchmark} pass_rate={pass_rate} "
          f"({model or 'unknown'}, n={n_samples})")
    return entry


def list_results():
    results = load_all()
    if not results:
        print("[tracker] No results recorded yet.")
        print("  Record your first: python benchmark_tracker.py --add humaneval --pass-rate 35.2")
        return

    print(f"\n{'='*80}")
    print(f"  Benchmark Registry ({len(results)} runs)")
    print(f"{'='*80}")
    print(f"  {'run_id':<18} {'benchmark':<12} {'pass_rate':<10} {'n':<4} {'model'}")
    print(f"  {'-'*76}")
    for r in results:
        print(f"  {r['run_id']:<18} {r['benchmark']:<12} "
              f"{r['pass_rate']:<10.1f} {r['n_samples']:<4} {r['model']}")
    print(f"{'='*80}")


def show_trend(benchmark: str):
    results = [r for r in load_all() if r["benchmark"] == benchmark]
    if not results:
        print(f"[tracker] No {benchmark} results yet.")
        return

    print(f"\n  {benchmark} trend ({len(results)} runs):")
    print(f"  {'run_id':<18} {'pass_rate':<10} {'delta':<8} {'model'}")
    print(f"  {'-'*60}")
    prev = None
    for r in results:
        delta = ""
        if prev is not None:
            d = r["pass_rate"] - prev
            delta = f"{d:+.1f}"
        print(f"  {r['run_id']:<18} {r['pass_rate']:<10.1f} {delta:<8} {r['model']}")
        prev = r["pass_rate"]

    best = max(results, key=lambda r: r["pass_rate"])
    print(f"\n  Best: {best['pass_rate']:.1f} ({best['run_id']}, {best['model']})")


def compare(run_a: str, run_b: str):
    results = load_all()
    a = [r for r in results if r["run_id"] == run_a]
    b = [r for r in results if r["run_id"] == run_b]
    if not a or not b:
        print("[tracker] Run IDs not found. Use --list to see available runs.")
        return

    print(f"\n  Comparing {run_a} vs {run_b}:")
    print(f"  {'benchmark':<12} {run_a:<10} {run_b:<10} delta")
    print(f"  {'-'*50}")
    for ra in a:
        rb = next((r for r in b if r["benchmark"] == ra["benchmark"]), None)
        if rb:
            d = rb["pass_rate"] - ra["pass_rate"]
            arrow = "▲" if d > 0 else "▼" if d < 0 else "="
            print(f"  {ra['benchmark']:<12} {ra['pass_rate']:<10.1f} "
                  f"{rb['pass_rate']:<10.1f} {d:+.1f} {arrow}")


def prune_failed():
    """Remove entries with pass_rate == 0.0 (crashed/failed runs)."""
    results = load_all()
    kept = [r for r in results if r.get("pass_rate", 0) > 0]
    removed = len(results) - len(kept)
    if removed:
        with open(TRACKER, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"[tracker] Removed {removed} failed runs (pass_rate=0.0)")
    else:
        print("[tracker] No failed runs to remove")


def export_csv(path: str = "benchmark_results.csv"):
    """Export all results as CSV for plotting."""
    import csv as _csv
    results = load_all()
    if not results:
        print("[tracker] No results to export")
        return
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[tracker] Exported {len(results)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark results tracker")
    parser.add_argument("--add", choices=["humaneval", "swebench", "toolcall", "agent"],
                        help="Record a result")
    parser.add_argument("--pass-rate", type=float, default=None,
                        help="Pass rate % for --add")
    parser.add_argument("--model", default="", help="Model/checkpoint label")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Test-time samples used")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--trend", default=None, help="Show trend for a benchmark")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two run IDs")
    parser.add_argument("--csv", nargs="?", const="benchmark_results.csv",
                        help="Export results to CSV (for plotting)")
    parser.add_argument("--prune", action="store_true",
                        help="Remove entries with 0.0 pass_rate (failed runs)")
    args = parser.parse_args()

    if args.add:
        if args.pass_rate is None:
            print("[tracker] --pass-rate required with --add")
            return
        add_result(args.add, args.pass_rate, model=args.model, n_samples=args.n_samples)
    if args.list:
        list_results()
    if args.trend:
        show_trend(args.trend)
    if args.compare:
        compare(*args.compare)
    if args.csv:
        export_csv(args.csv)
    if args.prune:
        prune_failed()
    if not (args.add or args.list or args.trend or args.compare or args.csv or args.prune):
        parser.print_help()


if __name__ == "__main__":
    main()
