#!/usr/bin/env python3
"""Viewer for experiment logs (JSONL format written by ``log.py``).

Displays training metrics in a terminal table and computes summary statistics.
Supports single-run viewing and multi-run comparison.

Usage:
    python log_view.py runs/train_20240701/log.jsonl
    python log_view.py runs/train_20240701/log.jsonl runs/train_20240702/log.jsonl
    python log_view.py runs/train_20240701/log.jsonl --last 20
    python log_view.py runs/*/log.jsonl --summary
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def load_log(path: str) -> list[dict]:
    """Load a JSONL experiment log, filter to step records."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only show step records (not meta/header)
            if "step" in rec and "loss" in rec:
                records.append(rec)
    return records


def fmt(v, width=10):
    """Format a value for table display."""
    if v is None:
        return "—".rjust(width)
    if isinstance(v, float):
        return f"{v:>{width}.4f}"
    if isinstance(v, int):
        return f"{v:>{width}d}"
    if isinstance(v, str):
        return f"{v:>{width}s}"
    return f"{str(v):>{width}}"


def show_table(records: list[dict], title: str = "", last: int = 0):
    """Print a tabular view of training metrics."""
    if not records:
        print("  (no step records)")
        return

    if last > 0:
        records = records[-last:]

    if title:
        print(f"\n{title}")
        print("-" * len(title))

    # Collect all unique keys across records
    keys = ["step", "loss", "val_loss", "lr", "aux_loss", "z_loss",
            "tok_s", "gpu_mem_gb", "elapsed_h", "prog_pct", "eta_s"]

    # Header
    header = "  " + "  ".join(f"{k:>10}" for k in keys if any(r.get(k) is not None for r in records))
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Rows
    for r in records:
        vals = [fmt(r.get(k), 10) for k in keys if any(rr.get(k) is not None for rr in records)]
        print("  " + "  ".join(vals))


def show_summary(records: list[dict], label: str = ""):
    """Print summary statistics for a run."""
    if not records:
        return

    losses = [r.get("loss") for r in records if r.get("loss") is not None]
    val_losses = [r.get("val_loss") for r in records if r.get("val_loss") is not None]
    tok_rates = [r.get("tok_s") for r in records if r.get("tok_s") is not None and r.get("tok_s", 0) > 0]

    prefix = f"  [{label}] " if label else "  "

    print(f"\n{prefix}Summary")
    print(f"{prefix}{'-'*40}")
    if losses:
        print(f"{prefix}steps:          {records[0]['step']} → {records[-1]['step']} ({len(records)} evals)")
        print(f"{prefix}final loss:     {losses[-1]:.4f}")
        print(f"{prefix}best val loss:  {min(val_losses):.4f} (step {records[val_losses.index(min(val_losses))]['step']})" if val_losses else "")
        print(f"{prefix}min val loss:   {min(val_losses):.4f}" if val_losses else "")
    if tok_rates:
        print(f"{prefix}avg tok/s:      {sum(tok_rates)/len(tok_rates):,.0f}")
        print(f"{prefix}peak tok/s:     {max(tok_rates):,}")


def compare_runs(run_data: list[tuple[str, list[dict]]]):
    """Side-by-side comparison of multiple runs."""
    if len(run_data) < 2:
        return

    print("\n\nRUN COMPARISON")
    print("=" * 60)

    for label, records in run_data:
        show_summary(records, label)

    # Best val loss comparison
    print("\n  Best val loss by run:")
    bests = []
    for label, records in run_data:
        val_losses = [r.get("val_loss") for r in records if r.get("val_loss") is not None]
        if val_losses:
            bests.append((min(val_losses), label, records[val_losses.index(min(val_losses))]))
    bests.sort(key=lambda x: x[0])
    for val_loss, label, rec in bests:
        print(f"    {label:40s}  {val_loss:.4f}  (step {rec['step']})")


def main():
    parser = argparse.ArgumentParser(
        description="View experiment logs (JSONL format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to log.jsonl file(s)")
    parser.add_argument("--last", type=int, default=0,
                        help="Show only the last N rows (default: all)")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary stats instead of table view")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple runs (summary side-by-side)")
    args = parser.parse_args()

    run_data: list[tuple[str, list[dict]]] = []
    for path in args.logs:
        if os.path.isdir(path):
            # Expand directory to log.jsonl inside
            inner = os.path.join(path, "log.jsonl")
            if os.path.exists(inner):
                path = inner
            else:
                print(f"skip {path}/ (no log.jsonl)")
                continue

        if not os.path.exists(path):
            print(f"skip {path} (not found)")
            continue

        records = load_log(path)
        if not records:
            print(f"skip {path} (no step records)")
            continue

        # Use directory name as label
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        run_data.append((label, records))
        print(f"\nloaded {path} ({len(records)} step records)")

    if not run_data:
        print("No valid log files found.")
        sys.exit(1)

    if args.compare or len(run_data) > 1:
        compare_runs(run_data)
    elif args.summary:
        for label, records in run_data:
            show_summary(records)
    else:
        for label, records in run_data:
            show_table(records, title=label, last=args.last)


if __name__ == "__main__":
    main()