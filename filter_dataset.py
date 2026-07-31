"""Data quality filtering for the Fable5 agentic coding dataset.

Filters out low-quality examples before SFT to improve training signal:
- Empty/trivial messages
- Extremely short assistant responses (< 100 chars)
- Repetitive content (same response repeated)
- Very long token counts that waste context

Usage:
    python filter_dataset.py                          # Filter full 160K dataset
    python filter_dataset.py --limit 5000             # Quick test on 5K rows
    python filter_dataset.py --output filtered.json   # Custom output
"""
import argparse
import json
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_dataset(limit: int | None = None):
    """Load the Fable5 dataset."""
    from datasets import load_dataset
    print("[filter] Loading Nexlab/fable5-agentic-coding-sft...")
    ds = load_dataset("Nexlab/fable5-agentic-coding-sft", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[filter] Loaded {len(ds)} rows")
    return ds


def message_stats(example: dict) -> dict:
    """Compute stats for a chat example."""
    messages = example.get("messages", [])
    n_msgs = len(messages)

    # Total content length
    total_chars = sum(len(m.get("content", "")) for m in messages)

    # Assistant content (the target output)
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assistant_chars = sum(len(m.get("content", "")) for m in assistant_msgs)

    # User content (the input)
    user_msgs = [m for m in messages if m.get("role") == "user"]
    user_chars = sum(len(m.get("content", "")) for m in user_msgs)

    return {
        "n_msgs": n_msgs,
        "total_chars": total_chars,
        "assistant_chars": assistant_chars,
        "user_chars": user_chars,
    }


def is_quality_example(example: dict, min_assistant_chars: int = 200,
                       max_assistant_chars: int = 20000,
                       min_msgs: int = 2) -> tuple[bool, str]:
    """Check if an example is high-quality.

    Returns (keep, reason_if_rejected).
    """
    stats = message_stats(example)

    # Too few messages
    if stats["n_msgs"] < min_msgs:
        return False, f"too_few_messages({stats['n_msgs']})"

    # Assistant response too short (trivial)
    if stats["assistant_chars"] < min_assistant_chars:
        return False, f"assistant_too_short({stats['assistant_chars']})"

    # Assistant response absurdly long (context waste)
    if stats["assistant_chars"] > max_assistant_chars:
        return False, f"assistant_too_long({stats['assistant_chars']})"

    # Check for repetitive content
    assistant_msgs = [m.get("content", "") for m in example.get("messages", [])
                      if m.get("role") == "assistant"]
    for i in range(len(assistant_msgs) - 1):
        a, b = assistant_msgs[i], assistant_msgs[i + 1]
        if a and b and a.strip() == b.strip():
            return False, "repetitive_assistant"

    # Check for near-identical consecutive messages (diff < 10%)
    for i in range(len(assistant_msgs) - 1):
        a, b = assistant_msgs[i], assistant_msgs[i + 1]
        if a and b and len(a) > 500 and len(b) > 500:
            # Simple overlap ratio
            common = set(a.split()).intersection(set(b.split()))
            overlap = len(common) / max(len(set(a.split())), 1)
            if overlap > 0.9:
                return False, "near_duplicate"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Filter Fable5 dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N rows (for quick testing)")
    parser.add_argument("--output", default="fable5_filtered",
                        help="Output dataset name/dir")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Min assistant response chars")
    parser.add_argument("--max-chars", type=int, default=20000,
                        help="Max assistant response chars")
    parser.add_argument("--save-json", action="store_true",
                        help="Also save as JSONL for inspection")
    args = parser.parse_args()

    ds = load_dataset(limit=args.limit)

    # Filter
    keep_indices = []
    reasons = {}
    for i, example in enumerate(ds):
        keep, reason = is_quality_example(
            example,
            min_assistant_chars=args.min_chars,
            max_assistant_chars=args.max_chars,
        )
        if keep:
            keep_indices.append(i)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    filtered = ds.select(keep_indices)

    print(f"\n[filter] Results:")
    print(f"  Total:      {len(ds)}")
    print(f"  Kept:       {len(filtered)} ({100*len(filtered)/max(len(ds),1):.1f}%)")
    print(f"  Removed:    {len(ds) - len(filtered)}")
    print(f"\n  Removal reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")

    # Compute stats on kept
    kept_stats = [message_stats(e) for e in filtered]
    avg_assistant = sum(s["assistant_chars"] for s in kept_stats) / max(len(kept_stats), 1)
    avg_msgs = sum(s["n_msgs"] for s in kept_stats) / max(len(kept_stats), 1)
    print(f"\n  Kept dataset stats:")
    print(f"    Avg assistant chars: {avg_assistant:.0f}")
    print(f"    Avg messages:        {avg_msgs:.1f}")

    # Save
    filtered.save_to_disk(args.output)
    print(f"\n[filter] Saved filtered dataset to {args.output}/")

    if args.save_json:
        with open(f"{args.output}.jsonl", "w") as f:
            for e in filtered:
                f.write(json.dumps(e) + "\n")
        print(f"[filter] Also saved JSONL to {args.output}.jsonl")


if __name__ == "__main__":
    main()
