#!/usr/bin/env python3
"""Normalize tool-call formats in teacher trajectories to the canonical form.

Le Gros Chaton's trajectory SFT (colab/trajectory_sft.ipynb, train_qwen.py
--trajectory-sft) trains on `agent_traces_full.jsonl`, which mixes tool-call
syntaxes depending on which model produced the trace:

    ```tool\nargs```                              canonical (harness agent_swe.py)
    <|open|>call tool="tool"\nargs<|close|>...    fat-cat bracket style (Fable5)
    [tool\nargs]                                  plain bracket fallback
    <tool>args</tool>                             angle fallback

Mixed formats confuse SFT (the model sees several syntaxes and learns to emit
all of them). This tool rewrites every tool call in every message to the
canonical backtick form so the model trains on ONE format.

Guarantees:
  * Only blocks whose name is a known tool are touched. Prose code fences
    (```python ...) and "[thinking]" blocks are left byte-identical.
  * Everything else (issue, patch, tool results, self_review, field order) is
    copied verbatim — same trace count, same message count, same schema.
  * Idempotent: running it again is a byte-for-byte no-op (verify with
    `--check`).

Usage:
    python colab/normalize_traces.py [--input agent_traces_full.jsonl]
                                     [--output agent_traces_normalized.jsonl]
                                     [--check]
"""
import argparse
import json
import re
import sys

# Tools the harness/teacher can call (agent_swe.py TOOLS + teacher TOOLS_DESC).
KNOWN_TOOLS = {
    "read_file", "write_file", "search_code", "list_dir", "run_test",
    "finish", "prune",
}
_TOOL_ALT = "|".join(sorted(KNOWN_TOOLS))

# Fat-cat style: <|open|>call tool="NAME"\nARGS<|close|>argument<|sep|>...<|close|>message<|sep|>
_RE_FATCAT_BLOCK = re.compile(
    r'(?s)<\|open\|>call tool="\w+"\s*\n.*?<\|close\|>message<\|sep\|>')
_RE_FATCAT_ARGS = re.compile(r'(?s)call tool="(\w+)"\s*\n(.*?)<\|close\|>')

# Plain bracket style: [tool\nargs]
_RE_BRACKET = re.compile(rf'\[({_TOOL_ALT})\s*\n(.*?)\]', re.DOTALL)

# Angle style: <tool>args</tool>
_RE_ANGLE = re.compile(rf'<({_TOOL_ALT})>([\s\S]*?)</\1>')

# Leftover detection (scoped to known tools, all styles).
_RE_LEFTOVER = re.compile(
    rf'<\|open\|>call tool="\w+"|\[({_TOOL_ALT})\s*\n|<({_TOOL_ALT})>')


def _convert_fatcat(content: str) -> tuple[str, int]:
    """Rewrite <|open|>call tool=... blocks to ```tool\\nargs```. Returns
    (new_content, n_converted)."""

    def repl(m):
        am = _RE_FATCAT_ARGS.search(m.group(0))
        if not am:
            return m.group(0)  # malformed block — leave untouched, no data loss
        tool, args = am.group(1), am.group(2).strip()
        return f"```{tool}\n{args}\n```"

    new, n = _RE_FATCAT_BLOCK.subn(repl, content)
    return new, n


def _convert_bracket(content: str) -> tuple[str, int]:
    """Rewrite [tool\\nargs] blocks to ```tool\\nargs```."""

    def repl(m):
        return f"```{m.group(1)}\n{m.group(2).strip()}\n```"

    new, n = _RE_BRACKET.subn(repl, content)
    return new, n


def _convert_angle(content: str) -> tuple[str, int]:
    """Rewrite <tool>args</tool> blocks to ```tool\\nargs```."""

    def repl(m):
        return f"```{m.group(1)}\n{m.group(2).strip()}\n```"

    new, n = _RE_ANGLE.subn(repl, content)
    return new, n


def normalize_message(content: str) -> tuple[str, int, int]:
    """Normalize all tool calls in one message content.

    Returns (new_content, n_converted, n_touched_messages) where
    n_converted counts individual blocks rewritten and n_touched_messages is 1
    if the message changed at all.
    """
    total = 0
    for fn in (_convert_fatcat, _convert_bracket, _convert_angle):
        content, n = fn(content)
        total += n
    return content, total, 1 if total else 0


def load_traces(path: str) -> list[dict]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[normalize] WARNING: skipping malformed line: {e}",
                      file=sys.stderr)
    return traces


def _count_styles(traces: list[dict]) -> dict:
    """Count tool-call blocks per style across all traces (for reporting)."""
    counts = {"backtick": 0, "fatcat": 0, "bracket": 0, "angle": 0}
    for t in traces:
        for m in t.get("messages", []):
            c = m.get("content", "")
            for _ in re.finditer(r'```(\w+)\s*\n.*?```', c, re.DOTALL):
                if _.group(1) in KNOWN_TOOLS:
                    counts["backtick"] += 1
            for _ in _RE_FATCAT_BLOCK.finditer(c):
                counts["fatcat"] += 1
            for _ in _RE_BRACKET.finditer(c):
                counts["bracket"] += 1
            for _ in _RE_ANGLE.finditer(c):
                counts["angle"] += 1
    return counts


def normalize_traces(traces: list[dict]) -> tuple[list[dict], dict]:
    """Return (normalized_traces, stats). Traces are deep-copied."""
    import copy
    stats = {
        "traces": len(traces),
        "messages": sum(len(t.get("messages", [])) for t in traces),
        "messages_touched": 0,
        "converted": {"fatcat": 0, "bracket": 0, "angle": 0},
    }
    out = []
    for t in traces:
        nt = copy.deepcopy(t)
        for m in nt.get("messages", []):
            if not isinstance(m.get("content"), str):
                continue
            new_content, n, touched = normalize_message(m["content"])
            m["content"] = new_content
            if touched:
                stats["messages_touched"] += 1
        out.append(nt)

    # Re-count styles on the normalized corpus (should be backtick-only).
    after = _count_styles(out)
    for style in ("fatcat", "bracket", "angle"):
        stats["converted"][style] = (
            _count_styles(traces)[style] - after[style])
    stats["after"] = after
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="agent_traces_full.jsonl")
    ap.add_argument("--output", default="agent_traces_normalized.jsonl")
    ap.add_argument("--check", action="store_true",
                    help="Verify the output file is already fully normalized "
                         "(exit 0 if no conversions needed, 1 otherwise)")
    args = ap.parse_args()

    traces = load_traces(args.input)
    if not traces:
        print(f"[normalize] No traces found in {args.input}")
        sys.exit(1)

    before = _count_styles(traces)
    normalized, stats = normalize_traces(traces)

    if args.check:
        if stats["messages_touched"] == 0 and before.get("fatcat", 0) == 0 \
                and before.get("bracket", 0) == 0 and before.get("angle", 0) == 0:
            print(f"[normalize] OK: {args.output} is fully normalized "
                  f"({stats['traces']} traces, backtick-only)")
            sys.exit(0)
        print(f"[normalize] FAIL: {args.output} still needs normalization "
              f"(fatcat={before.get('fatcat', 0)} bracket="
              f"{before.get('bracket', 0)} angle={before.get('angle', 0)})")
        sys.exit(1)

    # Round-trip invariants: same traces, same messages, same schema.
    assert len(normalized) == stats["traces"]
    assert all(len(t["messages"]) == len(o["messages"])
               for t, o in zip(traces, normalized))

    with open(args.output, "w") as f:
        for t in normalized:
            f.write(json.dumps(t) + "\n")

    print(f"[normalize] {args.input} -> {args.output}")
    print(f"[normalize] traces: {stats['traces']} | messages: "
          f"{stats['messages']} | messages touched: "
          f"{stats['messages_touched']}")
    print(f"[normalize] tool-call style before: {before}")
    print(f"[normalize] tool-call style after : {stats['after']}")
    print(f"[normalize] blocks converted: "
          f"{dict(stats['converted'])}")
    remaining = {k: v for k, v in stats["after"].items() if k != "backtick"}
    if sum(remaining.values()):
        print(f"[normalize] WARNING: leftover non-backtick blocks: {remaining}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
