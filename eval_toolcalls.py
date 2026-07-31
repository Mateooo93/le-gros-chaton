"""Tool-call format evaluation — measures the model's agentic precision.

The SLM survey found tool-calling precision is THE differentiator for small
agentic models. This harness checks whether the model emits correctly-formatted
tool calls, measures format accuracy, and reports failure modes.

Usage:
    python eval_toolcalls.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent
    python eval_toolcalls.py --model Qwen/Qwen3.5-9B --limit 20 --verbose
"""
import argparse
import json
import os
import re
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

TOOL_DESC = """You are a coding agent with these tools:
  read_file: read a file from the repo
  search_code: search for a pattern
  run_test: run a test
  list_dir: list files in a directory
  finish: submit your answer

To call a tool, output exactly:
```tool_name
arguments here
```"""

PROMPTS = [
    "Read the file main.py and search for the bug in the parser.",
    "List the files in src/ and read the first one.",
    "Run the tests to see what's failing.",
    "Search for 'TODO' across the codebase, then read the most relevant file.",
    "I need to fix a bug. Start by exploring the repo structure.",
]


def build_prompt(task: str) -> str:
    return f"{TOOL_DESC}\n\nTask: {task}\n\nRespond with tool calls:"


def parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse tool calls from model output. Returns (tool, args) list."""
    calls = []
    for m in re.finditer(r'```(\w+)\s*\n(.*?)```', text, re.DOTALL):
        calls.append((m.group(1), m.group(2).strip()))
    return calls


def evaluate_toolcalls(model, tokenizer, n_prompts: int = 10,
                       device: str = "cuda", verbose: bool = False) -> dict:
    """Evaluate tool-call format accuracy."""
    import torch

    results = {
        "total": 0,
        "valid_format": 0,
        "valid_tool": 0,
        "multiple_calls": 0,
        "empty": 0,
        "invalid_tools": {},
        "samples": [],
    }

    valid_tools = {"read_file", "search_code", "run_test", "list_dir", "finish"}

    for i in range(n_prompts):
        task = PROMPTS[i % len(PROMPTS)]
        prompt = build_prompt(task)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128,
                temperature=0.3, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        calls = parse_tool_calls(text)
        results["total"] += 1

        sample = {"task": task[:60], "output": text[:200], "calls": [c[0] for c in calls]}

        if not calls:
            results["empty"] += 1
            sample["valid"] = False
            sample["reason"] = "no_tool_calls"
            if verbose:
                print(f"✗ No tool calls: {text[:100]!r}")
        else:
            # Format is valid (at least one ```tool\nargs``` block)
            results["valid_format"] += 1
            if len(calls) > 1:
                results["multiple_calls"] += 1

            all_valid = all(c[0] in valid_tools for c in calls)
            if all_valid:
                results["valid_tool"] += 1
                sample["valid"] = True
            else:
                for c in calls:
                    if c[0] not in valid_tools:
                        results["invalid_tools"][c[0]] = results["invalid_tools"].get(c[0], 0) + 1
                sample["valid"] = False
                sample["reason"] = "unknown_tool"

            if verbose:
                status = "✓" if all_valid else "✗"
                print(f"{status} calls={[c[0] for c in calls]}")

        results["samples"].append(sample)

    # Aggregate
    total = max(results["total"], 1)
    results["format_accuracy"] = round(100 * results["valid_format"] / total, 1)
    results["tool_accuracy"] = round(100 * results["valid_tool"] / total, 1)
    return results


def report(results: dict):
    print(f"\n{'='*50}")
    print("  Tool-Call Evaluation")
    print(f"{'='*50}")
    print(f"  Prompts tested:        {results['total']}")
    print(f"  Valid format:          {results['valid_format']} ({results['format_accuracy']}%)")
    print(f"  Valid tool names:      {results['valid_tool']} ({results['tool_accuracy']}%)")
    print(f"  Multiple calls/turn:   {results['multiple_calls']}")
    print(f"  Empty (no calls):      {results['empty']}")
    if results["invalid_tools"]:
        print(f"  Unknown tools used:")
        for tool, count in sorted(results["invalid_tools"].items(), key=lambda x: -x[1]):
            print(f"    '{tool}': {count}x")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Tool-call format evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--limit", type=int, default=10, help="Prompts to test")
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="toolcall_results.json")
    args = parser.parse_args()

    from eval_qwen import load_qwen
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.four_bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = evaluate_toolcalls(
        model, tokenizer, n_prompts=args.limit,
        device=device, verbose=args.verbose,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[toolcall] Saved to {args.output}")

    report(results)


if __name__ == "__main__":
    main()
