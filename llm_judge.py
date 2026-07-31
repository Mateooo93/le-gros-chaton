"""LLM-as-judge grader for coding agent outputs.

Not every task has a unit test suite. Some need natural-language judging:
"did the agent actually fix the bug?" / "is the explanation correct?".
This reuses the frontier model as a judge — a small-model trick where the
verifier for hard-to-check tasks is a bigger model.

Usage:
    ANTHROPIC_API_KEY=... python llm_judge.py --task "fix the parser" --output /tmp/out.py
    ANTHROPIC_API_KEY=... python llm_judge.py --pair --code-a a.py --code-b b.py
    python llm_judge.py --json judge_results.json --summary
"""
import argparse
import json
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

JUDGE_TEMPLATE = """You are evaluating whether a solution correctly addresses a task.

TASK:
{task}

SOLUTION:
{solution}

Evaluate on:
1. Correctness — does it solve the stated problem?
2. Completeness — does it handle edge cases?
3. Code quality — is it clear, idiomatic, maintainable?

Respond with a JSON object:
{{"score": 0.0-1.0, "passed": true/false, "reason": "short justification"}}
"""


def judge(client, task: str, solution: str,
          model: str = "claude-sonnet-4-20250514",
          temperature: float = 0.0) -> dict:
    """Judge a solution against a task using the frontier model."""
    import json as _json
    import re

    prompt = JUDGE_TEMPLATE.format(task=task, solution=solution[:4000])
    resp = client.messages.create(
        model=model, max_tokens=512, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text

    # Extract JSON from response
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(0))
        except _json.JSONDecodeError:
            pass
    return {"score": 0.0, "passed": False, "reason": text[:200]}


def judge_batch(client, pairs: list[dict], model: str = "claude-sonnet-4-20250514") -> list[dict]:
    """Judge a batch of (task, solution) pairs."""
    results = []
    for i, pair in enumerate(pairs):
        task = pair.get("task", "")
        solution = pair.get("solution", "")
        result = judge(client, task, solution, model=model)
        result["task"] = task[:80]
        results.append(result)
        print(f"[judge] {i+1}/{len(pairs)} passed={result.get('passed')} "
              f"score={result.get('score', 0):.2f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge grader")
    parser.add_argument("--task", default=None, help="Task description")
    parser.add_argument("--solution", default=None, help="Solution text")
    parser.add_argument("--file", default=None, help="Solution file path")
    parser.add_argument("--json", dest="json_in", default=None,
                        help="JSON file with [{task, solution}] list")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--output", default="judge_results.json")
    args = parser.parse_args()

    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[judge] ERROR: Set ANTHROPIC_API_KEY env var")
        return
    client = anthropic.Anthropic(api_key=api_key)

    if args.json_in:
        with open(args.json_in) as f:
            pairs = json.load(f)
        results = judge_batch(client, pairs, model=args.model)
    elif args.task:
        solution = args.solution
        if args.file:
            with open(args.file) as f:
                solution = f.read()
        result = judge(client, args.task, solution or "", model=args.model)
        print(f"\nScore: {result.get('score', 0):.2f} | Passed: {result.get('passed')}")
        print(f"Reason: {result.get('reason', '')}")
        results = [result]
    else:
        print("[judge] Provide --task or --json")
        return

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[judge] Saved to {args.output}")


if __name__ == "__main__":
    main()
