"""Reasoning distillation — generate verified reasoning traces via a big model.

Applies the DeepSeek-R1-Distill finding from the SLM survey: small models learn
reasoning by training on big-model outputs. Uses Claude/Fable to generate
step-by-step reasoning + verified code solutions, then collects them into
SFT-ready chat data (same format as the Fable5 dataset).

Usage:
    ANTHROPIC_API_KEY=... python distill_reasoning.py --problems humaneval --limit 50
    ANTHROPIC_API_KEY=... python distill_reasoning.py --problems swebench --limit 20
    ANTHROPIC_API_KEY=... python distill_reasoning.py --output reasoning_data.json
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

REASONING_TEMPLATE = """You are solving a coding problem. Think step by step.

PROBLEM:
{prompt}

TESTS:
{tests}

Work through it carefully:
1. Understand the problem
2. Plan your approach
3. Write the solution
4. Check it against the tests mentally

Return ONLY this format:
<reasoning>
your step-by-step thinking here
</reasoning>
<solution>
```python
your code solution here
```
</solution>"""


def load_problems(source: str, limit: int | None = None) -> list:
    """Load problems from humaneval, swebench, or JSON."""
    if source == "humaneval":
        from eval.humaneval_loader import load as load_humaneval
        problems = load_humaneval(limit=limit)
        return [{"id": p.id, "prompt": p.prompt, "tests": p.tests,
                 "entry_point": p.entry_point} for p in problems]
    elif source == "swebench":
        from datasets import load_dataset
        print("[distill] Loading SWE-bench...")
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return [{"id": d.get("instance_id"), "prompt": d.get("problem_statement", ""),
                 "tests": d.get("FAIL_TO_PASS", ""), "entry_point": None}
                for d in ds]
    elif os.path.isfile(source):
        with open(source) as f:
            return json.load(f)[:limit] if limit else json.load(f)
    raise ValueError(f"Unknown source: {source}")


def extract_solution(text: str) -> str:
    """Extract code from the model response."""
    import re
    m = re.search(r'<solution>\s*```python\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def extract_reasoning(text: str) -> str:
    """Extract the reasoning block from the response."""
    import re
    m = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', text, re.DOTALL)
    return m.group(1) if m else ""


def distill(problems: list, out_path: str, model: str = "claude-sonnet-4-20250514",
            temperature: float = 0.3, max_tokens: int = 2048):
    """Generate verified reasoning traces via Claude API."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[distill] ERROR: Set ANTHROPIC_API_KEY env var")
        return

    client = anthropic.Anthropic(api_key=api_key)
    from verify.verifier import Problem, verify

    results = []
    for i, prob in enumerate(problems):
        pid = prob["id"]
        prompt = prob["prompt"]
        tests = prob["tests"]
        entry_point = prob.get("entry_point")

        print(f"[distill] {i+1}/{len(problems)} {pid}...")

        user_msg = REASONING_TEMPLATE.format(prompt=prompt, tests=tests)

        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text
            solution = extract_solution(raw)
            reasoning = extract_reasoning(raw)

            # Verify the solution
            p = Problem(id=str(pid), prompt=prompt, tests=tests, entry_point=entry_point)
            v = verify(p, solution)

            results.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"<reasoning>{reasoning}</reasoning>\n{solution}"},
                ],
                "problem_id": str(pid),
                "passed": v.passed,
                "n_pass": v.n_pass,
                "n_total": v.n_total,
                "has_reasoning": bool(reasoning),
            })

            status = "✓" if v.passed else "✗"
            print(f"  {status} {v.n_pass}/{v.n_total} passed, reasoning={'yes' if reasoning else 'no'}")
            time.sleep(0.5)  # rate limit

        except Exception as e:
            print(f"  ✗ Error: {e}")

    passed = sum(1 for r in results if r["passed"])
    with_reasoning = sum(1 for r in results if r["has_reasoning"])

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[distill] Saved {len(results)} examples to {out_path}")
    print(f"[distill]   {passed} verified passing ({100*passed/max(len(results),1):.0f}%)")
    print(f"[distill]   {with_reasoning} with reasoning traces")
    return results


def main():
    parser = argparse.ArgumentParser(description="Distill reasoning from big models")
    parser.add_argument("--problems", default="humaneval",
                        help="humaneval | swebench | path/to/json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="reasoning_data.json")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    print(f"[distill] Loading problems from {args.problems}...")
    problems = load_problems(args.problems, limit=args.limit)
    print(f"[distill] Loaded {len(problems)} problems")

    distill(problems, args.output, model=args.model, temperature=args.temperature)


if __name__ == "__main__":
    main()
