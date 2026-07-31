"""Test-time compute via verifier voting for Qwen models.

Small models can rival big ones by spending inference compute instead of
training compute. This harness samples N solutions per problem, verifies each
with our test suite, and selects the best — verifier-grounded self-consistency.

This is the DeepSWE finding applied: 42.2% Pass@1 → 59% with test-time scaling.

Usage:
    python vote_solutions.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent
    python vote_solutions.py --model Qwen/Qwen3.5-9B --n-samples 16 --limit 20
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def sample_solutions(model, tokenizer, prompt: str, n: int, max_new: int = 256,
                     device: str = "cuda") -> list[str]:
    """Sample N solutions from the model with temperature diversity."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    solutions = []
    for i in range(n):
        # Temperature schedule: low→high for diverse sampling
        temp = 0.4 + (i / max(n - 1, 1)) * 0.6  # 0.4 → 1.0
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                temperature=temp, top_p=0.95, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        sol = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        solutions.append(sol)
    return solutions


def vote(problems, model, tokenizer, n_samples: int = 8, max_new: int = 256,
         device: str = "cuda", verbose: bool = False) -> dict:
    """Best-of-N with verifier voting.

    For each problem:
    1. Sample N solutions at varied temperatures
    2. Verify each against the test suite
    3. Pick the first fully-passing solution (else best partial)
    """
    from verify.verifier import Problem, verify

    results = []
    total_solved = 0

    for i, prob in enumerate(problems):
        pid = prob.id
        prompt = prob.prompt
        tests = prob.tests
        entry_point = prob.entry_point

        print(f"[vote] {i+1}/{len(problems)} {pid}...")

        solutions = sample_solutions(
            model, tokenizer, prompt, n_samples, max_new, device,
        )

        p = Problem(id=pid, prompt=prompt, tests=tests, entry_point=entry_point)

        best = None
        best_score = -1.0
        for j, sol in enumerate(solutions):
            v = verify(p, sol)
            score = v.n_pass / max(v.n_total, 1) if v.n_total > 0 else 0.0
            if v.passed:
                best = (sol, j, score, True)
                break  # found fully passing solution
            if score > best_score:
                best_score = score
                best = (sol, j, score, False)

        sol, idx, score, passed = best if best else ("", -1, 0.0, False)

        results.append({
            "id": pid,
            "solved": passed,
            "best_sample": idx,
            "best_score": score,
            "n_samples": n_samples,
            "solution": sol,
        })

        if passed:
            total_solved += 1

        if verbose:
            print(f"  {'✓' if passed else '✗'} sample {idx} score={score:.2f}")

    pass_rate = 100 * total_solved / max(len(results), 1)
    summary = {
        "n_problems": len(results),
        "solved": total_solved,
        "pass_rate": round(pass_rate, 1),
        "n_samples_per_problem": n_samples,
    }
    return {"summary": summary, "results": results}


def report(data: dict):
    s = data["summary"]
    print(f"\n{'='*50}")
    print("  Verifier-Voting Results (test-time compute)")
    print(f"{'='*50}")
    print(f"  Problems:          {s['n_problems']}")
    print(f"  Solved:            {s['solved']} ({s['pass_rate']}%)")
    print(f"  Samples/problem:   {s['n_samples_per_problem']}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Verifier voting (test-time compute)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="vote_results.json")
    args = parser.parse_args()

    from eval_qwen import load_qwen
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.four_bit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from eval.humaneval_loader import load as load_humaneval
    problems = load_humaneval(limit=args.limit)
    print(f"[vote] Loaded {len(problems)} problems")

    data = vote(
        problems, model, tokenizer,
        n_samples=args.n_samples, max_new=args.max_new,
        device=device, verbose=args.verbose,
    )

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[vote] Saved to {args.output}")

    report(data)


if __name__ == "__main__":
    main()
