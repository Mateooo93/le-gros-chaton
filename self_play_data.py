"""Self-Play Data Generation for Coding Agents (SSR-style).

Generates training data by having the model play both "challenger" (injects bugs)
and "solver" (fixes them) roles against a test suite oracle.

Usage:
    python self_play_data.py --ckpt model.pt --problems humaneval --n 100
    python self_play_data.py --ckpt model.pt --problems problems.json --out self_play.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Lazy imports (torch not available on dev VM)
# from model import GPT
# from tokenizer import encode, decode
# from verify.verifier import Problem, verify


def load_problems(source: str, limit: int | None = None) -> list[dict]:
    """Load coding problems. Supports 'humaneval' or JSON file path."""
    if source == "humaneval":
        from eval.humaneval_loader import load as load_humaneval
        problems = load_humaneval(limit=limit)
        return [{"id": p.id, "prompt": p.prompt, "tests": p.tests,
                  "entry_point": p.entry_point} for p in problems]
    elif os.path.isfile(source):
        with open(source) as f:
            problems = json.load(f)
        if limit:
            problems = problems[:limit]
        return problems
    else:
        raise ValueError(f"Unknown source: {source}")


def generate_self_play_data(
    problems: list[dict],
    model,
    out_path: str,
    n_attempts: int = 3,
    max_tokens: int = 512,
    temperature: float = 0.8,
) -> list[dict]:
    """Generate self-play training data.

    For each problem:
    1. Generate a solution with the model
    2. Verify it passes tests → keep as "correct" reference
    3. Have the model inject a bug into the correct solution
    4. Have the model fix the bugged version
    5. Verify the fix passes tests

    Returns list of (problem_id, solution, buggy_version, fixed_version, passes)
    """
    from verify.verifier import Problem, verify
    from tokenizer import encode, decode
    import torch

    device = next(model.parameters()).device
    results = []

    for prob in problems:
        pid = prob["id"]
        prompt = prob["prompt"]
        tests = prob["tests"]
        entry_point = prob.get("entry_point")

        p = Problem(id=pid, prompt=prompt, tests=tests, entry_point=entry_point)

        for attempt in range(n_attempts):
            try:
                # Step 1: Generate a solution
                if tokenizer is not None:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    out = model.generate(
                        **inputs, max_new_tokens=max_tokens,
                        temperature=temperature, top_p=0.95, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                else:
                    from tokenizer import encode, decode
                    input_ids = encode(prompt)
                    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
                    out = model.generate(inp, max_new_tokens=max_tokens,
                                         temperature=temperature, top_k=50,
                                         top_p=0.95, typical_p=0.2)
                    gen = decode(out[0].tolist())
                solution = gen.split("\nclass ")[0].split("\ndef ")[0]

                # Step 2: Verify the solution
                v = verify(p, solution)
                if not v.passed:
                    continue  # Skip solutions that don't pass

                # Step 3: Inject a bug (simplified: modify a return/value)
                buggy = _inject_bug(solution)

                # Step 4: Verify the buggy version fails
                v_bug = verify(p, buggy)
                if v_bug.passed:
                    continue  # Bug wasn't meaningful

                # Step 5: Generate a fix
                fix_prompt = f"Fix this buggy code:\\n\\n{buggy}\\n\\nTests:\\n{tests}\\n\\nFixed:\\n"
                fix_ids = encode(fix_prompt)
                fix_inp = torch.tensor([fix_ids], dtype=torch.long, device=device)
                fix_out = model.generate(fix_inp, max_new_tokens=max_tokens,
                                          temperature=0.6, top_k=50, top_p=0.95,
                                          typical_p=0.2)
                fix_gen = decode(fix_out[0].tolist())
                fix = fix_gen[len(fix_prompt):].split("\nclass ")[0].split("\ndef ")[0]

                # Step 6: Verify the fix
                v_fix = verify(p, fix)
                if not v_fix.passed:
                    continue

                results.append({
                    "problem_id": pid,
                    "solution": solution,
                    "buggy": buggy,
                    "fixed": fix,
                    "passes": True,
                })

                if len(results) % 10 == 0:
                    print(f"[self-play] {len(results)} examples generated...")

            except Exception as e:
                print(f"[self-play] Error on {pid} attempt {attempt}: {e}")
                continue

    # Save results
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[self-play] Saved {len(results)} examples to {out_path}")
    return results


def _inject_bug(code: str) -> str:
    """Inject a simple bug into a code solution.

    Strategies:
    - Flip a comparison operator (< → >, <= → >=)
    - Change a return value (return x → return x + 1)
    - Swap variable names
    - Remove a line
    """
    import random
    lines = code.split("\n")
    if len(lines) <= 2:
        return code + "\n    pass  # bug\n"

    # Try to find a line to modify
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "return " in stripped and i < len(lines) - 1:
            # Flip a return value
            lines[i] = line.replace("return ", "return ", 1) + " + 1"
            break
        if ">" in stripped and "<" not in stripped.replace(">", ""):
            lines[i] = line.replace(">", "<")
            break
        if "<" in stripped and ">" not in stripped.replace("<", ""):
            lines[i] = line.replace("<", ">")
            break
        if "==" in stripped:
            lines[i] = line.replace("==", "!=")
            break

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play training data for coding agents"
    )
    parser.add_argument("--ckpt", default="model.pt", help="Model checkpoint")
    parser.add_argument("--problems", default="humaneval",
                        help="Problem source ('humaneval' or JSON file)")
    parser.add_argument("--out", default="self_play_data.json",
                        help="Output path")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of problems to process")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Attempts per problem")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qwen", action="store_true",
                        help="Use Qwen HuggingFace model instead of custom GPT")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B",
                        help="HuggingFace model ID (when --qwen)")

    args = parser.parse_args()

    if args.dry_run:
        print(f"[self-play] DRY RUN: would process {args.n} problems from {args.problems}")
        print(f"[self-play] Would generate {args.n * args.attempts} solution attempts")
        print(f"[self-play] Output to: {args.out}")
        print(f"[self-play] Model checkpoint: {args.ckpt}")
        return

    print(f"[self-play] Loading problems from {args.problems}...")
    problems = load_problems(args.problems, limit=args.n)
    print(f"[self-play] Loaded {len(problems)} problems")

    # Load model
    import torch

    if args.qwen:
        print(f"[self-play] Loading Qwen model {args.model}...")
        from agent_qwen import load_qwen
        model, tokenizer = load_qwen(args.model, args.ckpt)
        device = model.device
    else:
        from model import GPT
        import config as cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[self-play] Loading model from {args.ckpt}...")
        model = GPT.from_checkpoint(args.ckpt, device=device)
        tokenizer = None
        model.eval()
    print(f"[self-play] Model loaded on {device}")

    generate_self_play_data(
        problems, model, args.out,
        n_attempts=args.attempts, temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
