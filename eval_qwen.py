"""Evaluate fine-tuned Qwen models on coding benchmarks.

Adapts our existing evaluation harness to work with HuggingFace models.
Supports HumanEval pass@k, agentic evaluation, and test-time compute scaling.

Usage:
    python eval_qwen.py --model Qwen/Qwen3.5-9B --ckpt qwen_rlvr_final
    python eval_qwen.py --model Qwen/Qwen3-32B --ckpt ./lora_adapters --mode humaneval --limit 10
    python eval_qwen.py --model Qwen/Qwen3.5-9B --mode agent --n-samples 16
"""
import argparse
import json
import os
import sys
import time

try:
    import torch
except ImportError:
    torch = None

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_qwen(model_name: str, ckpt_path: str | None = None, use_4bit: bool = False):
    """Load a Qwen model with optional LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path or model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt: str, max_new: int = 512,
                        temperature: float = 0.8, top_p: float = 0.95,
                        typical_p: float = 0.0) -> str:
    """Generate a single completion from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            temperature=temperature, top_p=top_p, do_sample=True,
            typical_p=typical_p if typical_p > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def evaluate_humaneval(model, tokenizer, limit: int = 50, n_samples: int = 20,
                       temperature: float = 0.8):
    """HumanEval pass@k evaluation."""
    from eval.humaneval_loader import load as load_humaneval
    from verify.verifier import Problem, verify
    import math

    problems = load_humaneval(limit=limit)
    print(f"[eval_qwen] Evaluating {len(problems)} HumanEval problems...")

    results = []
    for p in problems:
        task_id = p.id
        prompt = p.prompt
        tests = p.tests
        entry_point = p.entry_point

        passed = 0
        for s in range(n_samples):
            solution = generate_completion(
                model, tokenizer, prompt,
                max_new=256, temperature=temperature,
            )
            prob = Problem(id=task_id, prompt=prompt, tests=tests, entry_point=entry_point)
            v = verify(prob, solution)
            if v.passed:
                passed += 1

        # pass@1 estimate
        pass1 = 1.0 - math.comb(n_samples - passed, 1) / math.comb(n_samples, 1) if n_samples > 0 else 0.0
        results.append({"id": task_id, "pass@1": pass1, "passed": passed, "n": n_samples})

        if len(results) % 10 == 0:
            avg = sum(r["pass@1"] for r in results) / len(results)
            print(f"[eval_qwen] {len(results)}/{len(problems)}: avg pass@1={avg:.3f}")

    avg_pass1 = sum(r["pass@1"] for r in results) / len(results) if results else 0.0
    print(f"\n[eval_qwen] Final pass@1={avg_pass1:.3f} ({len(results)} problems)")
    return results


def evaluate_agent(model, tokenizer, problems: list, max_steps: int = 5,
                   n_samples: int = 1, verbose: bool = False):
    """Agentic evaluation using our agent loop adapted for Qwen."""
    from agent.loop import run as run_agent

    print(f"[eval_qwen] Agentic eval: {len(problems)} problems, {n_samples}x scaling")
    solved = 0
    total = 0

    for prob in problems:
        task = prob.get("prompt", prob.get("instruction", ""))
        best_solved = False

        for s in range(n_samples):
            temp = 0.7 + (s / max(n_samples - 1, 1)) * 0.3
            # Use Qwen for generation within the agent loop
            # (simplified: just try to solve with generate)
            solution = generate_completion(
                model, tokenizer, task,
                max_new=512, temperature=temp,
            )
            from verify.verifier import Problem, verify
            p = Problem(
                id=prob.get("id", "unknown"),
                prompt=prob.get("prompt", ""),
                tests=prob.get("tests", ""),
                entry_point=prob.get("entry_point"),
            )
            v = verify(p, solution)
            if v.passed:
                best_solved = True
                break

        if best_solved:
            solved += 1
        total += 1

        if verbose:
            print(f"  {'✓' if best_solved else '✗'} {prob.get('id', '?')} ({solved}/{total})")

    print(f"\n[eval_qwen] Agentic: {solved}/{total} solved ({100*solved/max(total,1):.1f}%)")
    return {"solved": solved, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen models on coding benchmarks")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="Base model ID")
    parser.add_argument("--ckpt", default=None, help="Fine-tuned checkpoint path")
    parser.add_argument("--mode", choices=["humaneval", "agent"], default="humaneval")
    parser.add_argument("--limit", type=int, default=50, help="Problem limit")
    parser.add_argument("--n-samples", type=int, default=20, help="Test-time scaling")
    parser.add_argument("--n-vote", type=int, default=1,
                        help="Verifier voting: sample N and verify each (default 1=off)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="4-bit loading")
    parser.add_argument("--output", default=None, help="Results JSON path")
    parser.add_argument("--record", action="store_true",
                        help="Record result to benchmark_tracker.py")
    parser.add_argument("--all-evals", action="store_true",
                        help="Run HumanEval + SWE-bench + tool-call evals, then record")
    args = parser.parse_args()

    print(f"[eval_qwen] Loading {args.model}...")
    model, tokenizer = load_qwen(args.model, args.ckpt, use_4bit=args.four_bit)
    print(f"[eval_qwen] Model loaded on {model.device}")

    if args.mode == "humaneval":
        if args.n_vote > 1:
            print(f"[eval_qwen] Using verifier voting with n={args.n_vote}")
            from vote_solutions import vote
            from eval.humaneval_loader import load as load_humaneval
            probs = load_humaneval(limit=args.limit)
            data = vote(probs, model, tokenizer,
                        n_samples=args.n_vote, device="cuda")
            results = [{"id": r["id"], "pass@1": 100.0 if r["solved"] else 0.0,
                        "passed": 1 if r["solved"] else 0, "n": 1}
                       for r in data["results"]]
        else:
            results = evaluate_humaneval(
                model, tokenizer, limit=args.limit,
                n_samples=args.n_samples, temperature=args.temperature,
            )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {args.output}")

        if args.record:
            from benchmark_tracker import add_result
            avg = sum(r["pass@1"] for r in results) / max(len(results), 1) * 100
            add_result("humaneval", round(avg, 1),
                       model=args.ckpt or args.model, n_samples=args.n_samples)

    elif args.mode == "agent":
        from eval.humaneval_loader import load as load_humaneval
        problems = load_humaneval(limit=args.limit)
        results = evaluate_agent(
            model, tokenizer, problems,
            n_samples=args.n_samples, verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
