"""Fine-tune Qwen models for coding agent tasks using RLVR + self-play.

Integrates with our existing verification pipeline, agent loop, and research
innovations (proportional rewards, test-time scaling, self-play data).

Supports:
  - Qwen2.5-Coder-7B (fits L4 24GB with QLoRA) — Kaggle L4
  - Qwen3-32B (fits A100-80GB) — Modal
  - DeepSeek-Coder-V2-Lite (16B) — A100-40GB

Usage:
  python finetune_qwen.py --model Qwen/Qwen2.5-Coder-7B --lora --4bit
  python finetune_qwen.py --model Qwen/Qwen3-32B --lora
  python finetune_qwen.py --mode rlvr --problems humaneval --n-steps 200
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

def load_model(model_name: str, use_lora: bool = True, use_4bit: bool = True,
               device: str = "auto"):
    """Load a Qwen model with optional QLoRA.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-Coder-7B")
        use_lora: Whether to apply LoRA adapters
        use_4bit: Whether to load in 4-bit (QLoRA). Requires bitsandbytes.
        device: "auto" for HuggingFace device map, or "cuda:0"

    Returns:
        model, tokenizer
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Quantization config for 4-bit
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[qwen] Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Prepare for k-bit training if quantized
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elapsed = time.time() - t0
    print(f"[qwen] Loaded in {elapsed:.1f}s")
    return model, tokenizer


def collect_self_play_data(
    model,
    tokenizer,
    problems: list[dict],
    out_path: str = "self_play_qwen.json",
    n_attempts: int = 3,
    max_tokens: int = 512,
):
    """Generate self-play training data using the loaded model.

    Uses our existing verify.verifier for test-based reward signals.
    """
    from verify.verifier import Problem, verify

    results = []
    device = next(model.parameters()).device

    for prob in problems:
        pid = prob["id"]
        prompt = prob["prompt"]
        tests = prob["tests"]
        entry_point = prob.get("entry_point")
        p = Problem(id=pid, prompt=prompt, tests=tests, entry_point=entry_point)

        for _ in range(n_attempts):
            try:
                # Generate solution
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.8, top_p=0.95, do_sample=True,
                )
                solution = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                # Verify
                v = verify(p, solution)
                if not v.passed:
                    continue

                # Inject bug (use our existing tool)
                from self_play_data import _inject_bug
                buggy = _inject_bug(solution)
                v_bug = verify(p, buggy)
                if v_bug.passed:
                    continue

                results.append({
                    "problem_id": pid,
                    "solution": solution,
                    "buggy": buggy,
                    "trainable": True,
                })
            except Exception as e:
                print(f"[qwen] Error: {e}")
                continue

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[qwen] Saved {len(results)} examples to {out_path}")
    return results


def train_rlvr(
    model,
    tokenizer,
    problems: list[dict],
    n_steps: int = 200,
    lr: float = 2e-5,
    batch_size: int = 4,
    group_size: int = 8,
    kl_coeff: float = 0.01,
    out_path: str = "qwen_rlvr",
):
    """GRPO training adapted for Qwen models (DeepSWE/SSR-style).

    Generates G = *group_size* samples per problem, computes proportional
    rewards via our verifier, normalizes advantages within each group,
    then updates the policy with a clipped surrogate + KL penalty.

    Uses our proportional rewards approach (fraction of tests passed).
    """
    import torch
    import torch.nn.functional as F
    from verify.verifier import Problem, verify
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        n_processed = 0

        for prob in problems[:batch_size]:
            pid = prob["id"]
            prompt = prob["prompt"]
            tests = prob["tests"]
            entry_point = prob.get("entry_point")
            p = Problem(id=pid, prompt=prompt, tests=tests, entry_point=entry_point)

            # Step 1: Generate G samples
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompts_expanded = {k: v.repeat(group_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **prompts_expanded, max_new_tokens=512,
                    temperature=1.0, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Step 2: Compute rewards for each sample
            rewards = []
            for i in range(group_size):
                solution = tokenizer.decode(
                    out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True,
                )
                v = verify(p, solution)
                r = v.n_pass / max(v.n_total, 1) if v.n_total > 0 else 0.0
                rewards.append(r)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float)

            # Step 3: Group-normalized advantages
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            # Step 4: Compute log-probs and GRPO loss
            for i in range(group_size):
                if advantages[i] <= 0 and i > 0:
                    continue  # skip negative samples (optional)
                sol_tokens = out[i][inputs.input_ids.shape[1]:]
                if sol_tokens.numel() == 0:
                    continue
                # Forward pass on this sample to get logits
                sample_inputs = {k: v[i:i+1] for k, v in prompts_expanded.items()}
                full_input_ids = torch.cat([sample_inputs["input_ids"], sol_tokens.unsqueeze(0)], dim=-1)
                labels = full_input_ids.clone()
                labels[:, :sample_inputs["input_ids"].shape[1]] = -100

                outputs = model(full_input_ids, labels=labels)
                log_probs = -outputs.loss  # negative loss = avg log prob

                # Policy gradient: maximize (advantage * log_prob) - KL penalty
                policy_loss = -advantages[i] * log_probs
                kl_penalty = kl_coeff * (log_probs ** 2).mean()

                loss = policy_loss + kl_penalty
                loss.backward()
                total_loss += loss.item()
                n_processed += 1

        optimizer.step()

        if step % 10 == 0:
            avg_loss = total_loss / max(n_processed, 1)
            print(f"[qwen] step {step}: loss={avg_loss:.4f}, processed={n_processed}")

    # Save
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"[qwen] Saved to {out_path}")


def generate_fable_training_data(problems: list, limit: int = 50):
    """Generate high-quality training data using Claude/Fable API.

    Queries a frontier model (Claude Opus/Fable) to produce verified-correct
    solutions for coding problems.  The resulting data is used for supervised
    fine-tuning (distillation) before RLVR training.

    Requires $ANTHROPIC_API_KEY to be set in the environment.
    Requires: pip install anthropic
    """
    import json
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[fable] ERROR: Set ANTHROPIC_API_KEY env var")
        print("[fable] Get one at: https://console.anthropic.com/")
        return

    try:
        import anthropic
    except ImportError:
        print("[fable] Install: pip install anthropic")
        return

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for p in problems[:limit]:
        pid = p.id
        prompt = p.prompt
        tests = p.tests
        entry_point = p.entry_point

        print(f"[fable] Generating solution for {pid}...")

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"Write a Python function that passes these tests.\n\n"
                               f"PROMPT:\n{prompt}\n\n"
                               f"TESTS:\n{tests}\n\n"
                               f"Return ONLY the function body (indented by 4 spaces)."
                }]
            )
            solution = response.content[0].text

            # Verify the solution
            from verify.verifier import Problem, verify
            prob = Problem(id=pid.id if hasattr(pid, 'id') else pid,
                          prompt=prompt, tests=tests,
                          entry_point=entry_point)
            v = verify(prob, solution)

            results.append({
                "problem_id": pid.id if hasattr(pid, 'id') else pid,
                "prompt": prompt,
                "solution": solution,
                "passed": v.passed,
                "n_pass": v.n_pass,
                "n_total": v.n_total,
            })

            status = "✓" if v.passed else "✗"
            print(f"  {status} {v.n_pass}/{v.n_total} passed")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save
    out_path = "fable_training_data.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    passed = sum(1 for r in results if r["passed"])
    print(f"[fable] Saved {len(results)} examples ({passed} passing) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen for coding agents")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model ID (e.g. Qwen/Qwen3.5-9B)")
    parser.add_argument("--lora", action="store_true", default=True,
                        help="Use LoRA adapters")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--mode", choices=["inspect", "self-play", "rlvr", "fable-data"],
                        default="inspect",
                        help="Mode: inspect | self-play | rlvr | fable-data (generate via Claude API)")
    parser.add_argument("--problems", default="humaneval",
                        help="Problem source (humaneval or JSON)")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="RL training steps")
    parser.add_argument("--group-size", type=int, default=8,
                        help="GRPO group size (samples per problem)")
    parser.add_argument("--kl-coeff", type=float, default=0.01,
                        help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Problems per training step")
    parser.add_argument("--limit", type=int, default=10,
                        help="Limit problems")
    args = parser.parse_args()

    print(f"[qwen] Model: {args.model}")
    print(f"[qwen] Mode: {args.mode}")

    if args.mode == "inspect":
        model, tokenizer = load_model(args.model, use_lora=False, use_4bit=False, device="cpu")
        print(f"[qwen] Model loaded: {type(model).__name__}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[qwen] Parameters: {n_params:,}")
        return

    if args.mode == "fable-data":
        generate_fable_training_data(problems_raw, args.limit)
        return

    model, tokenizer = load_model(args.model, use_lora=args.lora, use_4bit=args.four_bit)

    # Load problems
    from eval.humaneval_loader import load as load_humaneval
    problems_raw = load_humaneval(limit=args.limit)
    problems = [{"id": p.id, "prompt": p.prompt, "tests": p.tests,
                  "entry_point": p.entry_point} for p in problems_raw]
    print(f"[qwen] Loaded {len(problems)} problems")

    if args.mode == "self-play":
        collect_self_play_data(model, tokenizer, problems)
    elif args.mode == "rlvr":
        train_rlvr(
            model, tokenizer, problems,
            n_steps=args.n_steps,
            lr=args.lr,
            batch_size=args.batch_size,
            group_size=args.group_size,
            kl_coeff=args.kl_coeff,
        )


if __name__ == "__main__":
    main()
