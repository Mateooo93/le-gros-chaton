"""Plug-and-play Qwen fine-tuning with Fable5 dataset + RLVR.

One command to train the best possible coding agent on your L4 GPU:

    python train_qwen.py                          # Full pipeline (SFT + RLVR)
    python train_qwen.py --sft-only               # SFT only
    python train_qwen.py --dataset Nexlab/fable5-agentic-coding-sft

Downloads the Fable5 dataset, loads Qwen3.5-9B in 4-bit QLoRA, trains SFT
then optionally runs GRPO with our verifier for RLVR.  Optimized for L4 24GB.
"""
import argparse
import json
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_model(model_name: str = "Qwen/Qwen3.5-9B", use_lora: bool = True):
    """Load model with 4-bit QLoRA, optimized for L4 24GB."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[train] Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if use_lora:
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

    print(f"[train] Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def format_chat(example: dict) -> str:
    """Format a chat example into a training string."""
    messages = example.get("messages", example.get("conversations", []))
    if not messages:
        return ""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def load_fable5_dataset(limit: int | None = None) -> list[dict]:
    """Load the Fable5 agentic coding SFT dataset from HuggingFace."""
    from datasets import load_dataset

    print("[train] Loading Fable5 dataset (160k rows)...")
    ds = load_dataset("Nexlab/fable5-agentic-coding-sft", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[train] Loaded {len(ds)} examples")
    return ds


def train_sft(model, tokenizer, dataset, out_dir: str = "qwen_sft",
              lr: float = 2e-4, epochs: int = 1, batch_size: int = 4,
            max_length: int = 2048):
    """Supervised fine-tuning on the Fable5 dataset."""
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    import torch

    def tokenize_fn(examples):
        texts = [format_chat({"messages": m}) if isinstance(m, list) and len(m) > 0
                 else format_chat(m) if isinstance(m, dict) else ""
                 for m in examples["messages"]]
        texts = [t for t in texts if t]

        encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings

    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_steps=100,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print(f"[train] Starting SFT ({epochs} epoch(s), {lr})...")
    t0 = time.time()
    trainer.train()
    print(f"[train] SFT completed in {(time.time()-t0)/60:.1f} min")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] Saved SFT model to {out_dir}")
    return out_dir


def train_rlvr(model, tokenizer, problems, out_dir: str = "qwen_rlvr",
               n_steps: int = 200, group_size: int = 4):
    """GRPO with proportional rewards (verifier-based)."""
    from verify.verifier import Problem, verify
    from torch.optim import AdamW
    import torch

    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = model.device

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for prob in problems[:2]:  # batch_size=2 for memory
            p = Problem(id=prob.id, prompt=prob.prompt,
                       tests=prob.tests, entry_point=prob.entry_point)

            # Generate G samples
            inputs = tokenizer(p.prompt, return_tensors="pt").to(device)
            prompts_exp = {k: v.repeat(group_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **prompts_exp, max_new_tokens=256,
                    temperature=1.0, top_p=0.95, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Compute proportional rewards
            rewards = []
            for i in range(group_size):
                sol = tokenizer.decode(out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                v = verify(p, sol)
                r = v.n_pass / max(v.n_total, 1) if v.n_total > 0 else 0.0
                rewards.append(r)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float)
            adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            for i in range(group_size):
                if adv[i] <= 0:
                    continue
                sol_ids = out[i][inputs.input_ids.shape[1]:]
                if sol_ids.numel() == 0:
                    continue
                full_ids = torch.cat([inputs["input_ids"][i:i+1], sol_ids.unsqueeze(0)], dim=-1)
                labels = full_ids.clone()
                labels[:, :inputs["input_ids"].shape[1]] = -100
                outputs = model(full_ids, labels=labels)
                loss = -adv[i] * outputs.loss
                loss.backward()
                total_loss += loss.item()

        optimizer.step()
        if step % 20 == 0:
            print(f"[train] RLVR step {step}: loss={total_loss:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] Saved RLVR model to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the best coding agent")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", default="Nexlab/fable5-agentic-coding-sft")
    parser.add_argument("--sft-only", action="store_true", help="SFT only (skip RLVR)")
    parser.add_argument("--rlvr-only", action="store_true", help="RLVR only (skip SFT)")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset rows")
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rlvr-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", default="qwen_coding_agent")
    args = parser.parse_args()

    print(f"[train] === Le Gros Chaton - Coding Agent Training ===")
    print(f"[train] Model: {args.model}")
    print(f"[train] Dataset: {args.dataset}")
    print(f"[train] Output: {args.output}")

    # Load model
    model, tokenizer = load_model(args.model)

    if not args.rlvr_only:
        # Phase 1: SFT on Fable5 dataset
        print(f"\n{'='*50}")
        print(f"[train] PHASE 1: SFT on {args.dataset}")
        print(f"{'='*50}")
        dataset = load_fable5_dataset(limit=args.limit)
        sft_path = train_sft(
            model, tokenizer, dataset,
            out_dir=f"{args.output}_sft",
            lr=args.lr, epochs=args.sft_epochs,
            batch_size=args.batch_size, max_length=args.max_length,
        )

    if not args.sft_only:
        # Phase 2: RLVR with verifier rewards
        print(f"\n{'='*50}")
        print(f"[train] PHASE 2: RLVR with proportional rewards")
        print(f"{'='*50}")
        from eval.humaneval_loader import load as load_humaneval
        problems = load_humaneval(limit=10)

        train_rlvr(
            model, tokenizer, problems,
            out_dir=f"{args.output}_rlvr",
            n_steps=args.rlvr_steps,
        )

    # Final save
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n{'='*50}")
    print(f"[train] ✓ Model saved to {args.output}")
    print(f"[train] Run: python eval_qwen.py --ckpt {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
