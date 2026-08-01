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

try:
    import torch
except ImportError:
    torch = None

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)


def load_model(model_name: str = "Qwen/Qwen3.5-9B", use_lora: bool = True):
    """Load Qwen3.5-9B in 4-bit QLoRA, optimized for T4/L4 16-24GB.

    Qwen/Qwen3.5-9B is a hybrid-attention vision-language model (model_type
    'qwen3_5', Qwen3_5ForConditionalGeneration) with mixed full-attention and
    linear-attention (gated-deltanet) layers, released only as a VLM checkpoint
    (weights stored under `model.language_model.*`). On transformers >= 5.14.1
    `AutoModelForCausalLM` resolves `qwen3_5` to the text-only
    `Qwen3_5ForCausalLM` ("VLM compatibility" mapping in modeling_auto.py) and
    `conversion_mapping.py` applies `PrefixChange(prefix_to_remove=
    "language_model")`, so the text weights load correctly while the vision
    encoder and MTP heads are dropped. We thus train a plain causal LM on text
    — no vision encoder in memory, no manual surgery. 4-bit QLoRA targets both
    attention flavors' linear projections plus the MLP.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

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

    # Sanity: confirm the hybrid-attention VLM config resolves to a text CausalLM.
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    print(f"[train] model_type={getattr(cfg, 'model_type', '?')} "
          f"text_model_type={getattr(text_cfg, 'model_type', '?')} "
          f"layers={getattr(text_cfg, 'num_hidden_layers', '?')}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if use_lora:
        # Skip prepare_model_for_kbit_training — it converts to fp32 causing OOM on T4.
        # 4-bit QLoRA doesn't need it for LoRA training.
        # target_modules spans both decoder block types:
        #   - full_attention (Qwen3_5Attention): q/k/v/o_proj
        #   - linear_attention (Qwen3_5GatedDeltaNet): in_proj_qkv/a/b/z + out_proj
        #   - MLP (Qwen3_5MLP): gate/up/down_proj
        # PEFT silently ignores target_modules that don't exist on a given layer.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z",
                "out_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print(f"[train] Loaded in {time.time() - t0:.1f}s")
    # Debug: print GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            used = torch.cuda.memory_allocated(i) / 1e9
            print(f"[train] GPU {i}: {used:.1f}GB / {mem:.1f}GB used")
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


def load_agent_traces(limit: int | None = None) -> list[dict]:
    """Load agent interaction traces (agent_traces.jsonl) as training data."""
    import json as _json

    path = os.path.join(PROJ_ROOT, "agent_traces.jsonl")
    if not os.path.exists(path):
        print("[train] No agent_traces.jsonl found — skipping")
        return []

    print(f"[train] Loading agent traces from {path}...")
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trace = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if trace.get("success"):
                # Convert successful traces to chat format
                steps = trace.get("steps", [])
                if steps:
                    convo = "\n".join(s.get("response", "") for s in steps)
                    examples.append({
                        "messages": [
                            {"role": "user", "content": trace.get("instance_id", "task")},
                            {"role": "assistant", "content": convo[:4000]},
                        ]
                    })
    if limit:
        examples = examples[:limit]
    print(f"[train] Loaded {len(examples)} trace examples")
    return examples


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
            max_length: int = 2048, resume_from_checkpoint: str | None = None):
    """Supervised fine-tuning on the Fable5 dataset.

    Saves a checkpoint every 20% of the run (plus the final step) and uploads
    each one to HF Hub so a disconnected Kaggle session never loses progress —
    the next run resumes from the latest uploaded checkpoint.
    """
    from transformers import (
        TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback,
    )
    import huggingface_hub as hf_hub

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

    # Steps per 20% checkpoint: effective batch = batch_size * grad_accum.
    n_examples = len(tokenized)
    eff_batch = batch_size * 8  # gradient_accumulation_steps fixed at 8 below
    total_steps = max(1, (n_examples + eff_batch - 1) // eff_batch) * epochs
    save_every = max(1, total_steps // 5)  # 5 checkpoints = every 20%
    print(f"[train] {n_examples} examples | eff_batch={eff_batch} "
          f"| {total_steps} steps | checkpoint every {save_every} steps (~20%)")

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_steps=100,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=save_every,
        save_total_limit=10,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Offloads optimizer states to CPU
    )

    # After every Trainer save, upload the checkpoint to HF Hub so it survives
    # a Kaggle disconnect. The next run resumes from the latest uploaded one.
    hf_token = os.environ.get("HF_TOKEN", "")
    ckpt_repo = None
    if hf_token:
        try:
            api = hf_hub.HfApi(token=hf_token)
            who = api.whoami()["name"]
            ckpt_repo = f"{who}/le-gros-chaton-qwen-sft-ckpt"
            api.create_repo(ckpt_repo, private=True, exist_ok=True)
            print(f"[train] Checkpoints will upload to HF Hub: {ckpt_repo}")
        except Exception as e:
            print(f"[train] HF Hub checkpoint upload disabled: {e}")
            ckpt_repo = None

    class HubUploadCallback(TrainerCallback):
        """Upload each saved checkpoint dir to the HF ckpt repo."""
        def on_save(self, args, state, control, **kwargs):
            if not ckpt_repo:
                return
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                return
            print(f"[train] Uploading {ckpt} -> {ckpt_repo} ...")
            try:
                hf_hub.upload_folder(
                    folder_path=ckpt, repo_id=ckpt_repo,
                    token=hf_token, ignore_patterns=["*.bin", "optimizer.pt"],
                )
                print(f"[train] ✓ Uploaded checkpoint-{state.global_step}")
            except Exception as e:
                print(f"[train] Upload failed (will retry next save): {e}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        callbacks=[HubUploadCallback()],
    )

    print(f"[train] Starting SFT ({epochs} epoch(s), {lr})...")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"[train] SFT completed in {(time.time()-t0)/60:.1f} min")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] Saved SFT model to {out_dir}")
    return out_dir


def train_rlvr(model, tokenizer, problems, out_dir: str = "qwen_rlvr",
               n_steps: int = 200, group_size: int = 4,
               batch_size: int = 2, max_new: int = 256,
               lr: float = 2e-5, save_every: int = 50):
    """GRPO with proportional rewards (verifier-based).

    Args:
        n_steps: Total training steps
        group_size: Samples per problem (GRPO group)
        batch_size: Problems per step (memory-limited on T4)
        max_new: Max tokens to generate per sample
        lr: Learning rate
        save_every: Checkpoint every N steps (Kaggle 9hr session safety)
    """
    from verify.verifier import Problem, verify
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=lr)
    device = model.device

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for prob in problems[:batch_size]:
            p = Problem(id=prob.id, prompt=prob.prompt,
                       tests=prob.tests, entry_point=prob.entry_point)

            # Generate G samples
            inputs = tokenizer(p.prompt, return_tensors="pt").to(device)
            prompts_exp = {k: v.repeat(group_size, 1) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **prompts_exp, max_new_tokens=max_new,
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

        # Checkpoint to survive Kaggle 9hr session limits
        if (step + 1) % save_every == 0:
            ckpt_dir = f"{out_dir}_step{step+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[train] Checkpoint saved to {ckpt_dir}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] Saved RLVR model to {out_dir}")


def resolve_sft_ckpt(resume: str | None) -> str | None:
    """Resolve --resume-sft to a local checkpoint dir.

    Accepts a local dir containing checkpoint-* subdirs, or an HF Hub repo id;
    for a repo, downloads it and picks the checkpoint-* dir with the highest
    step number (the one a disconnected run left behind). Returns None if
    nothing to resume from.
    """
    if not resume:
        return None
    # Local path with checkpoints?
    if os.path.isdir(resume):
        cks = sorted([d for d in os.listdir(resume)
                      if d.startswith("checkpoint-")],
                     key=lambda d: int(d.split("-")[1]))
        if cks:
            return os.path.join(resume, cks[-1])
        return resume
    # Treat as HF repo id: download, then find latest checkpoint-*
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=resume, token=os.environ.get("HF_TOKEN", ""),
            ignore_patterns=["*.bin", "optimizer.pt"],
        )
        cks = sorted([d for d in os.listdir(local)
                      if d.startswith("checkpoint-")],
                     key=lambda d: int(d.split("-")[1]))
        if cks:
            return os.path.join(local, cks[-1])
        return None
    except Exception as e:
        print(f"[train] Could not resolve SFT checkpoint {resume}: {e}")
        return None


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
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO samples per problem")
    parser.add_argument("--rlvr-lr", type=float, default=2e-5,
                        help="RLVR learning rate")
    parser.add_argument("--rlvr-max-new", type=int, default=256,
                        help="Max tokens per RLVR generation")
    parser.add_argument("--resume-rlvr", default=None,
                        help="Resume RLVR from this checkpoint dir")
    parser.add_argument("--resume-sft", default=None,
                        help="Resume SFT from a local checkpoint dir or an HF Hub "
                             "repo (auto-pulls the latest checkpoint-* subdir)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default="qwen_coding_agent")
    args = parser.parse_args()

    print(f"[train] === Le Gros Chaton - Coding Agent Training ===")
    print(f"[train] Model: {args.model}")
    print(f"[train] Dataset: {args.dataset}")
    print(f"[train] Output: {args.output}")

    # Load model (resume from RLVR checkpoint if requested)
    if args.resume_rlvr:
        from peft import PeftModel
        model, tokenizer = load_model(args.model)
        model = PeftModel.from_pretrained(model, args.resume_rlvr)
        model.eval()
        print(f"[train] Resumed RLVR adapter from {args.resume_rlvr}")
    else:
        model, tokenizer = load_model(args.model)

    if not args.rlvr_only:
        # Phase 1: SFT on Fable5 dataset
        print(f"\n{'='*50}")
        print(f"[train] PHASE 1: SFT on {args.dataset}")
        print(f"{'='*50}")
        fable = load_fable5_dataset(limit=args.limit)
        traces = load_agent_traces(limit=min(args.limit or 5000, 5000))
        dataset = fable
        if traces:
            print(f"[train] Merging {len(traces)} agent traces with Fable5 data")
            # Keep it simple: traces first, then Fable5
            dataset = {"messages": list(traces) + list(fable)}
        sft_path = train_sft(
            model, tokenizer, dataset,
            out_dir=f"{args.output}_sft",
            lr=args.lr, epochs=args.sft_epochs,
            batch_size=args.batch_size, max_length=args.max_length,
            resume_from_checkpoint=resolve_sft_ckpt(args.resume_sft),
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
            group_size=args.group_size,
            lr=args.rlvr_lr,
            max_new=args.rlvr_max_new,
        )

    # Final save
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n{'='*50}")
    print(f"[train] ✓ Model saved to {args.output}")
    print(f"[train] Run: python eval_qwen.py --ckpt {args.output}")

    # Upload to HuggingFace Hub (if HF_TOKEN is set)
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            repo_id = f"{api.whoami()['name']}/le-gros-chaton-qwen"
            api.create_repo(repo_id, private=True, exist_ok=True)
            api.upload_folder(folder_path=args.output, repo_id=repo_id)
            print(f"[train] ✓ Uploaded to HF Hub: {repo_id}")
        except Exception as e:
            print(f"[train] HF upload skipped: {e}")
    else:
        print(f"[train] Set HF_TOKEN env var to auto-upload to HF Hub")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
