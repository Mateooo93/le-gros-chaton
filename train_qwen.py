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


def load_model(model_name: str = "Qwen/Qwen3.5-9B", use_lora: bool = True,
               adapter: str | None = None):
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

    # Attach a previously trained LoRA adapter (e.g. the Phase 1 SFT adapter)
    # so Phase 2 RLVR starts from it. adapter can be a local dir or HF repo id.
    if adapter:
        if not use_lora:
            raise ValueError("adapter requires use_lora=True")
        if os.path.isdir(adapter):
            adapter_id = adapter
        else:
            try:
                from huggingface_hub import snapshot_download
                adapter_id = snapshot_download(
                    repo_id=adapter, token=os.environ.get("HF_TOKEN", ""))
            except Exception as e:
                raise RuntimeError(f"Could not download adapter {adapter}: {e}")
        print(f"[train] Loading SFT adapter from {adapter_id}")
        model.load_adapter(adapter_id, adapter_name="sft")
        model.set_adapter("sft")
        # Drop the untrained random adapter so training/evals only use the SFT one.
        try:
            model.delete_adapter("default")
        except Exception:
            pass
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


def format_trajectory(messages: list[dict]) -> str:
    """Format a full agent trajectory (from gen_trajectories.py) as text.

    Tool calls/results are user-role turns; the model's decisions are
    assistant-role turns. Mirrors format_chat but preserves the exact
    trajectory order.
    """
    if not messages:
        return ""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def load_agent_traces_full(limit: int | None = None) -> list[dict]:
    """Load full agent trajectories from agent_traces_full.jsonl (gen_trajectories)."""
    path = os.path.join(PROJ_ROOT, "agent_traces_full.jsonl")
    if not os.path.exists(path):
        print("[train] No agent_traces_full.jsonl — skipping")
        return []
    print(f"[train] Loading full trajectories from {path}")
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tr = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Keep verified trajectories (real training signal)
            if tr.get("verified"):
                out.append(tr)
    if limit:
        out = out[:limit]
    print(f"[train] Loaded {len(out)} verified trajectories")
    return out


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


def load_fable5_dataset(limit: int | None = None, start: int = 0) -> list[dict]:
    """Load the Fable5 agentic coding SFT dataset from HuggingFace.

    Args:
        limit: Max rows to load (None = all 160k)
        start: Row offset — used to continue training on the tail of the
            dataset after a smaller run already trained rows [0, start).
    """
    from datasets import load_dataset

    print("[train] Loading Fable5 dataset (160k rows)...")
    ds = load_dataset("Nexlab/fable5-agentic-coding-sft", split="train")
    n = len(ds)
    end = min(limit or n, n)
    if start >= end:
        raise ValueError(f"start={start} >= end={end} — nothing left to train on")
    if start > 0:
        print(f"[train] Skipping rows [0, {start}) — continuing from row {start}")
    ds = ds.select(range(start, end))
    print(f"[train] Loaded {len(ds)} examples (rows {start}-{end} of {n})")
    return ds


def train_sft(model, tokenizer, dataset, out_dir: str = "qwen_sft",
              lr: float = 2e-4, epochs: int = 1, batch_size: int = 4,
            max_length: int = 2048, resume_from_checkpoint: str | None = None,
            start: int = 0, trajectory: bool = False) -> tuple[str, int]:
    """Supervised fine-tuning on the Fable5 dataset (or agent trajectories).

    Saves a checkpoint every 20% of the run (plus the final step) and uploads
    each one to HF Hub so a disconnected Kaggle session never loses progress —
    the next run resumes from the latest uploaded checkpoint.

    If trajectory=True, the dataset is a list of full agent traces (from
    gen_trajectories.py) and loss is computed on ASSISTANT tokens only — the
    model learns to make the right tool call/decision given the history,
    without being rewarded for copying the tool outputs (research-backed,
    OmniCoder-style). Requires a longer max_length (e.g. 8192+).

    Returns (out_dir, n_rows_trained) where n_rows_trained is the number of
    dataset rows actually consumed (start + steps*eff_batch, capped), so the
    next stage (Modal 160k) can continue from the exact row offset.
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

    def tokenize_trajectory_fn(examples):
        """Tokenize trajectories with assistant-token-only loss masking.

        Every message is formatted with role tags; labels for non-assistant
        tokens (system, user/tool results) are set to -100 so the model only
        learns to produce its own decisions/tool calls.
        """
        batch_texts, batch_labels = [], []
        for m in examples["messages"]:
            if not m:
                continue
            text = format_trajectory(m)
            if not text:
                continue
            # Tokenize the full formatted trajectory (keep input_ids as ground truth)
            enc = tokenizer(text, truncation=True, max_length=max_length)
            ids = enc["input_ids"]
            labels = [-100] * len(ids)
            # Re-tokenize the same string but now find assistant spans: instead of
            # offset mapping, do a cheap pass: tokenize per-message and merge.
            merged_ids, merged_labels = [], []
            for msg in m:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    chunk = f"<|im_start|>system\n{content}<|im_end|>\n"
                    train = False
                elif role == "user":
                    chunk = f"<|im_start|>user\n{content}<|im_end|>\n"
                    train = False
                else:  # assistant
                    chunk = f"<|im_start|>assistant\n{content}<|im_end|>\n"
                    train = True
                enc_chunk = tokenizer(chunk, add_special_tokens=False)
                merged_ids.extend(enc_chunk["input_ids"])
                merged_labels.extend([c if train else -100 for c in enc_chunk["input_ids"]])
                if len(merged_ids) >= max_length:
                    break
            merged_ids = merged_ids[:max_length]
            merged_labels = merged_labels[:max_length]
            batch_texts.append(merged_ids)
            batch_labels.append(merged_labels)

        # Pad to max_length for a rectangular batch
        import torch as _t
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        max_len = max((len(t) for t in batch_texts), default=max_length)
        max_len = min(max_len, max_length)
        ids_t = _t.full((len(batch_texts), max_len), pad_id, dtype=_t.long)
        lab_t = _t.full((len(batch_texts), max_len), -100, dtype=_t.long)
        att_t = _t.zeros((len(batch_texts), max_len), dtype=_t.long)
        for i, (ids, labs) in enumerate(zip(batch_texts, batch_labels)):
            ids_t[i, :len(ids)] = _t.tensor(ids, dtype=_t.long)
            lab_t[i, :len(labs)] = _t.tensor(labs, dtype=_t.long)
            att_t[i, :len(ids)] = 1
        return {"input_ids": ids_t, "attention_mask": att_t, "labels": lab_t}

    if trajectory:
        tokenize_fn = tokenize_trajectory_fn

    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Steps per 20% checkpoint: effective batch = batch_size * grad_accum * n_gpus.
    n_dev = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 1
    n_examples = len(tokenized)
    eff_batch = batch_size * 8 * n_dev  # gradient_accumulation_steps fixed at 8 below
    total_steps = max(1, (n_examples + eff_batch - 1) // eff_batch) * epochs
    save_every = max(1, total_steps // 5)  # 5 checkpoints = every 20%
    print(f"[train] {n_examples} examples | {n_dev} GPU(s) | eff_batch={eff_batch} "
          f"| {total_steps} steps | checkpoint every {save_every} steps (~20%)")

    # --- Resume fix: if we're continuing from a checkpoint, the Trainer
    # restores global_step from it, but max_steps is computed from the
    # REMAINING dataset only. If global_step > max_steps (e.g. resume at 6200
    # of a 3799-step tail), HF Trainer concludes "already done" and exits in
    # 0 seconds — silently skipping the rest of training (this bit us).
    # Solution: set max_steps to the ABSOLUTE endpoint (resume_step + total).
    max_steps_arg = None
    resume_global_step = 0
    if resume_from_checkpoint:
        try:
            import json as _json
            with open(os.path.join(resume_from_checkpoint, "trainer_state.json")) as f:
                _st = _json.load(f)
            resume_global_step = int(_st.get("global_step", 0))
            if resume_global_step >= resume_global_step + total_steps - 1:
                # Sanity guard: never exit early just because the counter looks done.
                pass
        except Exception as e:
            print(f"[train] could not read resume checkpoint state: {e}")
        max_steps_arg = resume_global_step + total_steps
        print(f"[train] Resuming at global_step={resume_global_step}; "
              f"absolute max_steps={max_steps_arg} (remaining {total_steps})")

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        warmup_steps=100,
        num_train_epochs=epochs,
        max_steps=max_steps_arg,
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
        """Upload each saved checkpoint dir to the HF ckpt repo, and force a
        save at least every SAVE_FLOOR_SEC so a slow run never goes hours
        without a checkpoint (Kaggle disconnects lose progress otherwise)."""
        SAVE_FLOOR_SEC = 3600  # safety net: checkpoint at least every hour

        def __init__(self):
            self.last_save_t = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.last_save_t >= self.SAVE_FLOOR_SEC:
                control.should_save = True  # forces _save_checkpoint this step

        def on_save(self, args, state, control, **kwargs):
            self.last_save_t = time.time()
            if not ckpt_repo:
                return
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                return
            print(f"[train] Uploading {ckpt} -> {ckpt_repo} ...")
            try:
                hf_hub.upload_folder(
                    folder_path=ckpt, repo_id=ckpt_repo,
                    path_in_repo=f"checkpoint-{state.global_step}",
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

    # Rows actually consumed by this run (for the Modal continuation offset).
    steps_done = trainer.state.global_step
    rows_trained = min(start + steps_done * eff_batch, n_examples + start)
    print(f"[train] Trained {rows_trained} rows total (start={start}, "
          f"steps={steps_done}, eff_batch={eff_batch})")
    return out_dir, rows_trained


def train_rlvr(model, tokenizer, problems, out_dir: str = "qwen_rlvr",
               n_steps: int = 200, group_size: int = 4,
               batch_size: int = 2, max_new: int = 256,
               lr: float = 2e-5, save_every: int = 50,
               resume_step: int = 0):
    """GRPO with proportional rewards (verifier-based).

    Args:
        n_steps: Total training steps
        group_size: Samples per problem (GRPO group)
        batch_size: Problems per step (memory-limited on T4)
        max_new: Max tokens to generate per sample
        lr: Learning rate
        save_every: Checkpoint every N steps (Kaggle 9hr session safety)
        resume_step: Step number to start from (for --resume-rlvr)
    """
    from verify.verifier import Problem, verify
    from torch.optim import AdamW
    import huggingface_hub as hf_hub

    # Only trainable (LoRA) params get optimizer states — the 4-bit base is
    # frozen. Feeding model.parameters() to AdamW would allocate states for
    # frozen/quantized weights and blow up T4 memory.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr)
    device = model.device
    model.train()

    # HF repo for RLVR checkpoints (same layout as SFT: checkpoint-{step}/).
    hf_token = os.environ.get("HF_TOKEN", "")
    ckpt_repo = None
    if hf_token:
        try:
            api = hf_hub.HfApi(token=hf_token)
            ckpt_repo = f"{api.whoami()['name']}/le-gros-chaton-qwen-rlvr-ckpt"
            api.create_repo(ckpt_repo, private=True, exist_ok=True)
            print(f"[train] RLVR checkpoints will upload to HF Hub: {ckpt_repo}")
        except Exception as e:
            print(f"[train] RLVR HF Hub upload disabled: {e}")
            ckpt_repo = None

    n_probs = len(problems)
    for step in range(resume_step, n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        # Rotate through problems so every step sees fresh ones (the old
        # `problems[:batch_size]` re-used the same problems every step).
        start = (step * batch_size) % n_probs
        probs = (problems * ((start + batch_size + n_probs - 1) // n_probs))[
            start:start + batch_size]

        for prob in probs:
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

            # Compute proportional rewards (research-backed upgrades):
            # 1. Loop penalty (SWE-Protégé): repeated verbatim tool calls in a
            #    sample are penalized — the #1 small-model failure mode.
            # 2. Self-verification bonus: samples ending with a verifiable
            #    "run tests" pass are rewarded for finishing, not stopping early.
            rewards = []
            for i in range(group_size):
                sol = tokenizer.decode(out[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                v = verify(p, sol)
                r = v.n_pass / max(v.n_total, 1) if v.n_total > 0 else 0.0

                # Loop penalty: count repeated identical tool-call blocks.
                import re as _re
                calls = _re.findall(r'```(\w+)\s*\n(.*?)```', sol, _re.DOTALL)
                uniq = set()
                dup = 0
                for c in calls:
                    if c in uniq:
                        dup += 1
                    else:
                        uniq.add(c)
                loop_pen = 0.15 * min(dup, 5)  # cap so it can't dominate reward

                # Self-verification bonus: does the solution end with a
                # test-run/finish that asserts success?
                last_call = calls[-1][0] if calls else ""
                sv_bonus = 0.1 if (v.passed or (last_call in ("run_test", "finish")
                                                and "pass" in sol.lower()[:600])) else 0.0

                r = r - loop_pen + sv_bonus
                rewards.append(r)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float)
            adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            # DPPO-style masking: only update tokens where the sample's
            # logprob is trustworthy — here we use the standard GRPO advantage
            # but also skip degenerate all-negative groups (TMax finding:
            # naive GRPO collapses when groups are mostly failures).
            if (rewards_t.max() - rewards_t.min()) < 1e-4:
                continue  # no signal in this group — skip the update

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

        # Checkpoint to survive Kaggle session limits + upload to HF Hub
        if (step + 1) % save_every == 0:
            ckpt_dir = f"{out_dir}_step{step+1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[train] Checkpoint saved to {ckpt_dir}")
            if ckpt_repo:
                print(f"[train] Uploading {ckpt_dir} -> {ckpt_repo} ...")
                try:
                    hf_hub.upload_folder(
                        folder_path=ckpt_dir, repo_id=ckpt_repo,
                        path_in_repo=f"checkpoint-{step+1}",
                        token=hf_token, ignore_patterns=["*.bin", "optimizer.pt"],
                    )
                    print(f"[train] ✓ Uploaded RLVR checkpoint-{step+1}")
                except Exception as e:
                    print(f"[train] RLVR upload failed: {e}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] Saved RLVR model to {out_dir}")
    if ckpt_repo:
        try:
            hf_hub.upload_folder(
                folder_path=out_dir, repo_id=ckpt_repo,
                path_in_repo="final",
                token=hf_token, ignore_patterns=["*.bin", "optimizer.pt"],
            )
            print(f"[train] ✓ Uploaded RLVR final -> {ckpt_repo}/final")
        except Exception as e:
            print(f"[train] RLVR final upload failed: {e}")
    return out_dir


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
    # Treat as HF repo id: list remote checkpoints, download only the newest.
    try:
        from huggingface_hub import snapshot_download, HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN", ""))
        dirs = [f for f in api.list_repo_files(repo_id=resume)
                if "/" in f and f.split("/")[0].startswith("checkpoint-")]
        ck_subdirs = sorted({f.split("/")[0] for f in dirs},
                            key=lambda d: int(d.split("-")[1]))
        if not ck_subdirs:
            print(f"[train] No checkpoints in {resume} yet — starting fresh")
            return None
        latest = ck_subdirs[-1]
        print(f"[train] Resuming SFT from {resume}/{latest}")
        local = snapshot_download(
            repo_id=resume, token=os.environ.get("HF_TOKEN", ""),
            allow_patterns=[f"{latest}/**"],
            ignore_patterns=["*.bin", "optimizer.pt"],
        )
        return os.path.join(local, latest)
    except Exception as e:
        print(f"[train] Could not resolve SFT checkpoint {resume}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train the best coding agent")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", default="Nexlab/fable5-agentic-coding-sft")
    parser.add_argument("--sft-only", action="store_true", help="SFT only (skip RLVR)")
    parser.add_argument("--trajectory-sft", action="store_true",
                        help="SFT on agent trajectories (assistant-token loss, long ctx)")
    parser.add_argument("--rlvr-only", action="store_true", help="RLVR only (skip SFT)")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset rows")
    parser.add_argument("--sft-start", type=int, default=0,
                        help="Row offset to start SFT from (0 = beginning; use the row "
                             "count of a prior run to continue on the rest of the data)")
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rlvr-steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO samples per problem")
    parser.add_argument("--rlvr-lr", type=float, default=2e-5,
                        help="RLVR learning rate")
    parser.add_argument("--rlvr-max-new", type=int, default=256,
                        help="Max tokens per RLVR generation")
    parser.add_argument("--rlvr-save-every", type=int, default=25,
                        help="Save+upload an RLVR checkpoint every N steps")
    parser.add_argument("--resume-rlvr", default=None,
                        help="Resume RLVR from this checkpoint dir (local or HF repo)")
    parser.add_argument("--adapter", default=None,
                        help="Pretrained LoRA adapter to start from (local dir or HF repo id, "
                             "e.g. {user}/le-gros-chaton-qwen for the Phase 1 SFT adapter)")
    parser.add_argument("--resume-sft", default=None,
                        help="Resume SFT from a local checkpoint dir or an HF Hub "
                             "repo (auto-pulls the latest checkpoint-* subdir)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max tokens per training sequence (SFT)")
    parser.add_argument("--trajectory-ctx", type=int, default=16384,
                        help="Context length for trajectory SFT — Qwen3.5-9B natively "
                             "supports 262144 (256K); raise as hardware allows "
                             "(8K T4, 16-32K L4, 64K+ A100/H100)")
    parser.add_argument("--output", default="qwen_coding_agent")
    args = parser.parse_args()

    print(f"[train] === Le Gros Chaton - Coding Agent Training ===")
    print(f"[train] Model: {args.model}")
    print(f"[train] Dataset: {args.dataset}")
    print(f"[train] Output: {args.output}")

    # Load model — optionally attach the Phase 1 SFT adapter, then optionally
    # attach an existing RLVR adapter (resume) on top of it.
    resume_step = 0
    if args.resume_rlvr:
        rlvr_ckpt = resolve_sft_ckpt(args.resume_rlvr)  # same checkpoint-* layout
        if rlvr_ckpt:
            resume_step = int(os.path.basename(rlvr_ckpt).split("-")[1])
            print(f"[train] Resuming RLVR from step {resume_step}: {rlvr_ckpt}")
    model, tokenizer = load_model(args.model, adapter=args.adapter)
    if args.resume_rlvr and rlvr_ckpt:
        model.load_adapter(rlvr_ckpt, adapter_name="rlvr")
        model.set_adapter("rlvr")
        print(f"[train] RLVR adapter attached (step {resume_step})")

    if not args.rlvr_only:
        # Phase 1: SFT on Fable5 dataset (or agent trajectories)
        print(f"\n{'='*50}")
        if args.trajectory_sft:
            print(f"[train] PHASE 1b: TRAJECTORY SFT (assistant-token loss)")
            print(f"{'='*50}")
            trajs = load_agent_traces_full(limit=args.limit)
            if not trajs:
                raise SystemExit("[train] No trajectories found — run gen_trajectories.py first")
            dataset = {"messages": [t["messages"] for t in trajs]}
            sft_path, sft_rows = train_sft(
                model, tokenizer, dataset,
                out_dir=f"{args.output}_sft",
                lr=args.lr, epochs=args.sft_epochs,
                batch_size=args.batch_size, max_length=args.trajectory_ctx,
                resume_from_checkpoint=resolve_sft_ckpt(args.resume_sft),
                start=args.sft_start, trajectory=True,
            )
            try:
                with open(os.path.join(sft_path, "sft_progress.json"), "w") as f:
                    json.dump({"start": args.sft_start, "trained_rows": sft_rows,
                               "mode": "trajectory"}, f)
            except Exception as e:
                print(f"[train] could not write sft_progress.json: {e}")
        else:
            print(f"[train] PHASE 1: SFT on {args.dataset}")
            print(f"{'='*50}")
            fable = load_fable5_dataset(limit=args.limit, start=args.sft_start)
            traces = load_agent_traces(limit=min(args.limit or 5000, 5000))
            dataset = fable
            if traces:
                print(f"[train] Merging {len(traces)} agent traces with Fable5 data")
                # Keep it simple: traces first, then Fable5
                dataset = {"messages": list(traces) + list(fable)}
            sft_path, sft_rows = train_sft(
                model, tokenizer, dataset,
                out_dir=f"{args.output}_sft",
                lr=args.lr, epochs=args.sft_epochs,
                batch_size=args.batch_size, max_length=args.max_length,
                resume_from_checkpoint=resolve_sft_ckpt(args.resume_sft),
                start=args.sft_start,
            )
            # Marker for the next stage (Modal 160k continuation).
            try:
                with open(os.path.join(sft_path, "sft_progress.json"), "w") as f:
                    json.dump({"start": args.sft_start, "trained_rows": sft_rows}, f)
                print(f"[train] sft_progress.json: rows_trained={sft_rows}")
            except Exception as e:
                print(f"[train] could not write sft_progress.json: {e}")

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
            save_every=args.rlvr_save_every,
            resume_step=resume_step,
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
