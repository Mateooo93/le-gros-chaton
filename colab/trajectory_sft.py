#!/usr/bin/env python3
"""Le Gros Chaton — Trajectory SFT, one command on any GPU.

Standalone trajectory SFT (Phase 2b): pulls verified Kimi K3 teacher traces
from the HF dataset TRACES_REPO (file TRACES_FILE), loads Qwen3.5-9B + the
91% Fable5 SFT adapter (ADAPTER) in 4-bit, trains with ASSISTANT-token-only
loss — mirroring train_qwen.py's CURRENT trajectory pipeline (conditional
task grounding: an 'Issue: ...' user message is prepended only to old
assistant-first traces, new user-first traces are left untouched; plus its
tokenize_trajectory_fn: unknown-role masking, message-boundary truncation) —
then uploads the resulting adapter to OUT_REPO as "traj_sft".

No notebook / google.colab / repo-file imports — self-contained. Runs on a
local RTX 2070, Colab T4, Kaggle, or Modal. All config via env vars with
sane defaults; --no-upload skips the final hub upload.

Example:
    HF_TOKEN=hf_xxx python trajectory_sft.py            # full run + upload
    HF_TOKEN=hf_xxx python trajectory_sft.py --no-upload

Deps: torch, transformers, peft, bitsandbytes, datasets, huggingface_hub,
tiktoken (installed with the others in the notebook's cell 1).
"""
import argparse
import json
import os
import sys


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[trajsft] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[trajsft] ERROR: {msg}", flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------
# Trace pulling (step 1)
# --------------------------------------------------------------------------
def load_traces(repo_id: str, filename: str, token: str) -> list[dict]:
    """Download the trace file from TRACES_REPO and parse it."""
    from huggingface_hub import hf_hub_download

    log(f"Pulling '{filename}' from dataset '{repo_id}' ...")
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    except Exception as e:  # noqa: BLE001 — surface ANY hub failure friendly
        status = getattr(e, "response", None)
        code = status.status_code if status is not None else None
        hint = (
            f"(HTTP {code}) " if code else ""
        ) + (
            "Check that TRACES_REPO is spelled correctly and that HF_TOKEN "
            "has read access to it. If the repo is gated, accept the terms "
            "on the Hub first. Re-run with TRACES_REPO=<correct-repo>."
        )
        die(f"could not download '{filename}' from '{repo_id}': {type(e).__name__}: {e} {hint}")
    log(f"Downloaded traces to {local}")
    traces = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


# --------------------------------------------------------------------------
# Tokenization — copied VERBATIM from train_qwen.py's tokenize_trajectory_fn
# (assistant-token-only loss masking). Do not "improve" it.
# --------------------------------------------------------------------------
def format_trajectory(messages: list[dict]) -> str:
    """Format a full agent trajectory as text (verbatim from train_qwen.py).

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


def make_tokenize_fn(tokenizer, max_length: int):
    """Return train_qwen.py's tokenize_trajectory_fn bound to our tokenizer.

    Synced with the CURRENT train_qwen.py version: unknown roles (e.g.
    "tool") are masked as context, and truncation drops whole overflowing
    messages at a boundary instead of cutting mid-message.

    Every message is formatted with role tags; labels for non-assistant
    tokens (system, user/tool results) are set to -100 so the model only
    learns to produce its own decisions/tool calls.
    """
    def tokenize_trajectory_fn(examples):
        batch_texts, batch_labels = [], []
        for m in examples["messages"]:
            if not m:
                continue
            text = format_trajectory(m)
            if not text:
                continue
            # Tokenize per message and merge. Verified against the Qwen3.5
            # tokenizer: merging per-chunk ids is byte-identical to tokenizing
            # the full text (no merges cross the <|im_end|>\n<|im_start|>
            # boundary), so ids/labels are exact by construction.
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
                elif role == "assistant":
                    chunk = f"<|im_start|>assistant\n{content}<|im_end|>\n"
                    train = True
                else:
                    # Unknown roles (e.g. "tool") are context, never trained.
                    chunk = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    train = False
                enc_chunk = tokenizer(chunk, add_special_tokens=False)
                chunk_ids = enc_chunk["input_ids"]
                # Truncate at a message boundary: dropping an overflowing
                # message beats training on a tool call cut mid-args. The first
                # message (the prepended issue) never overflows in practice.
                if merged_ids and len(merged_ids) + len(chunk_ids) > max_length:
                    break
                merged_ids.extend(chunk_ids)
                merged_labels.extend([c if train else -100 for c in chunk_ids])
            merged_ids = merged_ids[:max_length]
            merged_labels = merged_labels[:max_length]
            batch_texts.append(merged_ids)
            batch_labels.append(merged_labels)

        if not batch_texts:
            raise ValueError("no tokenizable traces in this batch (all messages empty?)")

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

    return tokenize_trajectory_fn


# --------------------------------------------------------------------------
# Model + adapter (step 3) — mirrors the notebook's cell 4 exactly
# --------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str, adapter: str):
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    device_map = os.environ.get("DEVICE_MAP", "").strip() or None
    # NOTE: device_map="auto" makes accelerate wrap model.forward in
    # ConvertOutputsToFp32 (full-logits fp32 upcast) — harmless at 512 ctx but
    # a ~3GiB OOM at trajectory ctx on a 14.5GiB T4. Load directly on cuda:0
    # (default None) so the 4-bit model lands on the single GPU with no wrapper.
    max_memory_raw = os.environ.get("MAX_MEMORY", "").strip()
    max_memory = None
    if max_memory_raw:
        try:
            max_memory = json.loads(max_memory_raw)
            # transformers wants integer device keys for GPUs ("0" -> 0)
            max_memory = {int(k) if str(k).lstrip("-").isdigit() else k: v
                          for k, v in max_memory.items()}
        except Exception as e:
            log(f"WARN: MAX_MEMORY not valid JSON ({e}); ignoring")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    log(f"Loading base model '{model_name}' (4-bit nf4, fp16 compute, device_map={device_map}, max_memory={max_memory}) ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    log(f"Attaching 91% Fable5 SFT adapter '{adapter}' ...")
    if adapter and adapter.strip().lower() not in ("none", "null", ""):
        model = PeftModel.from_pretrained(model, adapter)
    else:
        log("SKIP adapter attach (ADAPTER=none) — training on quantized base directly")
    log(f"Model + SFT adapter loaded | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tok


def ground_traces(traces: list[dict]) -> tuple[list[list[dict]], int, int]:
    """Prepend an 'Issue: ...' user message to OLD-style traces only.

    Old traces (the 8 from the HF seed) start at the model's first response
    (assistant-first) and lack the issue — without the task in the input the
    model learns to act from an empty prompt. NEW GenPipeline-format traces
    already start with the real user prompt (role user, "Fix this issue in
    the repo at <dir>:\\n\\n<issue>") and end with a trainable assistant
    SELF-REVIEW — prepending again would duplicate the prompt, so they are
    left untouched. Returns (grounded, n_reconstructed, n_already_grounded).
    """
    grounded, n_reconstructed, n_already = [], 0, 0
    for t in traces:
        msgs = t.get("messages") or []
        if msgs and msgs[0].get("role") == "user":
            n_already += 1  # new format: real user prompt present
            grounded.append(msgs)
            continue
        issue = t.get("issue") or t.get("instance_id")
        if issue:
            n_reconstructed += 1
            grounded.append([{"role": "user",
                              "content": f"Issue: {issue}\n\nStart by exploring the codebase."}] + msgs)
        else:
            grounded.append(msgs)  # nothing to reconstruct with — keep as-is
    return grounded, n_reconstructed, n_already


# --------------------------------------------------------------------------
# Upload (step 5)
# --------------------------------------------------------------------------
def upload_adapter(out_dir: str, out_repo: str, token: str) -> None:
    from huggingface_hub import HfApi

    log(f"Uploading adapter '{out_dir}' -> '{out_repo}/traj_sft' ...")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=out_dir,
        repo_id=out_repo,
        path_in_repo="traj_sft",
        token=token,
        ignore_patterns=["*.bin", "optimizer.pt"],
    )
    log(f"Adapter uploaded to {out_repo}/traj_sft")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Le Gros Chaton trajectory SFT (one command, any GPU)",
    )
    parser.add_argument("--upload", dest="upload", action="store_true", default=True,
                        help="upload the adapter to OUT_REPO/traj_sft (default)")
    parser.add_argument("--no-upload", dest="upload", action="store_false",
                        help="skip the final hub upload (dry-run / local test)")
    args = parser.parse_args()

    # --- 1. Config (env vars with defaults) -------------------------------
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
    if not HF_TOKEN:
        die("HF_TOKEN is required (set it in the environment: "
            "export HF_TOKEN=hf_... or pass it inline)")
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
    ADAPTER = os.environ.get("ADAPTER", "mateo0093/le-gros-chaton-qwen")
    TRACES_REPO = os.environ.get("TRACES_REPO", "mateo0093/le-gros-chaton-traces")
    TRACES_FILE = os.environ.get("TRACES_FILE", "agent_traces_normalized.jsonl")
    OUT_REPO = os.environ.get("OUT_REPO", "mateo0093/le-gros-chaton-qwen")
    TRAJECTORY_CTX = int(os.environ.get("TRAJECTORY_CTX", "16384"))
    BATCH = int(os.environ.get("BATCH", "1"))
    EPOCHS = int(os.environ.get("EPOCHS", "3"))
    LR = float(os.environ.get("LR", "2e-4"))
    OUT_DIR = os.environ.get("OUT_DIR", "qwen_traj_sft")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log(f"config: MODEL_NAME={MODEL_NAME} ADAPTER={ADAPTER} "
        f"TRACES_REPO={TRACES_REPO} TRACES_FILE={TRACES_FILE} "
        f"OUT_REPO={OUT_REPO}")
    log(f"config: TRAJECTORY_CTX={TRAJECTORY_CTX} BATCH={BATCH} "
        f"EPOCHS={EPOCHS} LR={LR} OUT_DIR={OUT_DIR} upload={args.upload}")

    # --- 2. Pull verified teacher traces ----------------------------------
    traces = load_traces(TRACES_REPO, TRACES_FILE, HF_TOKEN)
    n_all, n_verified = len(traces), sum(1 for t in traces if t.get("verified"))
    traces = [t for t in traces if t.get("verified")]  # verified = real signal
    if not traces:
        die(f"no verified traces in {TRACES_REPO} "
            f"({n_all} loaded, {n_verified} verified) — nothing to train on")
    t0 = traces[0]
    log(f"Loaded {n_all} traces, keeping {len(traces)} verified "
        f"(sample: {t0.get('instance_id')} | turns={t0.get('turns')} "
        f"| msgs={len(t0.get('messages', []))})")

    # Task grounding (mirrors train_qwen.py, conditional on trace format):
    # old assistant-first traces get an 'Issue: ...' user message prepended;
    # new traces already start with the real user prompt and are untouched.
    grounded, n_reconstructed, n_already = ground_traces(traces)
    log(f"Task grounding: {n_reconstructed} old-style traces reconstructed "
        f"with an 'Issue: ...' preamble; {n_already} already start with a "
        "user prompt (left untouched)")

    import torch
    if not torch.cuda.is_available():
        die("no CUDA GPU detected — 4-bit loading + paged_adamw_8bit need a "
            "CUDA GPU (Colab T4, Kaggle P100/T4, Modal A10G, local RTX). "
            "Check `nvidia-smi` / runtime type.")
    gpu_name = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu_name}")

    # --- 3. Load model + 91% SFT adapter (4-bit) --------------------------
    try:
        model, tok = load_model_and_tokenizer(MODEL_NAME, ADAPTER)
    except ImportError as e:
        die(f"missing dependency for 4-bit loading: {e} — install with "
            "`pip install -q peft bitsandbytes transformers accelerate`")
    except Exception as e:  # noqa: BLE001 — 4-bit load fails w/ RuntimeError on CPU etc.
        die(f"failed to load model/adapter: {type(e).__name__}: {e} — make "
            "sure bitsandbytes is installed and CUDA is available "
            "(`pip install -U bitsandbytes`)")

    # 4-bit models are frozen by definition — the Trainer refuses pure
    # quantized models. Attach a fresh LoRA so the trajectory SFT has
    # trainable parameters (and a trainable adapter to save at the end).
    import torch as _t
    from peft import LoraConfig, get_peft_model
    lora_r = int(os.environ.get("LORA_R", "16"))
    log(f"Attaching fresh LoRA (r={lora_r}, alpha={2 * lora_r}, q/k/v/o/gate/up/down) ...")
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    log(f"LoRA attached | VRAM: {_t.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- 4. Tokenize trajectories (assistant-only loss) --------------------
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    class AssistantTokenTrainer(Trainer):
        """Compute CE only on assistant-token positions (labels != -100).

        trajectory SFT masks every non-assistant token to -100. The stock
        ForCausalLMLoss still materializes the FULL [seq, vocab] logits in
        fp32 (logits.float()) before masking, which OOMs on a 14.5GiB T4
        with this 248K-vocab checkpoint (~5.7GiB just for the fp32 upcast).
        Gathering only the ~700 assistant positions first shrinks the fp32
        tensor 4-5x and is numerically identical (masked positions contribute
        zero to the loss).
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits                     # [B, S, V] fp16
            shift_logits = logits[..., :-1, :]          # view, no copy
            shift_labels = labels[..., 1:]
            mask = shift_labels != -100
            active_logits = shift_logits[mask]          # [N, V] fp16, N<<S
            active_labels = shift_labels[mask]
            # Free the full [B,S,V] logits + inputs BEFORE the fp32 upcast so
            # we never hold the full vocab tensor in fp32 (which is ~5.7GiB at
            # 3K ctx on this 248K-vocab checkpoint and OOMs a 14.5GiB T4).
            outputs.logits = None
            del logits, shift_logits, inputs
            if active_logits.numel() == 0:
                loss = active_logits.float().sum()       # 0.0, keeps grad graph
            else:
                loss = torch.nn.functional.cross_entropy(
                    active_logits.float(), active_labels)
            return (loss, outputs) if return_outputs else loss

    ds = Dataset.from_dict({"messages": grounded})
    tokenized = ds.map(
        make_tokenize_fn(tok, TRAJECTORY_CTX), batched=True,
        remove_columns=ds.column_names, desc="Tokenizing",
    )
    first = tokenized[0]
    import torch as _t
    first_ids = _t.as_tensor(first["input_ids"])
    n_train = int((_t.as_tensor(first["labels"]) != -100).sum())
    log(f"Tokenized {len(tokenized)} trajectories | first trace: "
        f"{first_ids.shape[0]} tokens, {n_train} trainable (assistant)")

    # --- 5. Trajectory SFT (same recipe as the notebook) -------------------
    eff_batch = BATCH * 8  # gradient_accumulation_steps
    steps_per_epoch = max(1, (len(tokenized) + eff_batch - 1) // eff_batch)
    total_steps = steps_per_epoch * EPOCHS
    log(f"Training: {len(tokenized)} traces | eff batch {eff_batch} "
        f"(batch {BATCH} x grad-accum 8) | {steps_per_epoch} steps/epoch "
        f"x {EPOCHS} epochs = {total_steps} steps")

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=8,   # eff batch 8
        learning_rate=LR,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        logging_steps=5,
        save_steps=0,
        save_total_limit=1,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",        # offload optimizer states to CPU
    )

    trainer = AssistantTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8),
    )
    log("Starting training ...")
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    log(f"Trajectory SFT done: {OUT_DIR}")

    # --- 6. Upload (unless --no-upload) ------------------------------------
    if args.upload:
        upload_adapter(OUT_DIR, OUT_REPO, HF_TOKEN)
    else:
        log("--no-upload: skipping hub upload (adapter stays in "
            f"'{OUT_DIR}')")

    # --- 7. Summary --------------------------------------------------------
    log("============ done ============")
    log(f"traces trained on : {len(traces)} verified (of {n_all} loaded)")
    log(f"steps             : {total_steps} (eff batch {eff_batch}, {EPOCHS} epochs)")
    log(f"GPU               : {gpu_name}")
    log(f"VRAM used         : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    log(f"adapter           : {OUT_DIR}"
        + (f" -> uploaded to {OUT_REPO}/traj_sft" if args.upload
           else " (upload skipped)"))
    log("Next: RLVR with --diversity to bake in creativity.")


if __name__ == "__main__":
    main()
