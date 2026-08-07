#!/usr/bin/env python3
"""Merge the full adapter stack (base + Fable5 SFT + trajectory LoRA) into a
single complete model checkpoint that downstream consumers (TB eval, RLVR,
serve) load as a plain model — no adapter stacking at load time.

WHY this exists
---------------
trajectory_sft.py loads base + Fable5 adapter (PeftModel.from_pretrained),
then get_peft_model() attaches a NEW trainable LoRA ON TOP of that. The
resulting checkpoint contains only the outer (trajectory) LoRA weights; the
Fable layer lives in the inner PeftModel. Loading just the trajectory LoRA
without the Fable adapter is wrong (the LoRA deltas were tuned against
Fable-modified activations; missing Fable changes the base deltas).

eval_qwen.py/eval_models treat a checkpoint as a full model dir, not a LoRA
adapter — so the safe artifact downstream is a single merged model.

Usage
-----
    BASE_MODEL=Qwen/Qwen3.5-9B \
    FABLE_ADAPTER=mateo0093/le-gros-chaton-qwen \
    TRAJ_ADAPTER=mateo0093/le-gros-chaton-qwen-traj-sft (or local dir) \
    OUT_DIR=qwen_merged python3 merge_sft.py

Memory: merges in fp16. On a 24GB 4090 this fits; on smaller cards use
MAX_MEMORY={"0":"12GiB","cpu":"20GiB"} to offload.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")
FABLE_MODEL = os.environ.get("FABLE_MODEL", "mateo0093/le-gros-chaton-qwen")
TRAJ_ADAPTER = os.environ.get("TRAJ_ADAPTER", "mateo0093/le-gros-chaton-qwen-traj-sft")
OUT_DIR = os.environ.get("OUT_DIR", "qwen_merged")
MAX_MEMORY = os.environ.get("MAX_MEMORY", "")


def log(*a):
    print("[merge]", *a, flush=True)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not HF_TOKEN:
        log("WARN: HF_TOKEN not set — may fail on gated/private repos")

    max_memory = None
    if MAX_MEMORY:
        import json
        max_memory = {int(k) if str(k).lstrip("-").isdigit() else k: v
                      for k, v in json.loads(MAX_MEMORY).items()}

    log(f"Loading base '{BASE_MODEL}' (fp16, device_map=auto, max_memory={max_memory}) ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        max_memory=max_memory, trust_remote_code=True,
    )
    log(f"Base loaded | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Load BOTH adapters into ONE PeftModel as a stack (deltas compose on the
    # same base activations — mathematically identical to how trajectory_sft
    # trained: base -> Fable5 layer -> trajectory LoRA on top).
    log(f"Attaching Fable5 adapter '{FABLE_MODEL}' ...")
    model = PeftModel.from_pretrained(model, FABLE_MODEL, adapter_name="fable",
                                      is_trainable=False)
    log(f"Attaching trajectory adapter '{TRAJ_ADAPTER}' ...")
    model.load_adapter(TRAJ_ADAPTER, adapter_name="traj")
    model.set_adapter(["fable", "traj"])

    # Merge ALL active adapters into the base weights (PEFT merges the
    # composed deltas additively into W0 — same math as the trained stack).
    log("Merging adapters into base weights ...")
    merged = model.merge_and_unload(adapter_names=["fable", "traj"])
    log(f"  merged | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    os.makedirs(OUT_DIR, exist_ok=True)
    merged.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    log(f"Saved merged model -> {OUT_DIR}")

    if HF_TOKEN:
        out_repo = os.environ.get("MERGE_OUT_REPO", "mateo0093/le-gros-chaton-qwen-merged")
        log(f"Uploading to HF repo '{out_repo}' ...")
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(folder_path=OUT_DIR, repo_id=out_repo,
                          repo_type="model",
                          commit_message="merged base+Fable5+traj SFT")
        log(f"Uploaded merged model -> {out_repo}")
    log("============ merge done ============")


if __name__ == "__main__":
    main()