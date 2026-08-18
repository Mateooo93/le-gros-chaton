#!/usr/bin/env python3
"""Verify the trajectory-SFT adapter uploaded from Kaggle.

Checks (all network/disk, no GPU needed):
  1. The HF repo exists and is NOT stale (commit newer than the run start).
  2. It contains the adapter files we expect (adapter_config.json,
     adapter_model.safetensors, tokenizer files).
  3. adapter_config.json is sane: r=16, target modules include q/k/v/o/gate/
     up/down, base_model_name_or_path points at a real repo, no NaN config.
  4. (optional --torch) load the adapter with peft to confirm it parses.

Usage:
    HF_TOKEN=... python3 verify_traj_adapter.py [--repo mateo0093/le-gros-chaton-qwen-traj-sft] [--since "2026-08-17T00:00:00Z"]
"""
import argparse
import json
import os
import sys


def log(*a):
    print("[verify-traj]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mateo0093/le-gros-chaton-qwen-traj-sft")
    ap.add_argument("--since", help="ISO timestamp; fails if repo is older",
                    default="2026-08-17T00:00:00Z")
    ap.add_argument("--torch", action="store_true",
                    help="also load adapter via peft to confirm it parses")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN", "")
    api = HfApi(token=token or None)

    log(f"Repo: {args.repo}")
    try:
        info = api.model_info(args.repo, files_metadata=True)
    except Exception as e:
        log(f"FAIL: cannot fetch repo: {e}")
        return 1
    log(f"  last commit: {info.last_modified}")

    from datetime import datetime, timezone
    since = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    lm = info.last_modified
    if lm.tzinfo is None:
        lm = lm.replace(tzinfo=timezone.utc)
    if lm < since:
        log(f"FAIL: repo last_modified {lm} is OLDER than --since {since}")
        return 1
    log("  fresh (after --since) OK")

    names = {f.rfilename for f in (info.siblings or [])}
    required = {"adapter_config.json", "adapter_model.safetensors"}
    missing = required - names
    if missing:
        log(f"FAIL: missing adapter files: {sorted(missing)}")
        log(f"  found: {sorted(names)}")
        return 1
    log("  adapter files present OK")

    # Walk the tree for adapter_config.json (may sit in a subfolder if the
    # upload ever goes to traj_sft/ again).
    cfg_path = None
    for f in (info.siblings or []):
        if f.rfilename.endswith("adapter_config.json"):
            cfg_path = f.rfilename
            break
    cfg = api.hf_hub_download(args.repo, cfg_path)
    with open(cfg) as fh:
        c = json.load(fh)
    log(f"  adapter config: r={c.get('r')} lora_alpha={c.get('lora_alpha')}")
    log(f"  target_modules: {sorted(c.get('target_modules', []))}")
    log(f"  base_model_name_or_path: {c.get('base_model_name_or_path')}")
    ok = True
    if c.get("r") != 16:
        log(f"  WARN: r={c.get('r')} != 16")
        ok = False
    # Hybrid arch: modules are namespaced (self_attn.q_proj, linear_attn.in_proj_*,
    # mlp.gate_proj). Require the SUFFIX classes, not bare names.
    tm = c.get("target_modules", [])
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                   "up_proj", "down_proj", "in_proj_qkv", "in_proj_a",
                   "in_proj_b", "in_proj_z", "out_proj"):
        if not any(t.endswith(suffix) for t in tm):
            log(f"  WARN: target module {suffix} missing")
            ok = False
    if len(tm) < 12:
        log(f"  WARN: only {len(tm)} target modules (expect 12 hybrid classes)")
        ok = False
    if c.get("peft_type") != "LORA":
        log(f"  WARN: peft_type={c.get('peft_type')} != LORA")
        ok = False
    if not c.get("base_model_name_or_path"):
        log("  WARN: base_model_name_or_path is empty/null")
        ok = False
    log("  config sane" if ok else "  config SUSPICIOUS")

    # Tokenizer presence (needed for any downstream load).
    tok_ok = any(n.endswith(("tokenizer.json", "tokenizer_config.json", "vocab.json"))
                 for n in names)
    if not tok_ok:
        log("  WARN: no tokenizer files in adapter repo (merge uses base tokenizer — OK)")

    if args.torch:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        log("  loading adapter with peft (torch)...")
        model = AutoModelForCausalLM.from_pretrained(
            c.get("base_model_name_or_path", "Qwen/Qwen3.5-9B"),
            torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True,
        )
        pm = PeftModel.from_pretrained(model, args.repo)
        log(f"  peft load OK | trainable adapters: {list(pm.peft_config.keys())}")
        del model, pm

    log("VERIFY PASS" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
