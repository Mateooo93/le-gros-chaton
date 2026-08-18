#!/usr/bin/env python3
"""Repair the trajectory-SFT adapter's stale adapter_config.json.

peft 0.19's get_peft_model on top of an already-PEFT model inherits the inner
adapter's target_modules. The traj LoRA therefore has weights on ALL 12 module
classes (linear_attn in_proj_*/out_proj + self_attn q/k/v/o + mlp gate/up/down,
r=16 alpha=32), but trainer.save_model() serialized a config listing only the
7 modules from the OUTER LoraConfig plus base_model_name_or_path=null.

Merging with that stale config would attach the traj delta to only 7 modules —
the gated-delta-net layers would silently keep Fable5 behavior. This rewrites
the config to match the actual weights.

Usage:
    python3 fix_traj_config.py [dir]     # dir defaults to ./qwen_traj_sft
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open

EXPECTED_BASE = "Qwen/Qwen3.5-9B"


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "qwen_traj_sft")
    cfg_path = d / "adapter_config.json"
    st_path = d / "adapter_model.safetensors"
    if not cfg_path.exists() or not st_path.exists():
        sys.exit(f"need {cfg_path} and {st_path} in {d}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    # --- 1. Recover the true target modules from the weight keys ----------
    pat = re.compile(r"layers\.\d+\.(.+?)\.lora_A\.weight$")
    with safe_open(st_path, framework="pt") as sf:
        mods = Counter(
            m.group(1) for k in sf.keys()
            if (m := pat.search(k))
        )
    targets = sorted(mods)
    found_r = None
    with safe_open(st_path, framework="pt") as sf:
        for k in sf.keys():
            if pat.search(k) and k.endswith("lora_A.weight"):
                found_r = sf.get_tensor(k).shape[0]
                break

    print(f"[fix] weights: {len(mods)} module classes total "
          f"({len(targets) - len(cfg.get('target_modules', []))} more than config)")
    print(f"[fix] recovered targets ({len(targets)}): {targets}")
    print(f"[fix] recovered r from weight shape: {found_r}")

    # --- 2. Patch the config -------------------------------------------------
    old_targets = list(cfg.get("target_modules", []))
    cfg["target_modules"] = targets
    cfg["r"] = found_r or cfg.get("r", 16)
    cfg["lora_alpha"] = (found_r or 16) * 2  # training used alpha = 2*r
    cfg["base_model_name_or_path"] = EXPECTED_BASE
    cfg["task_type"] = "CAUSAL_LM"
    cfg["peft_version"] = "0.19.1"

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[fix] wrote {cfg_path}")
    print(f"[fix] target_modules: {len(old_targets)} -> {len(targets)}")
    print(f"[fix] base_model_name_or_path: null -> {EXPECTED_BASE}")
    print(f"[fix] r/alpha: {cfg['r']}/{cfg['lora_alpha']}")


if __name__ == "__main__":
    main()
