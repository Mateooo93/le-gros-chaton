#!/usr/bin/env python3
"""Re-save the trajectory-SFT adapter as a FLAT, loadable adapter.

The Kaggle traj training built the LoRA with get_peft_model() on top of an
already-PEFT model (Fable5). Two problems came out of that:

  1. STALE CONFIG: trainer.save_model() serialized only the OUTER LoraConfig's
     7 target modules even though the weights cover 12 (get_peft_model
     inherited the inner adapter's module set). base_model_name_or_path is
     null. Merging with this config attaches only 7 modules.

  2. NESTED KEYS: the saved weights carry the double prefix
     (base_model.model.base_model.model...) because they were saved from
     inside a PeftModel-wrapping-PeftModel. PeftModel.from_pretrained() onto a
     PLAIN model accepts that silently but does NOT transfer the values
     (verified: lora_A 0.0215 vs 0.123 expected, lora_B zeros) — so
     merge_sft.py's sequential merge would have been WRONG.

This script rebuilds the adapter flat:
  - recover target_modules + r from the weight keys themselves
  - map nested keys -> flat keys (base_model.model.<path>)
  - save with a correct adapter_config.json (base_model_name_or_path set)

Usage:
    GPU_TORCH=1 python3 flatten_traj_adapter.py \\
        --in qwen_traj_sft --out qwen_traj_sft_flat \\
        --base Qwen/Qwen3.5-9B
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="qwen_traj_sft")
    ap.add_argument("--out", dest="out_dir", default="qwen_traj_sft_flat")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors import safe_open
    pat = re.compile(r"layers\.\d+\.(.+?)\.lora_[AB]\.weight$")
    nested = {}
    targets = Counter()
    with safe_open(in_dir / "adapter_model.safetensors", framework="pt") as sf:
        for k in sf.keys():
            nested[k] = sf.get_tensor(k)
            if (m := pat.search(k)):
                targets[m.group(1)] += 1
    targets_list = sorted(targets)
    r = next(v.shape[0] for v in nested.values())  # any lora_A row = r

    # --- map nested keys to flat keys -------------------------------------
    # nested: base_model.model.base_model.model.model.layers.N.<mod>.lora_X.weight
    # flat:   base_model.model.model.layers.N.<mod>.lora_X.weight
    flat = {}
    loose = 0
    STRIP = "base_model.model.base_model.model."
    for k, v in nested.items():
        if k.startswith(STRIP):
            flat["base_model.model." + k[len(STRIP):]] = v
        else:
            loose += 1
            print(f"[flatten] no flat mapping for {k[:90]}...")

    from safetensors.torch import save_file
    save_file(flat, out_dir / "adapter_model.safetensors")
    print(f"[flatten] {len(flat)} tensors -> {out_dir / 'adapter_model.safetensors'}"
          f" ({loose} unmapped)")

    config = {
        "loftq_config": {},
        "target_modules": targets_list,
        "r": r,
        "lora_alpha": 2 * r,          # training used alpha = 2*r
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "peft_version": "0.19.1",
        "base_model_name_or_path": args.base,
        "inference_mode": True,
        "init_lora_weights": True,
        "modules_to_save": None,
        "fan_in_fan_out": False,
        "layers_pattern": None,
        "layers_to_transform": None,
        "exclude_modules": None,
        "use_rslora": False,
        "use_dora": False,
        "alpha_pattern": {},
        "rank_pattern": {},
        "revision": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    print(f"[flatten] config written: {len(targets_list)} modules, r={r}, alpha={2*r}")
    print(f"[flatten] OK: {out_dir}")


if __name__ == "__main__":
    main()
