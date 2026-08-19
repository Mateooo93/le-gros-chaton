#!/usr/bin/env python3
"""Patch the merged model config so vllm (docker vllm 0.16rc2) can load it
without complaining about multimodal requirements.

Two things to fix:
1. `architectures` -> ["Qwen3_5ForCausalLM"].
   We patch this in vllm's registry to point to the text-only handler (NOT
   the multimodal Qwen3_5ForConditionalGeneration that demands a vision_config).
2. Drop `rope_parameters.mrope_interleaved` and `mrope_section`.
   The merge script saved these from the multimodal class. Our text-only
   model has plain RoPE and vllm asserts M-RoPE is not implemented when
   these keys are present.
3. Drop `dtype` (vllm uses --dtype, not the config field).

Idempotent: re-running on an already-patched config is a no-op.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def patch(path: Path) -> None:
    cfg = json.loads(path.read_text())
    archs = cfg.get("architectures", [])
    if archs and archs[0] != "Qwen3_5ForCausalLM":
        cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        print(f"{path.name}: architectures -> {cfg['architectures']}")

    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict):
        before = dict(rp)
        rp.pop("mrope_interleaved", None)
        rp.pop("mrope_section", None)
        if rp != before:
            cfg["rope_parameters"] = rp
            print(f"{path.name}: stripped M-RoPE keys (kept {list(rp.keys())})")

    if "dtype" in cfg:
        cfg.pop("dtype")
        print(f"{path.name}: dropped dtype")

    path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    base = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            "/root/.cache/huggingface/models--mateo0093--le-gros-chaton-qwen-merged-16k/snapshots/dc99ab89fbff8f628ae5ee3f8dbecf4242fcad71/config.json"
        )
    )
    patch(base)
    print("patched ok")