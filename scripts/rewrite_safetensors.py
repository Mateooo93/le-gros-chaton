#!/usr/bin/env python3
"""Rewrite safetensors keys to add a prefix.

Used to make our flat-key Qwen3.5 merged model (model.embed_tokens.weight)
look like the nested format that vllm's Qwen3_5ForConditionalGeneration expects
(language_model.embed_tokens.weight).

Idempotent: re-running on already-prefixed files is a no-op.
"""
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

def rewrite(path: Path, prefix: str) -> None:
    tensors = load_file(str(path))
    new = {}
    renamed = 0
    for k, v in tensors.items():
        if k.startswith(prefix + "."):
            new[k] = v
        elif k.startswith("model.") or k == "lm_head.weight":
            new[k.replace("model.", prefix + ".", 1)] = v
            renamed += 1
        else:
            new[k] = v
    save_file(new, str(path), metadata={"format": "pt"})
    print(f"{path.name}: {renamed} keys rewritten")

if __name__ == "__main__":
    src = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else "language_model"
    rewrite(src, prefix)