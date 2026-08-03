"""Smoke test: build Modal image, verify fla/causal-conv1d compiled, model loads.

Cheap (~$0.15). Run BEFORE the expensive full-160k SFT:
    python modal_smoke_test.py
"""
import os
import modal

app = modal.App("le-gros-chaton-smoke")

# Same image as modal_train_qwen.py (kept in sync manually) so the smoke test
# tests exactly what the training run will use.
image = (
    # nvidia/cuda devel base provides nvcc for building causal-conv1d
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.14.1",
        "tokenizers==0.22.1",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "tiktoken",
        "safetensors",
    )
    .run_commands(
        "pip install wheel setuptools ninja && pip install causal-conv1d --no-build-isolation || true",
        "pip install 'flash-linear-attention[cuda]' --no-build-isolation || true",
    )
)

HF_SECRET = os.environ.get("CHATON_HF_SECRET", "chaton-hf")


@app.function(
    gpu="L4",
    image=image,
    timeout=1800,
    secrets=[modal.Secret.from_name(HF_SECRET)],
)
def smoke():
    import subprocess, sys
    import torch

    print("[smoke] torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
    print("[smoke] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    # 1. Did fla / causal-conv1d install?
    for pkg in ("fla", "causal_conv1d"):
        try:
            m = __import__(pkg)
            print(f"[smoke] {pkg} OK:", getattr(m, "__version__", "installed"))
        except Exception as e:
            print(f"[smoke] {pkg} FAILED to import: {e}")

    # 2. Does transformers 5.14.1 load Qwen3.5-9B in 4-bit (qwen3_5 arch)?
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", quantization_config=quant,
        device_map="auto", trust_remote_code=True, torch_dtype="auto")
    print("[smoke] model loaded:", type(model).__name__)

    # 3. Quick generate to confirm the full stack works
    inputs = tok("def add(a, b):\n    return", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=8)
    print("[smoke] generate OK:", tok.decode(out[0][-8:], skip_special_tokens=True))
    print("[smoke] SMOKE TEST PASSED")


if __name__ == "__main__":
    app.run(local=False)
