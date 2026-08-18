#!/usr/bin/env bash
# Le Gros Chaton — MI300X (ROCm) setup, one shot.
#
# Usage:
#   bash setup_mi300x.sh                # full setup + sanity check
#   bash setup_mi300x.sh --stack-only   # deps only, skip GPU smoke test
#
# Assumes: Ubuntu/Debian-ish box, ROCm drivers + /dev/kfd present, 50h GPU
# budget. Installs into a venv at ./venv (or $VENV). Everything after this
# goes through .venv/bin/python so the ROCm torch is never shadowed.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-$PWD/venv}"
: "${HF_TOKEN:?set HF_TOKEN (used to pull gated/private repos like our adapters)}"
export HF_TOKEN

echo "=== [1/5] ROCm sanity ==="
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd missing — ROCm drivers not loaded. Check machine." >&2
    exit 1
fi
rocm-smi --showproductname 2>/dev/null | grep -i "product name" || echo "rocm-smi not found (ok if drivers present but tools missing)"
nproc && free -g | head -2

echo "=== [2/5] Python venv ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip

echo "=== [3/5] ROCm PyTorch (bf16 tensor cores) ==="
# ROCm 6.2 wheels; pick the torch matching the installed ROCm driver.
"$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

echo "=== [4/5] Project deps ==="
"$VENV/bin/pip" install -q \
    "transformers==5.14.1" "tokenizers==0.22.1" \
    "peft" "datasets" "safetensors" "accelerate" \
    "huggingface_hub" "tiktoken" "trl" "bitsandbytes" \
    "httpx" "requests" "numpy"

echo "=== [5/5] GPU smoke test ==="
"$VENV/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu: {name} | {mem:.0f} GB")
    # bf16 matmul sanity (Qwen3.5 hybrid uses bf16; gated-delta-net must not NaN)
    a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
    c = (a @ b)
    assert torch.isfinite(c).all(), "bf16 matmul produced NaN — bad ROCm build"
    print("bf16 matmul OK (finite)")
else:
    print("WARNING: no GPU visible via torch — check ROCm install")
PY

echo "=== setup done ==="
echo "next: bash run_mi300x.sh  (traj SFT at 16K -> merge -> serve -> eval -> RLVR)"
