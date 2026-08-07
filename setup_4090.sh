#!/usr/bin/env bash
# Le Gros Chaton — 4090 bootstrap: install everything needed to run the full
# pipeline (trajectory SFT -> TB 2.0 eval -> RLVR) on an Ubuntu machine with
# an NVIDIA GPU, then hand off to run_sft_pipeline.sh.
#
# Usage:  bash setup_4090.sh                          # install deps only
# Usage:  HF_TOKEN=hf_... bash setup_4090.sh --run   # install + start pipeline
#
# Requirements: Ubuntu 22.04/24.04 with NVIDIA driver already installed
#   (check with `nvidia-smi`). If nvidia-smi missing, install the driver first:
#   sudo ubuntu-drivers install && sudo reboot
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[setup] $*"; }

# --- 0. prevent sudo prompts from hanging -----------------------------------
log "0/7 system deps (needs sudo; you may be prompted a few times)"
sudo -n true 2>/dev/null && SUDO="sudo -n" || SUDO="sudo"

# --- 1. base system packages ------------------------------------------------
$SUDO apt-get update -qq || true
$SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl \
    docker.io nvidia-container-toolkit >/dev/null 2>&1 \
    || $SUDO apt-get install -y -qq python3 python3-venv python3-pip git curl

# --- 2. Docker user group (avoids sudo docker in TB eval) -------------------
if ! groups | grep -q docker; then
    $SUDO usermod -aG docker "$USER" || true
    # Harbor containers must be runnable without sudo; if the group was just
    # added, the next login picks it up. Also allow current session:
    $SUDO chmod 666 /var/run/docker.sock 2>/dev/null || true
fi
$SUDO systemctl enable --now docker >/dev/null 2>&1 || true

# --- 3. python venv ----------------------------------------------------------
log "install/7 python deps in .venv"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/pip install -q --upgrade pip wheel setuptools
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -q \
    transformers peft bitsandbytes accelerate datasets huggingface_hub \
    sentencepiece protobuf pynvml "trl>=0.20"
# eval: terminal-bench 2.0 runs via ``harbor`` (the TB 2.0 CLI)
.venv/bin/pip install -q terminal-bench harbor 2>/dev/null || \
    .venv/bin/pip install -q terminal-bench

# --- 4. env file (never commit) ----------------------------------------------
if [ ! -f .env ]; then
    cat > .env <<'ENV'
HF_TOKEN=hf_YOUR_TOKEN_HERE
# TEACHER_API_URL / TEACHER_API_KEY only needed for trace generation (not here)
ENV
fi

# --- 5. sanity: GPU + CUDA visible ------------------------------------------
log "5/7 verify GPU"
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check nvidia-smi"
print(f"[gpu] {torch.cuda.get_device_name(0)} · {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB · torch {torch.__version__}")
PY

# --- 6. notify --------------------------------------------------------------
log "6/7 ready. GPU OK. Next: export HF_TOKEN and run:"
log "   HF_TOKEN=hf_... bash run_sft_pipeline.sh"
if [ "${1:-}" = "--run" ]; then
    log "7/7 --run detected: launching pipeline"
    [ -f .env ] && set -a && . ./.env && set +a
    bash run_sft_pipeline.sh || { echo "[gpu] pipeline finished with rc=$?"; exit 0; }
fi