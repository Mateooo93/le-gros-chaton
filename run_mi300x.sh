#!/usr/bin/env bash
# Le Gros Chaton — MI300X full pipeline runner.
#
# Orchestrates: 16K traj SFT -> merge -> vLLM serve -> TB eval -> RLVR -> eval.
# Each long step logs to logs/ and uploads artifacts to HF so a disconnect
# never costs more than the current step.
#
# Usage:
#   HF_TOKEN=... bash run_mi300x.sh sft      # 16K trajectory SFT (bf16)
#   HF_TOKEN=... bash run_mi300x.sh merge    # base+Fable5+traj -> qwen_merged
#   HF_TOKEN=... bash run_mi300x.sh serve    # vLLM on qwen_merged (blocking)
#   HF_TOKEN=... bash run_mi300x.sh eval     # TB eval vs the running server
#   HF_TOKEN=... bash run_mi300x.sh rlvr     # GRPO with diversity reward
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-$PWD/venv}"
PY="$VENV/bin/python"
: "${HF_TOKEN:?set HF_TOKEN}"
mkdir -p logs

export HF_TOKEN

step="${1:-help}"
case "$step" in

  sft)
    echo "=== [sft] trajectory SFT @16K ctx, bf16, QUANT=none ==="
    # 16K covers 100% of the 474 traces (max ~12.7K tok). v25 ran 1536 and
    # truncated 47% — this is the strict-data-quality fix.
    QUANT=none \
    MODEL_NAME=Qwen/Qwen3.5-9B \
    ADAPTER=mateo0093/le-gros-chaton-qwen \
    TRACES_REPO=mateo0093/le-gros-chaton-traces \
    TRACES_FILE=agent_traces_normalized.jsonl \
    TRAJECTORY_CTX=16384 EPOCHS=3 BATCH=1 LR=2e-4 LORA_R=16 \
    OUT_DIR=qwen_traj_sft_16k \
    OUT_REPO=mateo0093/le-gros-chaton-qwen-traj-sft-16k \
    nohup "$PY" -u colab/trajectory_sft.py --no-upload \
      > logs/sft_16k.log 2>&1 &
    echo "SFT started (nohup) -> logs/sft_16k.log ; upload happens in merge step"
    ;;

  merge)
    echo "=== [merge] sequential bf16 fuse: base + Fable5 + traj(16k) ==="
    TRAJ_ADAPTER="$(pwd)/qwen_traj_sft_16k" \
    OUT_DIR=qwen_merged \
    MERGE_OUT_REPO=mateo0093/le-gros-chaton-qwen-merged \
    HF_TOKEN="$HF_TOKEN" "$PY" merge_sft.py 2>&1 | tee logs/merge.log
    echo "merged -> qwen_merged/ (uploaded if HF_TOKEN set)"
    ;;

  serve)
    echo "=== [serve] vLLM on qwen_merged (ROCm) ==="
    # Prefer pip-installed vllm[rocm]; fall back to the AMD docker image.
    if "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
      VLLM_ROCM_USE_AITER=1 "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
        --model qwen_merged --port 8000 \
        --dtype bfloat16 --max-model-len 32768
    else
      echo "vllm not in venv — use the AMD docker:"
      echo "  docker pull rocm/vllm-dev:nightly_main_20260211"
      echo "  docker run --device /dev/kfd --device /dev/dri --network host -v \$(pwd):/workspace rocm/vllm-dev:nightly_main_20260211 vllm serve /workspace/qwen_merged --port 8000 --dtype bfloat16 --max-model-len 32768"
    fi
    ;;

  eval)
    echo "=== [eval] Terminal-Bench 2.0, 5 attempts/task ==="
    "$PY" eval/tbench_eval.py \
      --model-server http://localhost:8000 \
      --model-name qwen_merged \
      --label "le-gros-chaton-16k" --adapter merged \
      --attempts 5 2>&1 | tee logs/tb_16k.log
    ;;

  rlvr)
    echo "=== [rlvr] GRPO with diversity + novelty on qwen_merged ==="
    QUANT=none MODEL_NAME="$(pwd)/qwen_merged" ADAPTER=none DEVICE_MAP=cuda:0 \
      "$PY" rlvr_qwen.py --n 8 --n-steps 120 --limit 19 \
      --novelty-bonus 0.2 --out qwen_rlvr 2>&1 | tee logs/rlvr.log
    echo "RLVR done -> qwen_rlvr/ ; merge into qwen_rlvr_merged then eval"
    ;;

  help|*)
    sed -n '1,14p' "$0"
    ;;
esac
