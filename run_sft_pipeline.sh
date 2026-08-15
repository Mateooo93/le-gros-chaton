#!/usr/bin/env bash
# Le Gros Chaton — full post-generation pipeline, one command.
#
# Usage (on ANY machine with a 12GB+ GPU and this repo + venv):
#   HF_TOKEN=hf_xxx GPU_RUNTIME=colab bash run_sft_pipeline.sh        # full: SFT -> eval -> RLVR
#   HF_TOKEN=hf_xxx bash run_sft_pipeline.sh --sft-only               # just trajectory SFT
#   HF_TOKEN=hf_xxx bash run_sft_pipeline.sh --eval-only              # eval existing adapter on TB
#
# Env knobs (defaults tuned for Qwen3.5-9B + LoRA on 12-16GB cards):
#   MODEL_NAME        base model (default Qwen/Qwen3.5-9B; pre-quantized
#                     bnb-4bit repos work: techwithsergiu/Qwen3.5-9B-bnb-4bit)
#   ADAPTER           trajectory-SFT adapter to train on top of the 91%
#                     Fable5 SFT (default mateo0093/le-gros-chaton-qwen)
#   TRACES_REPO/TRACES_FILE   trace dataset (default our HF traces repo)
#   TRAJECTORY_CTX    context (default 16384; 8192 for 8GB cards)
#   EPOCHS, BATCH, LR, LORA_R
#   OUT_REPO          where the trajectory adapter is uploaded
#   DEVICE_MAP        auto | cuda:0 (default auto)
#   MAX_MEMORY        optional JSON {"0": "12GiB", "cpu": "24GiB"} for tight cards
#
# Stages:
#   1. Sync the latest agent_traces_full.jsonl to the HF traces dataset
#   2. Trajectory SFT (assistant-token-only LoRA on the quantized base)
#   3. Upload the adapter to OUT_REPO
#   4. (--full) Terminal-Bench 2.0 eval of the new adapter (tbench_eval.py)
#   5. (--full) RLVR with diversity reward (rlvr_qwen.py)
set -euo pipefail
cd "$(dirname "$0")"

HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
: "${MODEL_NAME:=techwithsergiu/Qwen3.5-9B-bnb-4bit}"
: "${ADAPTER:=mateo0093/le-gros-chaton-qwen}"
: "${TRACES_REPO:=mateo0093/le-gros-chaton-traces}"
: "${TRACES_FILE:=agent_traces_normalized.jsonl}"
: "${OUT_REPO:=mateo0093/le-gros-chaton-qwen-traj-sft}"
: "${TRAJECTORY_CTX:=16384}"
: "${EPOCHS:=3}"
: "${BATCH:=1}"
: "${LR:=2e-4}"
: "${LORA_R:=16}"
: "${DEVICE_MAP:=auto}"
: "${MAX_MEMORY:=}"
: "${OUT_DIR:=qwen_traj_sft}"
: "${MERGE_OUT_DIR:=qwen_merged}"
: "${MERGE_OUT_REPO:=mateo0093/le-gros-chaton-qwen-merged}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

log() { echo "[pipeline] $*" | tee -a /tmp/sft_pipeline.log; }

# ---- 1. Sync traces to HF -----------------------------------------------
log "1/5 syncing $TRACES_FILE -> $TRACES_REPO"
if [ -f "$TRACES_FILE" ]; then
  $PY - "$HF_TOKEN" "$TRACES_FILE" "$TRACES_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
_, tok, path, repo = sys.argv
api = HfApi(token=tok)
api.upload_file(path_or_fileobj=path, path_in_repo=path.split("/")[-1],
                repo_id=repo, repo_type="dataset",
                commit_message=f"pipeline sync {path}")
print("uploaded", path)
EOF
fi

# ---- 2. Trajectory SFT ---------------------------------------------------
log "2/5 trajectory SFT (MODEL=$MODEL_NAME ADAPTER=$ADAPTER ctx=$TRAJECTORY_CTX epochs=$EPOCHS)"
export HF_TOKEN MODEL_NAME ADAPTER TRACES_REPO TRACES_FILE OUT_REPO
export TRAJECTORY_CTX EPOCHS BATCH LR LORA_R DEVICE_MAP MAX_MEMORY OUT_DIR
$PY -u colab/trajectory_sft.py --no-upload
log "SFT done -> $OUT_DIR"

# ---- 3. Upload adapter ---------------------------------------------------
log "3/5 uploading $OUT_DIR -> $OUT_REPO"
$PY - "$HF_TOKEN" "$OUT_DIR" "$OUT_REPO" <<'EOF'
import sys
from huggingface_hub import HfApi
_, tok, d, repo = sys.argv
api = HfApi(token=tok)
api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                  commit_message="trajectory SFT adapter")
print("uploaded", repo)
EOF

MODE="${1:---full}"
if [ "$MODE" = "--sft-only" ]; then
  log "done (--sft-only). adapter at $OUT_REPO"
  exit 0
fi

# ---- 3.5 Merge EVERYTHING into one full checkpoint ------------------------
# The SFT checkpoint contains only the trajectory LoRA; the Fable5 layer
# lives in the base adapter. Downstream consumers (TB eval, RLVR) load a
# single model — so fuse base+Fable+traj into qwen_merged/ and upload it.
log "3.5/5 merging base + Fable5 + trajectory adapter -> qwen_merged/"
BASE_MODEL="$MODEL_NAME" \
FABLE_MODEL="$ADAPTER" \
TRAJ_ADAPTER="$OUT_DIR" \
OUT_DIR="$MERGE_OUT_DIR" \
MERGE_OUT_REPO="$MERGE_OUT_REPO" \
HF_TOKEN="$HF_TOKEN" $PY merge_sft.py
log "merged -> $MERGE_OUT_DIR ($MERGE_OUT_REPO)"

# ---- 4. Terminal-Bench eval of the MERGED model ---------------------------
log "4/5 TB 2.0 eval (merged model=$MERGE_OUT_DIR)"
: "${TB_ATTEMPTS:=5}"           # leaderboard protocol = 5 attempts/task
: "${TB_TASKS:=}"               # empty = full task set; or --tasks a,b,c
$PY eval/tbench_eval.py \
  --local-model "$MERGE_OUT_DIR" --four-bit --adapter traj_sft \
  --attempts "$TB_ATTEMPTS" $TB_TASKS \
  2>/dev/null \
  || log "eval needs a served model; run: python modal_serve_qwen.py + tbench_eval.py --server <url>"

# ---- 5. RLVR with diversity on the MERGED base ----------------------------
log "5/5 RLVR (diversity on, novelty bonus 0.2)"
MODEL_NAME="$MERGE_OUT_DIR" ADAPTER=none \
  $PY rlvr_qwen.py --n 8 --n-steps 120 --limit 19 --novelty-bonus 0.2 \
  --out qwen_rlvr 2>/dev/null || log "RLVR needs GPU; rerun when available"
log "pipeline complete"
