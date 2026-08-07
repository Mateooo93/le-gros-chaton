#!/usr/bin/env bash
# Supervisor for teacher trajectory generation.
# Restarts the generator on crash (API outages kill the process); the
# generator itself is incremental + resume-capable, so restarts never
# lose already-verified traces.
#
# Usage: bash supervise_teacher_gen.sh [--n 40] [--samples 2]
set -u
cd /home/mateo/le-gros-chaton
LOG=/tmp/teacher_gen.log
ARGS="${*:---n 40 --samples 2 --temp 0.7}"

MAX_RESTARTS=50
for i in $(seq 1 $MAX_RESTARTS); do
    echo "=== supervisor: attempt $i ($(date)) ===" >> "$LOG"
    .venv/bin/python -u teacher_trajectories.py $ARGS >> "$LOG" 2>&1
    RC=$?
    echo "=== supervisor: attempt $i exited rc=$RC ($(date)) ===" >> "$LOG"
    if [ $RC -eq 0 ]; then
        # generator ran to completion (all done or all skipped)
        echo "=== supervisor: generator finished cleanly, stopping ===" >> "$LOG"
        # Sync final traces + normalized copy to HF so the SFT pipeline
        # sees everything without manual steps.
        N=$(wc -l < agent_traces_full.jsonl 2>/dev/null || echo 0)
        if [ "$N" -gt 0 ]; then
            echo "=== supervisor: syncing $N traces to HF ===" >> "$LOG"
            .venv/bin/python colab/normalize_traces.py >> "$LOG" 2>&1 || true
            HF_TOKEN="HF_TOKEN_PLACEHOLDER" .venv/bin/python - "$N" >> "$LOG" 2>&1 <<'PYEOF' || true
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
n = sys.argv[1]
for f in ("agent_traces_full.jsonl", "agent_traces_normalized.jsonl"):
    api.upload_file(path_or_fileobj=f, path_in_repo=f,
                    repo_id="mateo0093/le-gros-chaton-traces",
                    repo_type="dataset",
                    commit_message=f"final sync {n} traces")
print(f"synced {n} traces")
PYEOF
        fi
        break
    fi
    # Check if we're making progress (traces file growing)
    N=$(wc -l < agent_traces_full.jsonl 2>/dev/null || echo 0)
    echo "=== supervisor: $N verified traces so far, restarting in 30s ===" >> "$LOG"
    sleep 30
done
echo "=== supervisor: done ===" >> "$LOG"
