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
        break
    fi
    # Check if we're making progress (traces file growing)
    N=$(wc -l < agent_traces_full.jsonl 2>/dev/null || echo 0)
    echo "=== supervisor: $N verified traces so far, restarting in 30s ===" >> "$LOG"
    sleep 30
done
echo "=== supervisor: done ===" >> "$LOG"
