#!/bin/bash
# Hackatime heartbeat script for Pi agent coding sessions
# Sends a heartbeat ONLY when there is real activity (file edits) since the last check.
# Run from cron (every 2 min) — idle runs exit silently without pinging the API.

PROJECT_DIR="/home/mateo/le-gros-chaton"
WAKA_CLI="/home/mateo/.wakatime/wakatime-cli-linux-amd64"
STATE_FILE="$PROJECT_DIR/.hackatime-last-check"   # stores epoch seconds of the last sent heartbeat

# --- 1. Last check time ---
# Stored as fractional epoch (e.g. 1785953847.669) so files edited in the
# same second as the last heartbeat are not re-reported as new.
if [ -s "$STATE_FILE" ]; then
    LAST_CHECK=$(cat "$STATE_FILE")
else
    LAST_CHECK=0   # first run: treat everything as new (sends one heartbeat to bootstrap)
fi

NOW=$(date +%s.%N)

# --- 2. Detect edits since the last check ---
# Any project file modified after LAST_CHECK counts as activity.
# Ignore our own state/heartbeat files, VCS dirs, and generated/runtime noise.
NEWEST_FILE=$(find "$PROJECT_DIR" \
    -type f \
    -newermt "@$LAST_CHECK" \
    ! -path "$PROJECT_DIR/.git/*" \
    ! -path "$PROJECT_DIR/.pi/*" \
    ! -path "$PROJECT_DIR/.pi-glla/*" \
    ! -path "$PROJECT_DIR/.firecrawl/*" \
    ! -path "$PROJECT_DIR/.pytest_cache/*" \
    ! -path "$PROJECT_DIR/__pycache__/*" \
    ! -path "$PROJECT_DIR/kaggle/*" \
    ! -path "$PROJECT_DIR/kaggle_output/*" \
    ! -path "$PROJECT_DIR/kaggle_output2/*" \
    ! -path "$PROJECT_DIR/models/*" \
    ! -path "$PROJECT_DIR/*/__pycache__/*" \
    ! -name ".hackatime-*" \
    ! -name "*.pyc" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn \
    | head -1 \
    | cut -d' ' -f2-)

# --- 3. Send heartbeat only if there was activity ---
if [ -z "$NEWEST_FILE" ]; then
    # Nothing edited since the last heartbeat — stay quiet.
    exit 0
fi

"$WAKA_CLI" \
    --entity "$NEWEST_FILE" \
    --plugin "pi/1.0" \
    --project "le-gros-chaton" \
    --category "coding" \
    --sync-ai-activity \
    --lines-in-file "$(wc -l < "$NEWEST_FILE")" \
    --log-to-stdout 2>&1 | grep -v '"level":"warn"' || true

# Remember when we sent, so the next run only fires on NEW edits.
echo "$NOW" > "$STATE_FILE"
echo "✓ Heartbeat sent for $(basename "$NEWEST_FILE")"
