#!/bin/bash
# Hackatime heartbeat script for Pi agent coding sessions
# Sends heartbeats and syncs AI activity from Pi transcripts

PROJECT_DIR="/home/mateo/le-gros-chaton"
WAKA_CLI="/home/mateo/.wakatime/wakatime-cli-linux-amd64"
CURRENT_FILE=""

# Find the most recently modified .py file as the active entity
CURRENT_FILE=$(find "$PROJECT_DIR" -maxdepth 1 -name "*.py" -newer "$PROJECT_DIR/.hackatime-heartbeat.sh" 2>/dev/null | head -1)

# If no recently changed file, pick the latest modified .py
if [ -z "$CURRENT_FILE" ]; then
    CURRENT_FILE=$(ls -t "$PROJECT_DIR"/*.py 2>/dev/null | head -1)
fi

if [ -n "$CURRENT_FILE" ]; then
    "$WAKA_CLI" \
        --entity "$CURRENT_FILE" \
        --plugin "pi/1.0" \
        --project "le-gros-chaton" \
        --category "coding" \
        --sync-ai-activity \
        --lines-in-file "$(wc -l < "$CURRENT_FILE")" \
        --log-to-stdout 2>&1 | grep -v '"level":"warn"' || true
    echo "✓ Heartbeat sent for $(basename "$CURRENT_FILE")"
else
    # Fallback: just sync AI activity
    "$WAKA_CLI" \
        --sync-ai-activity \
        --plugin "pi/1.0" \
        --log-to-stdout 2>&1 | grep -v '"level":"warn"' || true
    echo "✓ AI activity synced"
fi