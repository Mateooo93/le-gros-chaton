#!/bin/bash
# Hackatime heartbeat script for Pi agent / OMP coding sessions.
#
# Tracks time even when no file is being edited: the cron fires every 2 minutes
# and sends a heartbeat on EVERY run, so Hackatime counts continuous time.
#
# The tracked project is derived from the MOST RECENTLY ACTIVE OMP session
# (~/.pi/agent/sessions/**/*.jsonl). Each session's first line records its
# working directory ("cwd"), so whichever project you last worked on in OMP is
# reported as the project. When no OMP session is found, falls back to a
# default project.

# --- Configuration ---
DEFAULT_PROJECT_DIR="/home/mateo/le-gros-chaton"
WAKA_CLI="/home/mateo/.wakatime/wakatime-cli-linux-amd64"
SESSION_ROOT="$HOME/.pi/agent/sessions"
STATE_FILE="$HOME/.hackatime-last-check"   # per-user; shared across projects

# --- 1. Resolve the active project from the newest OMP session ---
# OMP writes one jsonl per session; the first line is a "session" record
# carrying {"cwd":"/abs/path",...}. Newest mtime = the session you're using.
PROJECT_DIR="$DEFAULT_PROJECT_DIR"
NEWEST_SESSION=$(find "$SESSION_ROOT" -type f -name '*.jsonl' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$NEWEST_SESSION" ]; then
    CWD=$(sed -n '1p' "$NEWEST_SESSION" 2>/dev/null \
        | sed -n 's/.*"cwd":"\([^"]*\)".*/\1/p')
    # Only adopt it if it is a real (non-{home,root}) directory that exists.
    if [ -n "$CWD" ] && [ "$CWD" != "$HOME" ] && [ "$CWD" != "/" ] && [ -d "$CWD" ]; then
        PROJECT_DIR="$CWD"
    fi
fi

PROJECT_NAME=$(basename "$PROJECT_DIR")

# --- 2. Last heartbeat time ---
if [ -s "$STATE_FILE" ]; then
    LAST_CHECK=$(cat "$STATE_FILE")
else
    LAST_CHECK=0
fi
NOW=$(date +%s.%N)

# --- 3. Pick the file entity ---
# Prefer a file edited since the last heartbeat; fall back to the newest
# project file so idle heartbeats land on a real, project-attributed file.
newest_file() {  # $1 = extra find predicates
    find "$PROJECT_DIR" \
        -type f \
        "$@" \
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
        | sort -rn | head -1 | cut -d' ' -f2-
}

NEWEST_FILE=$(newest_file -newermt "@$LAST_CHECK")
if [ -z "$NEWEST_FILE" ]; then
    NEWEST_FILE=$(newest_file)
fi

# Extreme safety net: never send an empty entity.
if [ -z "$NEWEST_FILE" ]; then
    exit 0
fi

# --- 4. Send a heartbeat (every run) ---
"$WAKA_CLI" \
    --entity "$NEWEST_FILE" \
    --plugin "pi/1.0" \
    --project "$PROJECT_NAME" \
    --category "coding" \
    --sync-ai-activity \
    --lines-in-file "$(wc -l < "$NEWEST_FILE")" \
    --log-to-stdout 2>&1 | grep -v '"level":"warn"' || true

echo "$NOW" > "$STATE_FILE"
echo "✓ Heartbeat sent for $(basename "$NEWEST_FILE") (project: $PROJECT_NAME)"