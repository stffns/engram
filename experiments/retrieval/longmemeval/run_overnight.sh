#!/usr/bin/env bash
#
# Overnight run of LongMemEval full split (n=500) for both
# merken-heuristic and raw vstash baselines. Phase A of the
# benchmark roadmap.
#
# Expected wall-clock: ~6 hours per baseline on current hardware
# → ~12 hours total. Disowned from the terminal so closing the
# shell doesn't kill it. All stdout/stderr captured to a log file
# with a timestamp in its name.
#
# Usage:
#
#     cd ~/Desktop/Personal/Projects/merken
#     experiments/retrieval/longmemeval/run_overnight.sh
#
# Then check progress with:
#
#     tail -f experiments/retrieval/longmemeval/.cache/overnight_*.log
#
# After it finishes, copy the final lines into RESULTS.md with:
#   - the merken commit SHA (`git rev-parse HEAD`)
#   - the date of the run
#   - n=500
#   - R@5 with the bootstrap CI the runner prints
#   - API calls per query: 0 (merken is local-first)
#   - wall-clock elapsed
#
# NEVER paste sample results as full-run results. If the run dies
# halfway, the row does not land. Re-run or investigate before
# publishing anything.

set -euo pipefail

cd "$(dirname "$0")/../../.."

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="experiments/retrieval/longmemeval/.cache"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/overnight_${TS}.log"

echo "=== merken longmemeval overnight run — ${TS} ===" | tee "$LOG"
echo "commit: $(git rev-parse HEAD)" | tee -a "$LOG"
echo "branch: $(git rev-parse --abbrev-ref HEAD)" | tee -a "$LOG"
echo "cwd:    $(pwd)" | tee -a "$LOG"
echo "log:    $LOG" | tee -a "$LOG"
echo "===============================================" | tee -a "$LOG"

nohup python3 -m experiments.retrieval.longmemeval.runner \
    --subset longmemeval_s \
    --questions 500 \
    --seed 42 \
    --top-k 5 \
    --baseline vstash \
    --baseline merken-heuristic \
    >> "$LOG" 2>&1 &

PID=$!
echo "runner PID: $PID"
echo "running in background. check progress:"
echo "  tail -f $LOG"
echo
echo "to kill if needed:"
echo "  kill $PID"
echo
echo "$PID" > "$LOG_DIR/overnight_${TS}.pid"
