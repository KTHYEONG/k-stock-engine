#!/usr/bin/env bash
# Detached stocks ML full-run launcher (H10-only pre-registered scope).
#
# Usage: tools/run_train_detached.sh <artifact-id> [extra train args...]
#
# Runs `src.stocks.cli.train` via nohup so the process survives the launching
# session; pid/log stay under scratch/ (repo-local temp policy). Poll
# scratch/<artifact-id>.log or docs/results/ml_runs/latest.json for progress.
#
# Canonical flags pin the measured-stable scope: H10-only grid (24 cells,
# Holm family m=72), rolling lookback 1260, forward holdout 252.
# Memory guard 4096 MiB is ~2.9x the measured peak (1.42 GiB) for this scope;
# --memory-reserve-mib subtracts concurrent workloads from host headroom only.
# NOTE: a full 48-cell grid doubles the family to m=144 and requires an
# explicit --bootstrap-resamples >= 2880 (the resolution guard enforces it).
set -euo pipefail

ARTIFACT_ID="${1:?usage: run_train_detached.sh <artifact-id> [extra train args...]}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="$ROOT/scratch"
mkdir -p "$SCRATCH"

LOG="$SCRATCH/${ARTIFACT_ID}.log"
PID_FILE="$SCRATCH/${ARTIFACT_ID}.pid"

nohup uv run python -m src.stocks.cli.train \
  --artifact-id "$ARTIFACT_ID" \
  --candidate-horizon-sessions 10 \
  --candidate-rebalance-frequency-sessions 5,10 \
  --candidate-top-k 12,16,20,24 \
  --fold-count 3 \
  --embargo-sessions 5 \
  --forward-holdout-sessions 252 \
  --max-training-lookback-sessions 1260 \
  --max-rss-mib 4096 \
  --memory-reserve-mib 2048 \
  --bootstrap-alpha 0.05 \
  --bootstrap-resamples 2000 \
  --model-threads 4 \
  "$@" >"$LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "detached pid $(cat "$PID_FILE") | log: $LOG"
