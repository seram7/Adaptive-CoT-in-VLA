#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

mkdir -p "$RUN_LOG_DIR"
SUPERVISOR_LOG="$RUN_LOG_DIR/campaign25_supervisor.log"
touch "$SUPERVISOR_LOG"

exec 9>"$EXPERIMENT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another odd25 supervisor already owns $EXPERIMENT_ROOT/supervisor.lock" >&2
  exit 2
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date -u +%FT%TZ)] START odd25 campaign attempt=$attempt" | tee -a "$SUPERVISOR_LOG"
  if "$SCRIPT_DIR/run_campaign.sh"; then
    echo "[$(date -u +%FT%TZ)] DONE odd25 campaign attempt=$attempt" | tee -a "$SUPERVISOR_LOG"
    exit 0
  else
    rc=$?
  fi
  echo "[$(date -u +%FT%TZ)] RETRY odd25 campaign attempt=$attempt rc=$rc after=30s" | tee -a "$SUPERVISOR_LOG"
  sleep 30
done
