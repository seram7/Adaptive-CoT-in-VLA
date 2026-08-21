#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/env.local" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_PATH=$(realpath "$1")
export ROBOTWIN_EXPERIMENT_ENV="$ENV_PATH"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

mkdir -p "$RUN_LOG_DIR"
CONSOLE_LOG="$RUN_LOG_DIR/supervisor_console.log"
setsid -f env ROBOTWIN_EXPERIMENT_ENV="$ENV_PATH" \
  /bin/bash "$SCRIPT_DIR/supervise.sh" >"$CONSOLE_LOG" 2>&1 </dev/null
sleep 2

echo "Detached odd25 campaign launched."
echo "Supervisor log: $RUN_LOG_DIR/campaign25_supervisor.log"
echo "Campaign log:   $RUN_LOG_DIR/campaign25_status.log"
echo "Console log:    $CONSOLE_LOG"
