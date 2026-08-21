#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

STATUS_LOG="$RUN_LOG_DIR/manifest_status.log"
mkdir -p "$RUN_LOG_DIR" "$CALIBRATION_MANIFEST_ROOT" "$MANIFEST_ROOT"
touch "$STATUS_LOG"
failed=0

prepare_one() {
  local task="$1" condition="$2" episodes="$3" seed_block="$4" output="$5" log="$6"
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  (
    cd "$ADAPTIVE_COT_ROOT"
    CUDA_VISIBLE_DEVICES="$SIM_CUDA_DEVICE" MESA_VK_DEVICE_SELECT="$SIM_MESA_DEVICE" \
      "$ROBOTWIN_PYTHON" -X faulthandler experiments/robotwin/prepare_pi05_seeds.py \
      --robotwin-root "$ROBOTWIN_ROOT" --task "$task" --task-config "$condition" \
      --episodes "$episodes" --seed-block "$seed_block" --output "$output"
  ) >"$log" 2>&1
}

for task in "${ODD25_TASKS[@]}"; do
  for condition in demo_clean demo_randomized; do
    if [[ "$condition" == "demo_clean" ]]; then
      calibration_episodes=$CLEAN_CALIBRATION_EPISODES
      main_episodes=$CLEAN_MAIN_EPISODES
    else
      calibration_episodes=$RANDOM_CALIBRATION_EPISODES
      main_episodes=$RANDOM_MAIN_EPISODES
    fi
    echo "[$(date -u +%FT%TZ)] START calibration manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
    if prepare_one "$task" "$condition" "$calibration_episodes" 4 \
      "$CALIBRATION_MANIFEST_ROOT/$condition/${task}.json" \
      "$RUN_LOG_DIR/manifest_calibration_${task}_${condition}.log"; then
      echo "[$(date -u +%FT%TZ)] DONE calibration manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
    else
      echo "[$(date -u +%FT%TZ)] FAIL calibration manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
      failed=1
      continue
    fi

    echo "[$(date -u +%FT%TZ)] START main manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
    if prepare_one "$task" "$condition" "$main_episodes" 3 \
      "$MANIFEST_ROOT/$condition/screening/${task}.json" \
      "$RUN_LOG_DIR/manifest_main_${task}_${condition}.log"; then
      echo "[$(date -u +%FT%TZ)] DONE main manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
    else
      echo "[$(date -u +%FT%TZ)] FAIL main manifest ${task}/${condition}" | tee -a "$STATUS_LOG"
      failed=1
    fi
  done
done

echo "MANIFEST_PREPARATION_DONE" | tee -a "$STATUS_LOG"
exit "$failed"
