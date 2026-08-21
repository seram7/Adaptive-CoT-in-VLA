#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
trap cleanup_servers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

STATUS_LOG="$RUN_LOG_DIR/campaign25_status.log"
mkdir -p "$RUN_LOG_DIR"
touch "$STATUS_LOG"
CAMPAIGN_FAILED=0

"$SCRIPT_DIR/prepare_manifests.sh" || exit 1

start_gpu_monitor "$RUN_LOG_DIR/campaign25_gpu_memory.csv"
echo "[$(date -u +%FT%TZ)] START odd25 stride16 N=${NUM_UNCERTAINTY_SAMPLES} pilots" | tee -a "$STATUS_LOG"
start_pi05 1 "$RUN_LOG_DIR/pi05_server_n5_campaign25_pilot.log" || exit 1
PILOT_FAILED=0
for task in "${ODD25_TASKS[@]}"; do
  for condition in demo_clean demo_randomized; do
    [[ "$condition" == "demo_clean" ]] && episodes=$CLEAN_CALIBRATION_EPISODES || episodes=$RANDOM_CALIBRATION_EPISODES
    echo "[$(date -u +%FT%TZ)] START pilot ${task}/${condition}" | tee -a "$STATUS_LOG"
    if run_eval "$task" "$condition" pilot "$episodes" \
      "$CALIBRATION_MANIFEST_ROOT/$condition/${task}.json" \
      "$PILOT_ROOT/$condition/$task/pilot" \
      "$RUN_LOG_DIR/pilot_${task}_${condition}.log"; then
      echo "[$(date -u +%FT%TZ)] DONE pilot ${task}/${condition}" | tee -a "$STATUS_LOG"
    else
      echo "[$(date -u +%FT%TZ)] FAIL pilot ${task}/${condition}" | tee -a "$STATUS_LOG"
      PILOT_FAILED=1
    fi
  done
done
stop_pi05

if [[ "$PILOT_FAILED" -ne 0 ]]; then
  echo "[$(date -u +%FT%TZ)] PILOT_PASS_INCOMPLETE; requesting supervised retry" | tee -a "$STATUS_LOG"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] START odd25 stride16 step-threshold calibration" | tee -a "$STATUS_LOG"
(
  cd "$ADAPTIVE_COT_ROOT"
  "$ROBOTWIN_PYTHON" experiments/robotwin/estimate_pi05_step_thresholds.py \
    --pilot-root "$PILOT_ROOT" --tasks "${ODD25_TASKS[@]}" \
    --target-ratio "$COT_RATIO" --cooldown-steps "$COOLDOWN_STEPS" \
    --replan-stride "$REPLAN_STRIDE" --force-first-query \
    --output "$THRESHOLDS_JSON"
) >"$RUN_LOG_DIR/thresholds_campaign25.log" 2>&1 || exit 1
echo "[$(date -u +%FT%TZ)] DONE odd25 stride16 step-threshold calibration" | tee -a "$STATUS_LOG"

run_arm() {
  local task="$1" condition="$2" label="$3" mode="$4" episodes="$5"
  shift 5
  echo "[$(date -u +%FT%TZ)] START ${task}/${condition}/${label}" | tee -a "$STATUS_LOG"
  if run_eval "$task" "$condition" "$mode" "$episodes" \
    "$MANIFEST_ROOT/$condition/screening/${task}.json" \
    "$MAIN_ROOT/$task/$condition/$label" \
    "$RUN_LOG_DIR/main25_${task}_${condition}_${label}.log" "$@"; then
    echo "[$(date -u +%FT%TZ)] DONE ${task}/${condition}/${label}" | tee -a "$STATUS_LOG"
  else
    echo "[$(date -u +%FT%TZ)] FAIL ${task}/${condition}/${label}" | tee -a "$STATUS_LOG"
    CAMPAIGN_FAILED=1
  fi
}

echo "[$(date -u +%FT%TZ)] START odd25 stride16 five-arm main campaign" | tee -a "$STATUS_LOG"
for task in "${ODD25_TASKS[@]}"; do
  echo "[$(date -u +%FT%TZ)] TASK_START ${task}" | tee -a "$STATUS_LOG"
  if ! start_zr0 "$RUN_LOG_DIR/zr0_server_main25_${task}.log"; then
    echo "[$(date -u +%FT%TZ)] TASK_ABORT ${task}: ZR-0 startup" | tee -a "$STATUS_LOG"
    CAMPAIGN_FAILED=1
    continue
  fi
  if ! start_pi05 0 "$RUN_LOG_DIR/pi05_server_nofm_main25_${task}.log"; then
    echo "[$(date -u +%FT%TZ)] TASK_ABORT ${task}: PI0.5 startup" | tee -a "$STATUS_LOG"
    CAMPAIGN_FAILED=1
    stop_zr0
    continue
  fi

  for condition in demo_clean demo_randomized; do
    [[ "$condition" == "demo_clean" ]] && episodes=$CLEAN_MAIN_EPISODES || episodes=$RANDOM_MAIN_EPISODES
    run_arm "$task" "$condition" baseline_pi05_only baseline "$episodes"
    run_arm "$task" "$condition" baseline_zr0_only_direct fixed "$episodes" --cot-ratio 1.0
    run_arm "$task" "$condition" fixed fixed "$episodes"
    run_arm "$task" "$condition" random random "$episodes" --force-first-query
  done

  stop_pi05
  if ! start_pi05 1 "$RUN_LOG_DIR/pi05_server_n5_main25_${task}.log"; then
    echo "[$(date -u +%FT%TZ)] TASK_ABORT ${task}: PI0.5 N=${NUM_UNCERTAINTY_SAMPLES} startup" | tee -a "$STATUS_LOG"
    CAMPAIGN_FAILED=1
    stop_zr0
    continue
  fi
  for condition in demo_clean demo_randomized; do
    [[ "$condition" == "demo_clean" ]] && episodes=$CLEAN_MAIN_EPISODES || episodes=$RANDOM_MAIN_EPISODES
    run_arm "$task" "$condition" adaptive_sampling_farmass_replan_max \
      sampling_farmass_replan_max "$episodes" \
      --thresholds-json "$THRESHOLDS_JSON" \
      --cooldown-steps "$COOLDOWN_STEPS" --force-first-query
  done

  stop_pi05
  stop_zr0
  echo "[$(date -u +%FT%TZ)] TASK_DONE ${task}" | tee -a "$STATUS_LOG"
done

if [[ "$CAMPAIGN_FAILED" -eq 0 ]]; then
  echo "CAMPAIGN25_ODD_STRIDE16_COOLDOWN32_DONE" | tee -a "$STATUS_LOG"
else
  echo "CAMPAIGN25_ODD_STRIDE16_COOLDOWN32_COMPLETED_WITH_FAILURES" | tee -a "$STATUS_LOG"
  exit 1
fi
