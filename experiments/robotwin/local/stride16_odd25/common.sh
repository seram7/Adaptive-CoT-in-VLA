#!/usr/bin/env bash

ODD25_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ODD25_DEFAULT_ROOT=$(cd "$ODD25_SCRIPT_DIR/../../../.." && pwd)

if [[ -z "${ROBOTWIN_EXPERIMENT_ENV:-}" || ! -f "$ROBOTWIN_EXPERIMENT_ENV" ]]; then
  echo "Set ROBOTWIN_EXPERIMENT_ENV to an absolute env.local path" >&2
  return 2 2>/dev/null || exit 2
fi
# shellcheck disable=SC1090
source "$ROBOTWIN_EXPERIMENT_ENV"

ADAPTIVE_COT_ROOT=${ADAPTIVE_COT_ROOT:-$ODD25_DEFAULT_ROOT}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ADAPTIVE_COT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MACHINE_ID
export CUDA_DEVICE_ORDER=PCI_BUS_ID

required_variables=(
  ADAPTIVE_COT_ROOT ROBOTWIN_ROOT OPENPI_ROOT ZR0_ROOT
  PI05_CHECKPOINT ZR0_CHECKPOINT ZR0_ROBOTWIN_METADATA
  ROBOTWIN_PYTHON OPENPI_PYTHON ZR0_PYTHON
  SIM_CUDA_DEVICE PI05_CUDA_DEVICE SIM_MESA_DEVICE
  PI05_HOST PI05_PORT ZR0_HOST ZR0_PORT
  EXPERIMENT_ROOT MANIFEST_ROOT CALIBRATION_MANIFEST_ROOT PILOT_ROOT
  THRESHOLDS_JSON MAIN_ROOT RUN_LOG_DIR
  CLEAN_CALIBRATION_EPISODES RANDOM_CALIBRATION_EPISODES
  CLEAN_MAIN_EPISODES RANDOM_MAIN_EPISODES NUM_UNCERTAINTY_SAMPLES
  ACTION_STEPS REPLAN_STRIDE COOLDOWN_STEPS COT_RATIO
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required variable $variable_name in $ROBOTWIN_EXPERIMENT_ENV" >&2
    return 2 2>/dev/null || exit 2
  fi
done

for required_dir in \
  "$ADAPTIVE_COT_ROOT" "$ROBOTWIN_ROOT" "$OPENPI_ROOT" "$ZR0_ROOT" \
  "$PI05_CHECKPOINT" "$ZR0_CHECKPOINT" "$ZR0_ROBOTWIN_METADATA"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required directory does not exist: $required_dir" >&2
    return 2 2>/dev/null || exit 2
  fi
done
for python_bin in "$ROBOTWIN_PYTHON" "$OPENPI_PYTHON" "$ZR0_PYTHON"; do
  if [[ ! -x "$python_bin" ]]; then
    echo "Python executable does not exist: $python_bin" >&2
    return 2 2>/dev/null || exit 2
  fi
done

mapfile -t ODD25_TASKS < <(
  "$ROBOTWIN_PYTHON" -c \
    'from experiments.robotwin.pi05_task_registry import PI05_ODD_TASK_NAMES; print("\n".join(PI05_ODD_TASK_NAMES))'
)
if [[ "${#ODD25_TASKS[@]}" -ne 25 ]]; then
  echo "Expected 25 odd-indexed tasks, found ${#ODD25_TASKS[@]}" >&2
  return 2 2>/dev/null || exit 2
fi

PI05_PID=""
ZR0_PID=""
GPU_MONITOR_PID=""

wait_for_health() {
  local url="$1" pid="$2" label="$3"
  for _ in $(seq 1 300); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited during startup" >&2
      return 1
    fi
    sleep 2
  done
  echo "$label health check timed out" >&2
  return 1
}

start_pi05() {
  local compute_far_mass="$1" log_path="$2"
  local extra=()
  if [[ "$compute_far_mass" == "1" ]]; then
    extra=(--compute-far-mass --num-uncertainty-samples "$NUM_UNCERTAINTY_SAMPLES")
  fi
  mkdir -p "$(dirname "$log_path")"
  (
    cd "$OPENPI_ROOT"
    exec env CUDA_VISIBLE_DEVICES="$PI05_CUDA_DEVICE" JAX_PLATFORMS=cpu \
      XLA_PYTHON_CLIENT_PREALLOCATE=false "$OPENPI_PYTHON" \
      scripts/serve_motus_robotwin.py --checkpoint-dir "$PI05_CHECKPOINT" \
      --port "$PI05_PORT" "${extra[@]}"
  ) >"$log_path" 2>&1 &
  PI05_PID=$!
  wait_for_health "http://${PI05_HOST}:${PI05_PORT}/healthz" "$PI05_PID" "PI0.5"
}

stop_pi05() {
  if [[ -n "$PI05_PID" ]]; then
    kill "$PI05_PID" 2>/dev/null || true
    wait "$PI05_PID" 2>/dev/null || true
    PI05_PID=""
  fi
}

start_zr0() {
  local log_path="$1" ld_preload=""
  local env_prefix=${ZR0_PYTHON%/bin/python}
  [[ -e "$env_prefix/lib/libstdc++.so.6" ]] && ld_preload="$env_prefix/lib/libstdc++.so.6"
  mkdir -p "$(dirname "$log_path")"
  (
    cd "$ZR0_ROOT"
    exec env CUDA_VISIBLE_DEVICES="$SIM_CUDA_DEVICE" LD_PRELOAD="$ld_preload" \
      "$ZR0_PYTHON" -X faulthandler server.py \
      --dataset_entry robotwin2.0-aloha-agilex \
      --dataset_path "$ZR0_ROBOTWIN_METADATA" \
      --ckpt_dir "$ZR0_CHECKPOINT" --inference_mode direct_action \
      --window_size 1 --port "$ZR0_PORT"
  ) >"$log_path" 2>&1 &
  ZR0_PID=$!
  wait_for_health "http://${ZR0_HOST}:${ZR0_PORT}/healthz" "$ZR0_PID" "ZR-0"
}

stop_zr0() {
  if [[ -n "$ZR0_PID" ]]; then
    kill "$ZR0_PID" 2>/dev/null || true
    wait "$ZR0_PID" 2>/dev/null || true
    ZR0_PID=""
  fi
}

start_gpu_monitor() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  printf 'timestamp,index,uuid,name,memory_used_mib,utilization_gpu_pct\n' >"$output"
  (
    while true; do
      stamp=$(date -u +%FT%TZ)
      nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | sed "s/^/${stamp},/" >>"$output"
      sleep 2
    done
  ) &
  GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}

cleanup_servers() {
  stop_gpu_monitor
  stop_pi05
  stop_zr0
}

run_eval() {
  local task="$1" task_config="$2" mode="$3" episodes="$4"
  local manifest="$5" output_dir="$6" log_path="$7"
  shift 7
  mkdir -p "$output_dir" "$(dirname "$log_path")"
  (
    cd "$ADAPTIVE_COT_ROOT"
    CUDA_VISIBLE_DEVICES="$SIM_CUDA_DEVICE" MESA_VK_DEVICE_SELECT="$SIM_MESA_DEVICE" \
      "$ROBOTWIN_PYTHON" -X faulthandler experiments/robotwin/evaluate_pi05_zr0.py \
      --robotwin-root "$ROBOTWIN_ROOT" \
      --openpi-root "$OPENPI_ROOT" --pi05-checkpoint "$PI05_CHECKPOINT" \
      --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
      --zr0-root "$ZR0_ROOT" --zr0-host "$ZR0_HOST" --zr0-port "$ZR0_PORT" \
      --zr0-inference-mode direct_action \
      --task "$task" --task-config "$task_config" --manifest "$manifest" \
      --mode "$mode" --cot-ratio "$COT_RATIO" --action-steps "$ACTION_STEPS" \
      --replan-stride "$REPLAN_STRIDE" --episodes "$episodes" \
      --instruction-type unseen --output-dir "$output_dir" "$@"
  ) >"$log_path" 2>&1
}
