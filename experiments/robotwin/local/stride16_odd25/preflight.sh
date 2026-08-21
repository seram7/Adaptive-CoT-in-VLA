#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

required_files=(
  "$PI05_CHECKPOINT/model.safetensors"
  "$PI05_CHECKPOINT/assets/pi0.5_clean_randomize_joint_training/norm_stats.json"
  "$ZR0_ROBOTWIN_METADATA/meta/info.json"
  "$ZR0_ROBOTWIN_METADATA/meta/stats.json"
  "$ZR0_ROBOTWIN_METADATA/meta/tasks.jsonl"
  "$ZR0_ROBOTWIN_METADATA/meta/episodes.jsonl"
  "$OPENPI_ROOT/scripts/serve_motus_robotwin.py"
  "$OPENPI_ROOT/src/openpi/uncertainty.py"
  "$ZR0_ROOT/server.py"
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 2; }
done

for asset_dir in embodiments objects background_texture; do
  [[ -d "$ROBOTWIN_ROOT/assets/$asset_dir" ]] || {
    echo "Missing RoboTwin asset directory: $ROBOTWIN_ROOT/assets/$asset_dir" >&2
    exit 2
  }
done
for task in "${ODD25_TASKS[@]}"; do
  [[ -f "$ROBOTWIN_ROOT/envs/${task}.py" ]] || {
    echo "Missing odd task implementation: $ROBOTWIN_ROOT/envs/${task}.py" >&2
    exit 2
  }
done

for port in "$PI05_PORT" "$ZR0_PORT"; do
  if ss -ltn "sport = :$port" | tail -n +2 | grep -q .; then
    echo "Port $port is already in use" >&2
    exit 2
  fi
done

"$ROBOTWIN_PYTHON" -c 'import h5py, msgpack, numpy, sapien, websockets, yaml'
(
  cd "$OPENPI_ROOT"
  "$OPENPI_PYTHON" -c \
    'from openpi.policies.sampling_far_mass_policy import SamplingFarMassPolicy; from openpi.uncertainty import FarMassConfig'
)
(
  cd "$ZR0_ROOT"
  "$ZR0_PYTHON" -c 'from policies.reasoning_vla_policy import ZR0Policy'
)

nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
printf 'odd_tasks=%s\n' "${#ODD25_TASKS[@]}"
printf 'preflight=ok\n'
