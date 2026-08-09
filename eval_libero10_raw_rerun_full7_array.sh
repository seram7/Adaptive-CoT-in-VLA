#!/bin/bash
#SBATCH --account=rrg-florian7_gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --job-name=l10_raw_rerun
#SBATCH --output=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs_raw_rerun/libero10_raw_rerun_%A_%a.out
#SBATCH --error=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs_raw_rerun/libero10_raw_rerun_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"

CONDITIONS=(far_mass_raw_top far_mass_raw_bottom)
DIRECTIONS=(gt lt)
TARGETS=(0.2 0.4 0.6 0.8 1.0)
TRIAL_CHUNKS=(0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49)

declare -A THRESHOLDS=(
  [far_mass_raw_top,0.2]=0.283078474856
  [far_mass_raw_top,0.4]=0.0650139576682
  [far_mass_raw_top,0.6]=0.010167022096
  [far_mass_raw_top,0.8]=0.000320866597963
  [far_mass_raw_top,1.0]=-0.000000001
  [far_mass_raw_bottom,0.2]=0.000320866597963
  [far_mass_raw_bottom,0.4]=0.010167022096
  [far_mass_raw_bottom,0.6]=0.0650139576682
  [far_mass_raw_bottom,0.8]=0.283078474856
  [far_mass_raw_bottom,1.0]=30.1524568576
)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

trial_chunk_idx=$((idx % ${#TRIAL_CHUNKS[@]}))
idx=$((idx / ${#TRIAL_CHUNKS[@]}))
task_id=$((idx % 10))
idx=$((idx / 10))
target_idx=$((idx % ${#TARGETS[@]}))
idx=$((idx / ${#TARGETS[@]}))
condition_idx=$((idx % ${#CONDITIONS[@]}))

condition="${CONDITIONS[$condition_idx]}"
direction="${DIRECTIONS[$condition_idx]}"
target="${TARGETS[$target_idx]}"
threshold="${THRESHOLDS[$condition,$target]}"
trial_ids="${TRIAL_CHUNKS[$trial_chunk_idx]}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: condition=${condition} target=${target} direction=${direction} threshold=${threshold} task=${task_id} trials=${trial_ids}"

exec "${PROJECT_ROOT}/eval_libero10_raw_rerun_full7_40gb_chunk1.sh" \
  "${condition}" "${target}" "${direction}" "${threshold}" "${task_id}" "${trial_ids}"
