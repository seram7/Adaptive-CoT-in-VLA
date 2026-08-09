#!/bin/bash
#SBATCH --account=rrg-florian7_gpu
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --job-name=l10_cot_rem
#SBATCH --output=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs/libero10_cotratio_remaining_%A_%a.out
#SBATCH --error=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs/libero10_cotratio_remaining_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"

MODES=(far_mass random)
TARGETS=(0.2 0.4 0.6 0.8 1.0)
TRIAL_CHUNKS=(5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49)

declare -A FAR_THRESHOLDS=(
  [0.2]=0.83
  [0.4]=0.25
  [0.6]=0.064
  [0.8]=0.015
  [1.0]=-0.000001
)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

trial_chunk_idx=$((idx % ${#TRIAL_CHUNKS[@]}))
idx=$((idx / ${#TRIAL_CHUNKS[@]}))
task_id=$((idx % 10))
idx=$((idx / 10))
target_idx=$((idx % ${#TARGETS[@]}))
idx=$((idx / ${#TARGETS[@]}))
mode_idx=$((idx % ${#MODES[@]}))

mode="${MODES[$mode_idx]}"
target="${TARGETS[$target_idx]}"
threshold="${FAR_THRESHOLDS[$target]}"
trial_ids="${TRIAL_CHUNKS[$trial_chunk_idx]}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: mode=${mode} target=${target} threshold=${threshold} task=${task_id} trials=${trial_ids}"

exec "${PROJECT_ROOT}/eval_libero10_cotratio_dual2gpu_50ep.sh" \
  "${mode}" "${target}" "${threshold}" "${task_id}" "${trial_ids}"
