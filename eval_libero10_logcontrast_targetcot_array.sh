#!/bin/bash
#SBATCH --account=rrg-florian7_gpu
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --job-name=l10_lc_cot
#SBATCH --output=./slurm_logs/logcontrast_libero10_targetcot_%A_%a.out
#SBATCH --error=./slurm_logs/logcontrast_libero10_targetcot_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"

TARGETS=(0.2 0.4 0.6 0.8 1.0)
H_HIS=(1.05 0.4 -0.2 -0.75 -10.0)
H_LOS=(0.35 0.13333333333333333 -0.3 -1.0 -10.1)
TRIAL_CHUNKS=(0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39 40-44 45-49)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

trial_chunk_idx=$((idx % ${#TRIAL_CHUNKS[@]}))
idx=$((idx / ${#TRIAL_CHUNKS[@]}))
task_id=$((idx % 10))
idx=$((idx / 10))
target_idx=$((idx % ${#TARGETS[@]}))

target="${TARGETS[$target_idx]}"
h_hi="${H_HIS[$target_idx]}"
h_lo="${H_LOS[$target_idx]}"
trial_ids="${TRIAL_CHUNKS[$trial_chunk_idx]}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: target=${target} h_hi=${h_hi} h_lo=${h_lo} task=${task_id} trials=${trial_ids}"

exec "${PROJECT_ROOT}/eval_libero10_logcontrast_targetcot_40gb_50ep.sh" \
  "${target}" "${h_hi}" "${h_lo}" "${task_id}" "${trial_ids}"
