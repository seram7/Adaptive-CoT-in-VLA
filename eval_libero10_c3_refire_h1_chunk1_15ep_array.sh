#!/bin/bash
#SBATCH --account=rrg-florian7_gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --job-name=l10_c3_refire
#SBATCH --output=./slurm_logs/libero10_c3_refire_%A_%a.out
#SBATCH --error=./slurm_logs/libero10_c3_refire_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"
TRIAL_CHUNKS=(0-4 5-9 10-14)

idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
trial_chunk_idx=$((idx % ${#TRIAL_CHUNKS[@]}))
task_id=$((idx / ${#TRIAL_CHUNKS[@]}))
trial_ids="${TRIAL_CHUNKS[$trial_chunk_idx]}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: C3 refire h=1.0 task=${task_id} trials=${trial_ids}"

exec "${PROJECT_ROOT}/eval_libero10_c3_refire_h1_chunk1_15ep.sh" \
  "${task_id}" "${trial_ids}"
