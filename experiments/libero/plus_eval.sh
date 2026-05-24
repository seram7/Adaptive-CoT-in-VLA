#!/bin/bash
#SBATCH --account=def-bussmann
#SBATCH --time=9:30:00 ######
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --job-name=plus_l5_10_ent ######
#SBATCH --array=0-70 ######
#SBATCH --output=/scratch/%u/Adaptive-CoT-in-VLA/results/result_%A_%a.out
#SBATCH --error=/scratch/%u/Adaptive-CoT-in-VLA/results/error_%A_%a.err
#SBATCH --nodes=1

USER_NAME="${USER:-$(whoami)}"
SCRATCH_ROOT="/scratch/${USER_NAME}"

ADAPTIVE_ROOT="${SCRATCH_ROOT}/Adaptive-CoT-in-VLA"
LIBERO_PLUS_ROOT="${SCRATCH_ROOT}/LIBERO-plus"
VENV_PATH="${SCRATCH_ROOT}/envs/ecot"
HF_ASSETS_ROOT="${SCRATCH_ROOT}/hf_assets"

CHUNK_SIZE=10
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

SCHEDULE="${LIBERO_PLUS_ROOT}/schedules/libero_plus_l5_libero_10.json" ######

echo "========================================"
echo "Host        : $(hostname)"
echo "User        : ${USER_NAME}"
echo "Array ID    : ${SLURM_ARRAY_TASK_ID}"
echo "Chunk size  : ${CHUNK_SIZE}"
echo "Start index : ${START_INDEX}"
echo "Schedule    : ${SCHEDULE}"
echo "Adaptive    : ${ADAPTIVE_ROOT}"
echo "LIBERO-plus : ${LIBERO_PLUS_ROOT}"
echo "HF assets   : ${HF_ASSETS_ROOT}"
echo "========================================"

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 mujoco/3.1.6 opencv/4.10.0 arrow/15.0.1

source "${VENV_PATH}/bin/activate"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MUJOCO_ROOT_DIR}/bin"
export CUDA_HOME="${CUDA_DIR}"

# LIBERO-plus를 기존 LIBERO보다 먼저 import하게 함
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${PYTHONPATH:-}"

# LIBERO-plus 전용 config 사용
export HOME="${LIBERO_PLUS_ROOT}/runtime_home"

export HF_HOME="${HF_ASSETS_ROOT}"
export HF_ASSETS_ROOT="${HF_ASSETS_ROOT}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NUMBA_DISABLE_JIT=1

cd "${ADAPTIVE_ROOT}/experiments/libero"

python plus_eval.py \
    --schedule-json $SCHEDULE \
    --eval-start-index $START_INDEX \
    --num-eval-episodes $CHUNK_SIZE \
    --prompt-control-mode metric_window_total_variation \
    --uncertainty-metric-name entropy \
    --score-threshold 0.27 \
    --tv-window 5 \
    --seed 42
