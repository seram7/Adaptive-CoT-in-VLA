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

TASK_ID="${1:?task id required}"
TRIAL_IDS="${2:?trial ids required, e.g. 0-4}"

if [[ "${TRIAL_IDS}" =~ ^[0-9]+-[0-9]+$ ]]; then
  START_TRIAL="${TRIAL_IDS%-*}"
  END_TRIAL="${TRIAL_IDS#*-}"
  TRIAL_IDS="$(seq -s, "${START_TRIAL}" "${END_TRIAL}")"
fi

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"
DEEPTHINK_ROOT="/home/seram7/codes/DeepThinkVLA"
VENV_PATH="/scratch/${USER}/.env/vla"
LIBERO_ROOT="/scratch/${USER}/.env/vla/src/libero"
OUTPUT_ROOT="${PROJECT_ROOT}/rollouts_logcontrast"
TMP_ROOT="${PROJECT_ROOT}/tmp/openvla_deepthink_c3_refire_h1_chunk1/${SLURM_JOB_ID:-manual}"

H_HI="1.0"
H_LO="0.3333333333333333"
REFIRE_MAX_FIRES="3"
REFIRE_INTERVAL="5"
REFIRE_MIN_SCORE="0.6666666666666666"
RUN_NAME="openvla_deepthink_far_mass_x_peak_separation_c3_refire_h1p0_lo0p3333333333333333_a3_b10_max3_int5_mid0p6666666666666666_chunk1_15ep"

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 mujoco/3.1.6 opencv/4.10.0 arrow/15.0.1

source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r SHARED_VISIBLE_DEVICE _ <<< "${ALLOCATED_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES="${SHARED_VISIBLE_DEVICE}"
DEEPTHINK_VISIBLE_DEVICE="${SHARED_VISIBLE_DEVICE}"

export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONWARNINGS="ignore"
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM
unset EGL_DEVICE_ID
export LD_LIBRARY_PATH="${EBROOTMUJOCO}/lib:${LD_LIBRARY_PATH:-}:/usr/lib/nvidia"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LIBERO_ROOT="${LIBERO_ROOT}"
export LIBERO_PATH="${LIBERO_ROOT}"
export PYTHONPATH="${LIBERO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_LOG_LEVEL=30
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONFAULTHANDLER=1
export HF_HOME="/scratch/${USER}/hf"
export HUGGINGFACE_HUB_CACHE="/scratch/${USER}/hf/hub"
export TORCH_HOME="/scratch/${USER}/torch_cache"
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export DEEPTHINKVLA_REPO_ROOT="${DEEPTHINK_ROOT}"
export DEEPTHINKVLA_PYTHON="${VENV_PATH}/bin/python"

mkdir -p slurm_logs "${OUTPUT_ROOT}" "${TMP_ROOT}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "Host: $(hostname)"
echo "Dataset: libero_10"
echo "Mode: C3 refire"
echo "h_hi: ${H_HI}"
echo "h_lo: ${H_LO}"
echo "refire_max_fires: ${REFIRE_MAX_FIRES}"
echo "refire_interval: ${REFIRE_INTERVAL}"
echo "refire_min_score: ${REFIRE_MIN_SCORE}"
echo "Task ID: ${TASK_ID}"
echo "Trials: ${TRIAL_IDS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name: ${RUN_NAME}"
which python
python --version
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,mig.mode.current --format=csv || true

python -X faulthandler experiments/libero/eval_openvla_deepthink_dual.py \
  --dataset libero_10 \
  --num-trials-per-task 15 \
  --task-ids "${TASK_ID}" \
  --trial-ids "${TRIAL_IDS}" \
  --seed 42 \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --tmp-dir "${TMP_ROOT}" \
  --skip-existing-rollouts \
  --save-rollout-pt \
  --openvla-device cuda:0 \
  --deepthink-python "${DEEPTHINKVLA_PYTHON}" \
  --deepthink-repo-root "${DEEPTHINK_ROOT}" \
  --deepthink-device cuda:0 \
  --deepthink-cuda-visible-devices "${DEEPTHINK_VISIBLE_DEVICE}" \
  --deepthink-execute-chunk-steps 1 \
  --uncertainty-metric-name far_mass_x_peak_separation \
  --router-control-mode farmass_log_contrast \
  --short-window 3 \
  --long-window 10 \
  --h-hi "${H_HI}" \
  --h-lo "${H_LO}" \
  --trigger-mode refire \
  --refire-max-fires "${REFIRE_MAX_FIRES}" \
  --refire-interval "${REFIRE_INTERVAL}" \
  --refire-min-score "${REFIRE_MIN_SCORE}"

echo "Job finished at $(date)"
