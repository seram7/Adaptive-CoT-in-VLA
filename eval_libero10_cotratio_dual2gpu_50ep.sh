#!/bin/bash
#SBATCH --account=rrg-florian7_gpu
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --job-name=l10_cot_dual
#SBATCH --output=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs/libero10_cotratio_dual2gpu_result_%A.out
#SBATCH --error=/scratch/seram7/Adaptive-CoT-in-VLA/slurm_logs/libero10_cotratio_dual2gpu_error_%A.err

set -euo pipefail

ROUTER_KIND="${1:?router kind required: far_mass or random}"
TARGET_COT_RATIO="${2:?target CoT ratio required}"
FAR_MASS_THRESHOLD="${3:?far_mass threshold required, use 0 for random}"
TASK_ID="${4:?task id required}"
TRIAL_IDS="${5:?trial ids required, e.g. 0-4}"

if [[ "${TRIAL_IDS}" =~ ^[0-9]+-[0-9]+$ ]]; then
  START_TRIAL="${TRIAL_IDS%-*}"
  END_TRIAL="${TRIAL_IDS#*-}"
  TRIAL_IDS="$(seq -s, "${START_TRIAL}" "${END_TRIAL}")"
fi

PROJECT_ROOT="/home/seram7/codes/Adaptive-CoT-in-VLA"
DEEPTHINK_ROOT="/home/seram7/codes/DeepThinkVLA"
VENV_PATH="/scratch/${USER}/.env/vla"
LIBERO_ROOT="/scratch/${USER}/.env/vla/src/libero"
OUTPUT_ROOT="/scratch/${USER}/Adaptive-CoT-in-VLA/rollouts_42_h100_new/libero10_cotratio_50ep_dual2gpu_chunk1"
TMP_ROOT="/scratch/${USER}/Adaptive-CoT-in-VLA/tmp/libero10_cotratio_dual2gpu_chunk1/${ROUTER_KIND}/${TARGET_COT_RATIO}/${SLURM_JOB_ID:-manual}"
SLURM_LOG_ROOT="/scratch/${USER}/Adaptive-CoT-in-VLA/slurm_logs"
PYCACHE_ROOT="/scratch/${USER}/pycache/adaptive_cot"

module purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 mujoco/3.1.6 opencv/4.10.0 arrow/15.0.1

source "${VENV_PATH}/bin/activate"

cd "${PROJECT_ROOT}"

ALLOCATED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r OPENVLA_VISIBLE_DEVICE DEEPTHINK_VISIBLE_DEVICE _ <<< "${ALLOCATED_CUDA_VISIBLE_DEVICES}"
if [[ -z "${DEEPTHINK_VISIBLE_DEVICE:-}" ]]; then
  echo "Expected two visible GPUs, got CUDA_VISIBLE_DEVICES=${ALLOCATED_CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${OPENVLA_VISIBLE_DEVICE}"

TARGET_LABEL="${TARGET_COT_RATIO//./p}"
THRESHOLD_LABEL="${FAR_MASS_THRESHOLD//./p}"
THRESHOLD_LABEL="${THRESHOLD_LABEL//-/m}"

COMMON_CONTROL_ARGS=(
  --uncertainty-metric-name far_mass_x_peak_separation
  --tv-window 5
  --deepthink-execute-chunk-steps 1
)

if [[ "${ROUTER_KIND}" == "far_mass" ]]; then
  RUN_NAME="openvla_deepthink_far_mass_targetcot${TARGET_LABEL}_th${THRESHOLD_LABEL}_wtv_w5_chunk1_dual2gpu_50ep"
  CONTROL_ARGS=(
    --router-control-mode metric_window_total_variation
    --score-threshold "${FAR_MASS_THRESHOLD}"
    --score-threshold-direction gt
  )
elif [[ "${ROUTER_KIND}" == "random" ]]; then
  RUN_NAME="openvla_deepthink_random_targetcot${TARGET_LABEL}_p${TARGET_LABEL}_chunk1_dual2gpu_50ep"
  CONTROL_ARGS=(
    --router-control-mode random
    --score-threshold 0
    --score-threshold-direction gt
    --random-deepthink-probability "${TARGET_COT_RATIO}"
  )
else
  echo "Unsupported ROUTER_KIND=${ROUTER_KIND}; expected far_mass or random" >&2
  exit 2
fi

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
export PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
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

mkdir -p "${SLURM_LOG_ROOT}" "${OUTPUT_ROOT}" "${TMP_ROOT}" "${PYCACHE_ROOT}" \
  "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "Host: $(hostname)"
echo "Dataset: libero_10"
echo "Task ID: ${TASK_ID}"
echo "Trials: ${TRIAL_IDS}"
echo "Router kind: ${ROUTER_KIND}"
echo "Target CoT ratio: ${TARGET_COT_RATIO}"
echo "Far-mass threshold: ${FAR_MASS_THRESHOLD}"
echo "Control mode args: ${CONTROL_ARGS[*]}"
echo "Metric: far_mass_x_peak_separation"
echo "Metric window size: 5"
echo "Action chunk size: 1"
echo "Allocated CUDA_VISIBLE_DEVICES: ${ALLOCATED_CUDA_VISIBLE_DEVICES}"
echo "OpenVLA CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DeepThink CUDA_VISIBLE_DEVICES: ${DEEPTHINK_VISIBLE_DEVICE}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name: ${RUN_NAME}"
which python
python --version
python - <<'PY'
import transformers
print("transformers", transformers.__version__)
PY

python -X faulthandler experiments/libero/eval_openvla_deepthink_dual.py \
  --dataset libero_10 \
  --num-trials-per-task 50 \
  --task-ids "${TASK_ID}" \
  --trial-ids "${TRIAL_IDS}" \
  --seed 42 \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --tmp-dir "${TMP_ROOT}" \
  --skip-existing-rollouts \
  --openvla-device cuda:0 \
  --deepthink-python "${DEEPTHINKVLA_PYTHON}" \
  --deepthink-repo-root "${DEEPTHINK_ROOT}" \
  --deepthink-device cuda:0 \
  --deepthink-cuda-visible-devices "${DEEPTHINK_VISIBLE_DEVICE}" \
  "${COMMON_CONTROL_ARGS[@]}" \
  "${CONTROL_ARGS[@]}"

echo "Job finished at $(date)"
