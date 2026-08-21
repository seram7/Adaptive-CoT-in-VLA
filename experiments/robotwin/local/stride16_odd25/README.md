# PI0.5 + ZR-0: RoboTwin odd-task campaign

This directory is the portable launcher for the 25 odd, zero-based RoboTwin
task IDs. It reproduces the five-arm experiment with a 16-action replanning
stride, lightweight sampling Flow-Far-Mass routing, and common-request
inference sharing (`robotwin-policy-request-v1`).

## Protocol

Every arm uses `action_steps=32` and executes at most
`replan_stride=16` actions before observing again. This keeps the decision
frequency matched across PI0.5-only, ZR-0-only, fixed, random, and adaptive.

| Arm | Routing rule |
|---|---|
| `baseline_pi05_only` | Always execute PI0.5 |
| `baseline_zr0_only_direct` | Always execute ZR-0 direct actions |
| `fixed` | ZR-0 at query indices 0, 5, 10, ... |
| `random` | Seeded Bernoulli 0.2, with the first query forced to ZR-0 |
| `adaptive_sampling_farmass_replan_max` | ZR-0 when max Flow-Far-Mass over the next 16 planned actions crosses the calibrated threshold |

Adaptive uses five PI0.5 action samples, `force_first_query=true`, and
`cooldown_steps=32`. Fixed and random deliberately do not use cooldown because
their schedules are the control conditions. PI0.5 is queried first at every
decision point to obtain the candidate action; adaptive additionally computes
Far-Mass. A routed ZR-0 decision is therefore one additional model call. Always
report both `policy_queries` and `zr0_queries`.

### Deterministic policy requests and shared prefixes

Every policy random draw uses a SHA-256-derived seed over
`(protocol, task, split, episode_seed, query_index, stream)`. Streams are
domain-separated as `pi05_execution`, `pi05_farmass`,
`zr0_direct_action_execution`, and `random_router`. Routing-arm names are
intentionally absent, so fixed/random/adaptive receive the same model output
for the same observation and query.

The main campaign runs adaptive first and persists inference results in
`INFERENCE_CACHE` (SQLite). Later arms reuse an entry only when its exact
observation hash, request identity, and seed match. Thus fixed, random, and
adaptive perform no duplicate PI0.5 or ZR-0 inference along their common
trajectory prefix. At the first different routing decision the selected
actions differ, future observation hashes change, and each branch computes and
caches its own requests. PI0.5 execution always uses a separately seeded N=1
batch; the N=5 samples are used only for Far-Mass and can never become the
executed action.

ZR-0 resets its observation buffer before the first uncached inference in each
episode and creates initial flow noise with a request-local `torch.Generator`.
Each episode JSON records observation, PI0.5 action, ZR-0 action, selected
action, and executed-action hashes. After each task/condition, the runner calls
`audit_shared_prefix.py`; a hash mismatch before the first pairwise routing
divergence makes the campaign fail and retry.

Calibration and main evaluation use disjoint expert-feasible seed blocks:

- Calibration: block 4, 20 clean + 30 randomized episodes per task.
- Main: block 3, 50 clean + 200 randomized episodes per task and arm.
- Main total: 25 tasks x 250 episodes x 5 arms = 31,250 episodes.

The odd task set is fixed by `pi05_task_registry.py`:

| task_id | Task | Step limit |
|---:|---|---:|
| 1 | `beat_block_hammer` | 400 |
| 3 | `blocks_ranking_size` | 1200 |
| 5 | `click_bell` | 400 |
| 7 | `grab_roller` | 400 |
| 9 | `handover_mic` | 600 |
| 11 | `lift_pot` | 400 |
| 13 | `move_pillbottle_pad` | 400 |
| 15 | `move_stapler_pad` | 400 |
| 17 | `open_microwave` | 1500 |
| 19 | `pick_dual_bottles` | 400 |
| 21 | `place_a2b_right` | 400 |
| 23 | `place_bread_skillet` | 500 |
| 25 | `place_can_basket` | 700 |
| 27 | `place_container_plate` | 400 |
| 29 | `place_empty_cup` | 500 |
| 31 | `place_mouse_pad` | 400 |
| 33 | `place_object_scale` | 400 |
| 35 | `place_phone_stand` | 400 |
| 37 | `press_stapler` | 400 |
| 39 | `put_object_cabinet` | 700 |
| 41 | `scan_object` | 500 |
| 43 | `shake_bottle_horizontally` | 700 |
| 45 | `stack_blocks_two` | 800 |
| 47 | `stack_bowls_two` | 900 |
| 49 | `turn_switch` | 400 |

## Tested system

The reference run used Ubuntu Linux, NVIDIA driver 580.76.05, an RTX 3090
24 GB for RoboTwin + ZR-0, and an RTX 4090 24 GB for PI0.5. Two 24 GB GPUs are
recommended. The tested Python/package split was:

| Environment | Python | Important versions |
|---|---:|---|
| RoboTwin evaluator | 3.10.20 | torch 2.6.0+cu126, numpy 1.26.4, sapien 3.0.0b1, mplib 0.2.1, websockets 16.1.1 |
| PI0.5 server | 3.11.15 | torch 2.6.0, numpy 1.26.4, JAX 0.5.0, flax 0.10.2 |
| ZR-0 server | 3.10.20 | torch 2.6.0+cu126, transformers 4.57.1, flash-attn 2.7.3 |

Keep the three environments separate. RoboTwin/SAPIEN, OpenPI, and ZR-0 have
incompatible dependency pins.

## 1. Clone exact revisions

```bash
git clone --branch exp/robotwin \
  https://github.com/seram7/Adaptive-CoT-in-VLA.git

git clone https://github.com/RoboTwin-Platform/RoboTwin.git
git -C RoboTwin checkout 13c3c47ff4312dd62484bcd51be034af55c062d1

git clone --recurse-submodules https://github.com/RUCKBReasoning/ZR-0.git
git -C ZR-0 checkout b1440d4cf27624da2b1aa31268637cf46601c15d
git -C ZR-0 submodule update --init --recursive
```

Apply the checked-in inference, Flow-Far-Mass, WebSocket, and RoboTwin task
fixes. The script refuses unexpected commits or dirty worktrees and is safe to
run again after a successful application.

```bash
bash Adaptive-CoT-in-VLA/experiments/robotwin/apply_dependency_patches.sh \
  "$PWD/RoboTwin" "$PWD/ZR-0"
```

The patched dependency worktrees are expected to become dirty. Do not run a
reset or checkout after applying the patches.

## 2. Install RoboTwin and assets

Follow the official RoboTwin 2.0 installation instructions. On a Conda or
Micromamba Python 3.10 environment, the upstream helper is:

```bash
conda create -y -n robotwin-oft python=3.10
conda activate robotwin-oft
cd RoboTwin
bash script/_install.sh
pip install websockets msgpack-numpy pyyaml h5py pytest
bash script/_download_assets.sh
```

If assets are stored outside the Git checkout, create all three links expected
by RoboTwin:

```bash
ln -s /ABS/ASSETS/embodiments RoboTwin/assets/embodiments
ln -s /ABS/ASSETS/objects RoboTwin/assets/objects
ln -s /ABS/ASSETS/background_texture RoboTwin/assets/background_texture
```

Do not run SAPIEN through an SSH X-forwarded display. The launcher selects the
Vulkan GPU with `MESA_VK_DEVICE_SELECT`.

## 3. Install the PI0.5 server environment

RoboTwin vendors OpenPI under `policy/pi05` with a lock file. Use its Python
3.11 environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd RoboTwin/policy/pi05
uv sync --frozen
```

The corresponding interpreter is normally
`RoboTwin/policy/pi05/.venv/bin/python`.

## 4. Install ZR-0

Use the upstream ZR-0 instructions in a separate Python 3.10 environment:

```bash
conda create -y -n zr0 python=3.10
conda activate zr0
cd ZR-0/lerobot
pip install -e .
cd ..
pip install -r requirements.txt
conda install -y -c conda-forge ffmpeg
pip install flash-attn==2.7.3 --no-build-isolation
```

## 5. Download checkpoints

Download the Motus 50-task clean+randomized PI0.5 checkpoint:

```bash
hf download motus-robotics/pi0.5_robotwin2 \
  --local-dir /ABS/CHECKPOINTS/motus-pi05-robotwin2
```

The directory must contain `model.safetensors` and
`assets/pi0.5_clean_randomize_joint_training/norm_stats.json`.

Download the RoboTwin ZR-0 checkpoint from ModelScope:

```bash
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "seeklhy/ZR-0-robotwin",
    local_dir="/ABS/CHECKPOINTS/ZR-0-robotwin",
)
PY
```

## 6. Build ZR-0 normalization metadata

ZR-0 expects LeRobot v2.1 normalization metadata that is not bundled with the
checkpoint. Download the official 50 clean archives and build it once:

```bash
hf download TianxingChen/RoboTwin2.0 --repo-type dataset \
  --include 'dataset/*/aloha-agilex_clean_50.zip' \
  --local-dir /ABS/DATA/RoboTwin2.0-clean50

/ABS/ENV/robotwin-oft/bin/python \
  Adaptive-CoT-in-VLA/experiments/robotwin/build_zr0_metadata.py \
  --archives-root /ABS/DATA/RoboTwin2.0-clean50 \
  --output /ABS/METADATA/robotwin2.0-aloha-agilex
```

The output must contain `meta/info.json`, `meta/stats.json`,
`meta/tasks.jsonl`, and `meta/episodes.jsonl`.

## 7. Configure the machine

```bash
cd Adaptive-CoT-in-VLA/experiments/robotwin/local/stride16_odd25
cp env.example env.local
```

Edit every `/ABS/...` path in `env.local`. Find stable CUDA UUIDs with:

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv
lspci -nn | grep -i nvidia
```

Use the GPU that renders RoboTwin as `SIM_CUDA_DEVICE`; ZR-0 is colocated on
that GPU. Use the other GPU as `PI05_CUDA_DEVICE`. `SIM_MESA_DEVICE` is the
simulator GPU's lower-case PCI vendor/device ID followed by `!`, for example
`10de:2204!` for an RTX 3090.

Keep `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` when that file
exists. If your distribution installs the NVIDIA ICD elsewhere, update the
path before running preflight.

Create the cache, temporary, and result directories named in `env.local`.
Ensure ports 18000 and 18100 are free:

```bash
ss -ltnp | grep -E ':(18000|18100) ' || true
```

## 8. Validate and launch

Run the local unit tests before consuming GPU time:

```bash
cd /ABS/PATH/Adaptive-CoT-in-VLA
/ABS/ENV/robotwin-oft/bin/python -m pytest -q experiments/robotwin/test_*.py

cd /ABS/PATH/RoboTwin/policy/pi05
/ABS/ENV/robotwin-openpi/bin/python -m pytest -q \
  src/openpi/uncertainty_test.py \
  src/openpi/policies/aloha_policy_batch_test.py

cd /ABS/PATH/Adaptive-CoT-in-VLA
ROBOTWIN_EXPERIMENT_ENV=$PWD/experiments/robotwin/local/stride16_odd25/env.local \
  bash experiments/robotwin/local/stride16_odd25/preflight.sh
```

Launch the retrying supervisor in a session detached from SSH:

```bash
bash /ABS/PATH/Adaptive-CoT-in-VLA/experiments/robotwin/local/stride16_odd25/launch_detached.sh \
  /ABS/PATH/Adaptive-CoT-in-VLA/experiments/robotwin/local/stride16_odd25/env.local
```

The launcher first generates expert-feasible manifests, then collects all
pilots, estimates 50 task-condition thresholds, and finally runs the five arms
task by task. Adaptive runs first to populate the shared inference cache;
PI0.5-only, ZR-0-only, fixed, and random follow. Completed episode JSON files
are skipped on restart, and each episode is written atomically. The supervisor
retries failures after 30 seconds. A second supervisor for the same result
directory is rejected with a file lock.

## Monitoring and outputs

```bash
tail -f /ABS/RESULTS/runner_logs/campaign25_status.log
tail -f /ABS/RESULTS/runner_logs/campaign25_supervisor.log

find /ABS/RESULTS/pilot -name 'episode_*.json' | wc -l       # 1,250
find /ABS/RESULTS/main_25 -name 'episode_*.json' | wc -l     # 31,250
find /ABS/RESULTS/runner_logs -name 'prefix_audit_*.json' | wc -l  # 50

nvidia-smi
```

Important outputs are:

```text
manifests/calibration/                 expert-feasible seed block 4
manifests/main/                        expert-feasible seed block 3
pilot/<condition>/<task>/pilot/        PI0.5 metric-only calibration episodes
thresholds/tasks25_step.json           per-task, per-condition thresholds
main_25/<task>/<condition>/<arm>/       main episode JSON and summary
runner_logs/                            services, status, retry, and GPU traces
```

The random arm's forced first query increases its realized CoT ratio above the
nominal 0.2, especially in short episodes. Compare methods using realized
`zr0_queries / policy_queries`, total calls (`policy_queries + zr0_queries`),
and success rate rather than the configured ratio alone.
