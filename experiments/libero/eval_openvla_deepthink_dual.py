#!/usr/bin/env python
"""OpenVLA/DeepThinkVLA router evaluation for standard LIBERO suites.

OpenVLA stays in-process so we can reuse its action-token uncertainty metrics.
DeepThinkVLA runs as a JSONL worker process, which lets it use the separate
DeepThinkVLA environment/venv without entangling dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import torch
from tqdm import tqdm

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
if os.environ.get("LIBERO_ROOT"):
    sys.path.insert(0, str(Path(os.environ["LIBERO_ROOT"]).expanduser()))

from experiments.libero.eval_dualgpu import (  # noqa: E402
    DEFAULT_OPENVLA_CKPTS,
    OpenVLAWorker,
    compute_control_score,
    compute_step_uncertainty_from_action_scores,
    parse_int_filter,
    threshold_direction_to_uncertain,
    validate_local_checkpoint,
)
from experiments.libero.router_triggers import FarmassLogContrastTrigger  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    resize_image,
    save_rollout_video,
)
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

try:  # noqa: E402
    from libero.libero import benchmark
except ImportError as exc:  # noqa: E402
    raise ImportError(
        "Could not import LIBERO. Install it or set LIBERO_ROOT before running this eval."
    ) from exc


DEFAULT_DEEPTHINK_CHECKPOINT = os.environ.get(
    "DEEPTHINKVLA_CHECKPOINT",
    "yinchenghust/deepthinkvla_libero_cot_sft",
)
DEFAULT_DEEPTHINK_REPO_ROOT = os.environ.get("DEEPTHINKVLA_REPO_ROOT")
DEFAULT_DEEPTHINK_PYTHON = os.environ.get("DEEPTHINKVLA_PYTHON", sys.executable)

DEFAULT_SUITE_THRESHOLDS = {
    "libero_spatial": 1.05,
    "libero_object": 1.21,
    "libero_goal": 0.79,
    "libero_10": 0.93,
}

MAX_STEPS = {
    "libero_spatial": 200,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:90]


def extract_cached_reasoning_prefix(decoded_text: str) -> str:
    """Extract a reusable reasoning prefix for logging/cache introspection."""

    text = str(decoded_text or "")
    if "</think>" in text:
        return text.split("</think>", 1)[0] + "</think>"
    if "<action>" in text:
        return text.split("<action>", 1)[0] + "<action>"
    if " MOVE REASONING: " in text:
        return text.split(" MOVE REASONING: ", 1)[0] + " MOVE REASONING: "
    if " GRIPPER POSITION: " in text:
        return text.split(" GRIPPER POSITION: ", 1)[0] + " GRIPPER POSITION: "
    return ""


def build_save_dir(
    output_root: Path,
    dataset: str,
    run_name: str,
    task_slug: str,
    trial_id: int,
) -> Path:
    episode_root = output_root / dataset / run_name / task_slug
    return episode_root / f"trial{trial_id:02d}"


def run_name_for(args) -> str:
    threshold_suffix = (
        f"{args.score_threshold}"
        if args.score_threshold_direction == "auto"
        else f"{args.score_threshold}_{args.score_threshold_direction}"
    )
    if args.router_control_mode == "fixed_interval":
        suffix = f"fixed{args.fixed_deepthink_interval}"
    elif args.router_control_mode == "random":
        suffix = f"random_p{args.random_deepthink_probability}"
    elif args.router_control_mode == "farmass_log_contrast":
        suffix = (
            "far_mass_x_peak_separation_farmass_log_contrast"
            f"_h{args.h_hi}_lo{args.h_lo}"
            f"_a{args.short_window}_b{args.long_window}_{args.trigger_mode}"
        )
        if args.trigger_mode == "refire":
            suffix += f"_max{args.refire_max_fires}_int{args.refire_interval}"
            if args.refire_min_score is not None:
                suffix += f"_mid{args.refire_min_score}"
    elif "window" in args.router_control_mode:
        suffix = f"{args.uncertainty_metric_name}_{args.router_control_mode}_{threshold_suffix}_w{args.tv_window}"
    else:
        suffix = f"{args.uncertainty_metric_name}_{args.router_control_mode}_{threshold_suffix}"
    if args.deepthink_masked_cot:
        suffix += "_maskedcot"
    return f"openvla_deepthink_{suffix}"


def summary_contains_episode(summary_path: Path, task_id: int, trial_id: int) -> bool:
    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("task_id", -1)) == task_id and int(row.get("trial_id", -1)) == trial_id:
                return True
    return False


def append_summary(run_dir: Path, row: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "episode_summary.jsonl"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return summary_path


def save_reasoning_jsonl(episode_dir: Path, records: list[dict[str, Any]]) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / "deepthink_reasoning.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def save_deepthink_rollout_pt(
    episode_dir: Path,
    task_name: str,
    task_id: int,
    trial_id: int,
    success: int,
    step_ids: list[int],
    selected_metric_series: list[float],
    farmass_log_m_series: list[float],
    farmass_log_z_series: list[float],
    farmass_log_s_series: list[float],
    farmass_log_conflicted_series: list[float],
    farmass_log_fired_series: list[float],
    control_score_series: list[float],
    uncertain_decision_series: list[float],
    used_deepthink_series: list[float],
    openvla_inference_times: list[float],
    deepthink_inference_times: list[float],
    selected_inference_times: list[float],
    ee_positions: list[np.ndarray],
    proprio_states: list[np.ndarray],
    far_mass_per_slot_series: list[np.ndarray],
    peak_separation_per_slot_series: list[np.ndarray],
    far_mass_x_peak_separation_per_slot_series: list[np.ndarray],
    openvla_actions: list[np.ndarray],
    deepthink_actions: list[np.ndarray],
    executed_actions: list[np.ndarray],
    selected_policy_series: list[str],
    reasoning_records: list[dict[str, Any]],
    router_config: dict[str, Any],
) -> Path:
    episode_dir.mkdir(parents=True, exist_ok=True)

    def _array(values, dtype=np.float32):
        if values is None or len(values) == 0:
            return np.empty((0,), dtype=dtype)
        return np.asarray(values, dtype=dtype)

    payload = {
        "task_name": str(task_name),
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "success": int(success),
        "step_ids": _array(step_ids, dtype=np.int32),
        "openvla_metric_series": _array(selected_metric_series),
        "selected_metric_series": _array(selected_metric_series),
        "farmass_log_m_series": _array(farmass_log_m_series),
        "farmass_log_z_series": _array(farmass_log_z_series),
        "farmass_log_s_series": _array(farmass_log_s_series),
        "farmass_log_conflicted_series": _array(farmass_log_conflicted_series),
        "farmass_log_fired_series": _array(farmass_log_fired_series),
        "control_score_series": _array(control_score_series),
        "uncertain_decision_series": _array(uncertain_decision_series),
        "used_deepthink_series": _array(used_deepthink_series),
        "openvla_inference_times": _array(openvla_inference_times),
        "deepthink_inference_times": _array(deepthink_inference_times),
        "selected_inference_times": _array(selected_inference_times),
        "ee_positions": _array(ee_positions),
        "proprio_states": _array(proprio_states),
        "far_mass_per_slot_series": _array(far_mass_per_slot_series),
        "peak_separation_per_slot_series": _array(peak_separation_per_slot_series),
        "far_mass_x_peak_separation_per_slot_series": _array(far_mass_x_peak_separation_per_slot_series),
        "openvla_actions": _array(openvla_actions),
        "deepthink_actions": _array(deepthink_actions),
        "executed_actions": _array(executed_actions),
        "selected_policy_series": list(selected_policy_series),
        "deepthink_reasoning_records": list(reasoning_records),
        "router_config": dict(router_config),
    }
    pt_path = episode_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
    torch.save(payload, pt_path)
    return pt_path


def mean_or_none(values) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def get_libero_wrist_image(obs, resize_size: int):
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]
    return resize_image(img, (resize_size, resize_size))


def deepthink_observation_from_obs(obs, image_size: int) -> dict[str, np.ndarray]:
    return {
        "full_image": np.ascontiguousarray(get_libero_image(obs, image_size)),
        "wrist_image": np.ascontiguousarray(get_libero_wrist_image(obs, image_size)),
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32),
    }


def write_npz(tmp_dir: Path, eval_label: str, step: int, observation: dict[str, np.ndarray]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{eval_label}_step{step:04d}_{time.time_ns()}.npz"
    np.savez_compressed(path, **observation)
    return path


def normalize_deepthink_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[-1] != 7:
        raise ValueError(f"Expected DeepThink action chunk shape (T, 7), got {arr.shape}")
    arr[:, -1] = np.where(arr[:, -1] >= 0, 1.0, -1.0)
    return np.clip(arr, -1.0, 1.0)


def prepare_openvla_action(action: np.ndarray) -> np.ndarray:
    action = normalize_gripper_action(action.copy(), binarize=True)
    return invert_gripper_action(action)


class JsonlWorker:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        log_path: Path,
        cwd: Path,
        env_updates: dict[str, str],
    ) -> None:
        self.name = name
        self.counter = 0
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(env_updates)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            bufsize=1,
        )

    def request(self, npz_path: Path, task: str, reset: bool = False) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"{self.name} server exited with code {self.proc.returncode}; see {self.log_path}"
            )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError(f"{self.name} server pipes are unavailable")

        self.counter += 1
        request_id = f"{self.name}-{self.counter:06d}"
        self.proc.stdin.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "npz_path": str(npz_path),
                    "task": task,
                    "reset": bool(reset),
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    f"{self.name} server closed stdout; returncode={self.proc.poll()} log={self.log_path}"
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("request_id") in {request_id, None}:
                break

        if not response.get("ok", False):
            raise RuntimeError(
                f"{self.name} request failed: {response.get('error_type')}: {response.get('error')}"
            )
        return response

    def close(self) -> None:
        try:
            if self.proc.poll() is None and self.proc.stdin is not None:
                self.proc.stdin.write(
                    json.dumps({"request_id": f"{self.name}-shutdown", "command": "shutdown"}) + "\n"
                )
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
        except Exception:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            self.log_file.close()


def build_deepthink_worker(args, run_dir: Path) -> JsonlWorker:
    if not args.deepthink_repo_root:
        raise ValueError(
            "DeepThinkVLA repo root is required. Pass --deepthink-repo-root "
            "or set DEEPTHINKVLA_REPO_ROOT=/path/to/DeepThinkVLA."
        )
    server = Path(args.deepthink_server)
    if not server.is_absolute():
        server = PROJECT_ROOT / server
    cmd = [
        args.deepthink_python,
        str(server),
        "--checkpoint",
        args.deepthink_checkpoint,
        "--deepthink-repo-root",
        args.deepthink_repo_root,
        "--device",
        args.deepthink_device,
        "--compute-dtype",
        args.deepthink_compute_dtype,
        "--num-images-in-input",
        str(args.deepthink_num_images),
        "--max-new-tokens",
        str(args.deepthink_max_new_tokens),
    ]
    if args.deepthink_masked_cot:
        cmd.append("--masked-cot")
    env_updates = {
        "CUDA_VISIBLE_DEVICES": args.deepthink_cuda_visible_devices,
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
        "HF_HOME": os.environ.get(
            "DEEPTHINK_HF_HOME",
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        ),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if os.environ.get("DEEPTHINK_TRANSFORMERS_CACHE"):
        env_updates["TRANSFORMERS_CACHE"] = os.environ["DEEPTHINK_TRANSFORMERS_CACHE"]
    if os.environ.get("HF_HUB_OFFLINE"):
        env_updates["HF_HUB_OFFLINE"] = os.environ["HF_HUB_OFFLINE"]
    if os.environ.get("TRANSFORMERS_OFFLINE"):
        env_updates["TRANSFORMERS_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"]
    return JsonlWorker(
        "deepthink",
        cmd,
        run_dir / "server_logs" / "deepthink_stderr.log",
        PROJECT_ROOT,
        env_updates,
    )


def eval_openvla_deepthink(args) -> None:
    dataset = args.dataset
    if dataset not in DEFAULT_OPENVLA_CKPTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if args.router_control_mode == "fixed_interval" and args.fixed_deepthink_interval <= 0:
        raise ValueError("--fixed-deepthink-interval must be positive")
    if args.router_control_mode == "random" and not 0.0 <= args.random_deepthink_probability <= 1.0:
        raise ValueError("--random-deepthink-probability must be in [0, 1]")
    if args.router_control_mode == "farmass_log_contrast":
        if args.long_window <= args.short_window:
            raise ValueError("--long-window must be larger than --short-window")
        if args.h_lo >= args.h_hi:
            raise ValueError("--h-lo must be lower than --h-hi")
        if args.trigger_mode == "refire":
            if args.refire_max_fires < 1:
                raise ValueError("--refire-max-fires must be at least 1")
            if args.refire_interval <= 0:
                raise ValueError("--refire-interval must be positive")
    if args.score_threshold is None:
        args.score_threshold = DEFAULT_SUITE_THRESHOLDS[dataset]

    set_seed_everywhere(args.seed)
    openvla_checkpoint = args.openvla_checkpoint or DEFAULT_OPENVLA_CKPTS[dataset]
    validate_local_checkpoint(openvla_checkpoint)

    high_score_means_uncertain = threshold_direction_to_uncertain(
        args.score_threshold_direction,
        args.uncertainty_metric_name,
    )
    threshold_operator = ">" if high_score_means_uncertain else "<"
    run_name = args.run_name or run_name_for(args)
    output_root = Path(args.output_root).expanduser().resolve()
    run_root = output_root / dataset / run_name
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()

    print(f"Seed: {args.seed}")
    print(f"Task suite: {dataset}")
    print(f"Run name: {run_name}")
    print(f"Output root: {output_root}")
    print(f"OpenVLA checkpoint: {openvla_checkpoint}")
    print(f"OpenVLA device: {args.openvla_device}")
    print(f"DeepThink checkpoint: {args.deepthink_checkpoint}")
    print(f"DeepThink visible GPUs: {args.deepthink_cuda_visible_devices}")
    print(f"DeepThink device: {args.deepthink_device}")
    print(
        f"Router: metric={args.uncertainty_metric_name}, mode={args.router_control_mode}, "
        f"threshold={args.score_threshold}, uncertain_when=score {threshold_operator} threshold"
    )
    if args.router_control_mode == "farmass_log_contrast":
        refire_min_score = (
            args.refire_min_score
            if args.refire_min_score is not None
            else 0.5 * (float(args.h_hi) + float(args.h_lo))
        )
        print(
            "Farmass log contrast: "
            f"short={args.short_window}, long={args.long_window}, "
            f"h_hi={args.h_hi}, h_lo={args.h_lo}, trigger_mode={args.trigger_mode}, "
            f"refire_max_fires={args.refire_max_fires}, "
            f"refire_interval={args.refire_interval}, refire_min_score={refire_min_score}"
        )
    openvla_worker = OpenVLAWorker(
        checkpoint=openvla_checkpoint,
        dataset=dataset,
        device=args.openvla_device,
        dtype_name=args.torch_dtype,
        center_crop=True,
    )
    deepthink_worker = build_deepthink_worker(args, run_root)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[dataset]()
    task_id_set = set(parse_int_filter(args.task_ids) or range(task_suite.n_tasks))
    trial_id_set = set(parse_int_filter(args.trial_ids) or range(args.num_trials_per_task))
    task_iter = [task_id for task_id in range(task_suite.n_tasks) if task_id in task_id_set]

    router_config = {
        "openvla_checkpoint": str(openvla_checkpoint),
        "deepthink_checkpoint": str(args.deepthink_checkpoint),
        "openvla_device": str(args.openvla_device),
        "deepthink_device": str(args.deepthink_device),
        "deepthink_cuda_visible_devices": str(args.deepthink_cuda_visible_devices),
        "router_control_mode": str(args.router_control_mode),
        "uncertainty_metric_name": str(args.uncertainty_metric_name),
        "score_threshold": float(args.score_threshold),
        "score_threshold_direction": str(args.score_threshold_direction),
        "high_score_means_uncertain": bool(high_score_means_uncertain),
        "tv_window": int(args.tv_window),
        "fixed_deepthink_interval": int(args.fixed_deepthink_interval),
        "random_deepthink_probability": float(args.random_deepthink_probability),
        "short_window": int(args.short_window),
        "long_window": int(args.long_window),
        "h_hi": float(args.h_hi),
        "h_lo": float(args.h_lo),
        "trigger_mode": str(args.trigger_mode),
        "refire_max_fires": int(args.refire_max_fires),
        "refire_interval": int(args.refire_interval),
        "refire_min_score": (
            None if args.refire_min_score is None else float(args.refire_min_score)
        ),
        "deepthink_max_new_tokens": int(args.deepthink_max_new_tokens),
        "deepthink_masked_cot": bool(args.deepthink_masked_cot),
        "deepthink_execute_chunk_steps": int(args.deepthink_execute_chunk_steps),
    }

    total_episodes = 0
    total_successes = 0
    try:
        for task_id in tqdm(task_iter, desc="Tasks", unit="task"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, "openvla", resolution=args.env_resolution)
            task_slug = slugify(task_description)
            episode_jobs = [
                trial_id
                for trial_id in range(args.num_trials_per_task)
                if trial_id_set is None or trial_id in trial_id_set
            ]

            try:
                for trial_id in tqdm(episode_jobs, desc="Trials", leave=False):
                    save_dir = build_save_dir(
                        output_root=output_root,
                        dataset=dataset,
                        run_name=run_name,
                        task_slug=task_slug,
                        trial_id=trial_id,
                    )
                    summary_path = run_root / "episode_summary.jsonl"
                    if args.skip_existing_rollouts and summary_contains_episode(
                        summary_path,
                        task_id,
                        trial_id,
                    ):
                        print(f"Skipping existing task={task_id} trial={trial_id}")
                        continue

                    print(f"\nTask {task_id} trial {trial_id}: {task_description}")
                    eval_label = f"task{task_id:04d}_trial{trial_id:02d}"

                    obs = None
                    episode_error = None
                    success = 0
                    env.reset()
                    obs = env.set_init_state(initial_states[trial_id])

                    max_steps = MAX_STEPS.get(dataset, 200)
                    if args.max_eval_steps is not None:
                        max_steps = min(max_steps, args.max_eval_steps)

                    replay_images = []
                    step_ids = []
                    selected_metric_series = []
                    control_score_series = []
                    uncertain_decision_series = []
                    used_deepthink_series = []
                    farmass_log_m_series = []
                    farmass_log_z_series = []
                    farmass_log_s_series = []
                    farmass_log_conflicted_series = []
                    farmass_log_fired_series = []
                    openvla_inference_times = []
                    deepthink_inference_times = []
                    selected_inference_times = []
                    selected_policy_series = []
                    ee_positions = []
                    proprio_states = []
                    far_mass_per_slot_series = []
                    peak_separation_per_slot_series = []
                    far_mass_x_peak_separation_per_slot_series = []
                    openvla_actions = []
                    deepthink_actions = []
                    executed_actions = []
                    reasoning_records = []

                    first_deepthink_request = True
                    deepthink_queue: list[np.ndarray] = []
                    deepthink_reasoning = ""
                    cached_reasoning_prefix = ""
                    farmass_trigger = FarmassLogContrastTrigger(
                        short_window=args.short_window,
                        long_window=args.long_window,
                        h_hi=args.h_hi,
                        h_lo=args.h_lo,
                        refire_max_fires=(
                            args.refire_max_fires if args.trigger_mode == "refire" else 1
                        ),
                        refire_interval=args.refire_interval,
                        refire_min_score=args.refire_min_score,
                    )
                    episode_rng = np.random.default_rng(
                        int(args.seed) + int(task_id) * 100_000 + int(trial_id)
                    )
                    t = 0
                    done = False

                    try:
                        while t < max_steps + args.num_steps_wait:
                            if t < args.num_steps_wait:
                                obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                                t += 1
                                continue

                            img = get_libero_image(obs, 224)
                            if args.save_video:
                                replay_images.append(img)

                            observation = {
                                "full_image": img,
                                "state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                            }
                            ee_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                            proprio_states.append(np.asarray(observation["state"], dtype=np.float32))
                            openvla_out = openvla_worker.infer(observation, task_description)
                            openvla_inference_times.append(openvla_out.inference_time)

                            action_uncertainty = compute_step_uncertainty_from_action_scores(
                                openvla_out.action_scores
                            )
                            metric_key = f"{args.uncertainty_metric_name}_mean"
                            farmass_slots = np.asarray(
                                action_uncertainty["far_mass_x_peak_separation_per_slot"],
                                dtype=np.float64,
                            )
                            far_mass_per_slot_series.append(
                                np.asarray(action_uncertainty["far_mass_per_slot"], dtype=np.float32)
                            )
                            peak_separation_per_slot_series.append(
                                np.asarray(action_uncertainty["peak_separation_per_slot"], dtype=np.float32)
                            )
                            far_mass_x_peak_separation_per_slot_series.append(
                                np.asarray(
                                    action_uncertainty["far_mass_x_peak_separation_per_slot"],
                                    dtype=np.float32,
                                )
                            )
                            farmass_arm_m = float(np.nanmean(farmass_slots[:6]))
                            if args.router_control_mode == "farmass_log_contrast":
                                selected_metric_score = farmass_arm_m
                            else:
                                selected_metric_score = float(action_uncertainty[metric_key])
                            selected_metric_series.append(selected_metric_score)
                            farmass_log_m_series.append(farmass_arm_m)

                            log_z = 0.0
                            log_s = 0.0
                            log_conflicted = False
                            log_fired = False
                            force_new_deepthink_request = False
                            if args.router_control_mode == "farmass_log_contrast":
                                log_z, log_s, log_conflicted, log_fired = farmass_trigger.update(
                                    farmass_log_m_series
                                )
                                control_score = float(log_s)
                                if args.trigger_mode == "sustain":
                                    is_uncertain = bool(log_conflicted)
                                    force_new_deepthink_request = bool(log_conflicted)
                                else:
                                    is_uncertain = bool(log_fired or (log_conflicted and deepthink_queue))
                                    force_new_deepthink_request = bool(log_fired)
                            elif args.router_control_mode == "fixed_interval":
                                control_step_idx = len(step_ids)
                                is_uncertain = control_step_idx % args.fixed_deepthink_interval == 0
                                control_score = float(is_uncertain)
                            elif args.router_control_mode == "random":
                                control_score = float(episode_rng.random())
                                is_uncertain = control_score < args.random_deepthink_probability
                            else:
                                control_score = compute_control_score(
                                    args.router_control_mode,
                                    selected_metric_series,
                                    args.tv_window,
                                )
                                if high_score_means_uncertain:
                                    is_uncertain = control_score > args.score_threshold
                                else:
                                    is_uncertain = control_score < args.score_threshold
                            farmass_log_z_series.append(float(log_z))
                            farmass_log_s_series.append(float(log_s))
                            farmass_log_conflicted_series.append(float(log_conflicted))
                            farmass_log_fired_series.append(float(log_fired))
                            control_score_series.append(float(control_score))
                            uncertain_decision_series.append(float(is_uncertain))

                            deepthink_time = 0.0
                            if is_uncertain:
                                if (
                                    args.router_control_mode == "farmass_log_contrast"
                                    and args.trigger_mode == "sustain"
                                ):
                                    deepthink_queue = []
                                if force_new_deepthink_request or not deepthink_queue:
                                    deep_obs = deepthink_observation_from_obs(obs, args.deepthink_image_size)
                                    npz_path = write_npz(tmp_dir, eval_label, t, deep_obs)
                                    try:
                                        deepthink_resp = deepthink_worker.request(
                                            npz_path=npz_path,
                                            task=task_description,
                                            reset=first_deepthink_request,
                                        )
                                        first_deepthink_request = False
                                    finally:
                                        if not args.keep_tmp_npz:
                                            try:
                                                npz_path.unlink()
                                            except FileNotFoundError:
                                                pass
                                    chunk = normalize_deepthink_actions(deepthink_resp["actions"])
                                    chunk_len = max(1, min(args.deepthink_execute_chunk_steps, len(chunk)))
                                    deepthink_queue = [a.copy() for a in chunk[:chunk_len]]
                                    deepthink_time = float(deepthink_resp.get("inference_time", 0.0))
                                    deepthink_reasoning = str(deepthink_resp.get("reasoning", ""))
                                    cached_reasoning_prefix = extract_cached_reasoning_prefix(deepthink_reasoning)
                                    reasoning_records.append(
                                        {
                                            "step": t,
                                            "task_id": task_id,
                                            "trial_id": trial_id,
                                            "metric_score": selected_metric_score,
                                            "control_score": control_score,
                                            "farmass_log_m": farmass_arm_m,
                                            "farmass_log_z": log_z,
                                            "farmass_log_s": log_s,
                                            "conflicted": log_conflicted,
                                            "fired": log_fired,
                                            "cached_reasoning_prefix": cached_reasoning_prefix,
                                            "reasoning": deepthink_reasoning,
                                        }
                                    )
                                action = deepthink_queue.pop(0)
                                selected_policy = "deepthink"
                                selected_time = openvla_out.inference_time + deepthink_time
                                if args.router_control_mode == "random":
                                    print(
                                        f"[router] random draw={control_score:.6f} "
                                        f"< p={args.random_deepthink_probability:.6f} -> DeepThinkVLA"
                                    )
                                elif args.router_control_mode == "farmass_log_contrast":
                                    print(
                                        f"[router] farmass_log_contrast s={control_score:.6f}, "
                                        f"conflicted={log_conflicted}, fired={log_fired}, "
                                        f"mode={args.trigger_mode} -> DeepThinkVLA"
                                    )
                                else:
                                    print(
                                        f"[router] uncertain score={control_score:.6f} "
                                        f"{threshold_operator} {args.score_threshold:.6f} -> DeepThinkVLA"
                                    )
                            else:
                                if args.router_control_mode != "farmass_log_contrast":
                                    deepthink_queue = []
                                    deepthink_reasoning = ""
                                    cached_reasoning_prefix = ""
                                action = prepare_openvla_action(openvla_out.action)
                                selected_policy = "openvla"
                                selected_time = openvla_out.inference_time
                                if args.router_control_mode == "random":
                                    print(
                                        f"[router] random draw={control_score:.6f} "
                                        f">= p={args.random_deepthink_probability:.6f} -> OpenVLA"
                                    )
                                elif args.router_control_mode == "farmass_log_contrast":
                                    print(
                                        f"[router] farmass_log_contrast s={control_score:.6f}, "
                                        f"conflicted={log_conflicted}, fired={log_fired}, "
                                        f"mode={args.trigger_mode} -> OpenVLA"
                                    )
                                else:
                                    print(
                                        f"[router] certain score={control_score:.6f}, "
                                        f"threshold={args.score_threshold:.6f} -> OpenVLA"
                                    )

                            step_ids.append(t)
                            used_deepthink_series.append(float(selected_policy == "deepthink"))
                            selected_inference_times.append(float(selected_time))
                            deepthink_inference_times.append(float(deepthink_time))
                            selected_policy_series.append(selected_policy)
                            openvla_actions.append(prepare_openvla_action(openvla_out.action))
                            deepthink_actions.append(action.copy() if selected_policy == "deepthink" else np.full(7, np.nan))
                            executed_actions.append(action.copy())

                            print(f"Step: {t}")
                            print(f"OpenVLA inference time: {openvla_out.inference_time:.4f} seconds")
                            if selected_policy == "deepthink" and deepthink_time > 0:
                                print(f"DeepThink inference time: {deepthink_time:.4f} seconds")
                            print(f"Selected policy: {selected_policy}")
                            print(f"Action: {action}")

                            obs, _, done, _ = env.step(action.tolist())
                            if done:
                                success = 1
                                total_successes += 1
                                break
                            t += 1
                    except Exception as exc:
                        episode_error = f"{type(exc).__name__}: {exc}"
                        print(f"Caught episode exception: {episode_error}")

                    total_episodes += 1
                    row = {
                        "task_suite": dataset,
                        "task_name": task_slug,
                        "task_id": int(task_id),
                        "trial_id": int(trial_id),
                        "success": int(success),
                        "error": episode_error,
                        "num_steps": int(len(step_ids)),
                        "deepthink_ratio": mean_or_none(used_deepthink_series),
                        "cot_ratio": mean_or_none(used_deepthink_series),
                        "num_deepthink_steps": int(np.sum(used_deepthink_series)) if used_deepthink_series else 0,
                        "num_fires": int(np.sum(farmass_log_fired_series)) if farmass_log_fired_series else 0,
                        "mean_openvla_metric": mean_or_none(selected_metric_series),
                        "mean_control_score": mean_or_none(control_score_series),
                        "mean_openvla_inference_time": mean_or_none(openvla_inference_times),
                        "mean_deepthink_inference_time": mean_or_none(
                            [x for x in deepthink_inference_times if x > 0]
                        ),
                        "mean_selected_inference_time": mean_or_none(selected_inference_times),
                        "router_config": router_config,
                        "openvla_checkpoint": str(openvla_checkpoint),
                        "deepthink_checkpoint": str(args.deepthink_checkpoint),
                        "save_dir": str(save_dir),
                    }
                    append_summary(run_root, row)

                    if args.save_rollout_pt:
                        save_deepthink_rollout_pt(
                            episode_dir=save_dir,
                            task_name=task_slug,
                            task_id=task_id,
                            trial_id=trial_id,
                            success=success,
                            step_ids=step_ids,
                            selected_metric_series=selected_metric_series,
                            farmass_log_m_series=farmass_log_m_series,
                            farmass_log_z_series=farmass_log_z_series,
                            farmass_log_s_series=farmass_log_s_series,
                            farmass_log_conflicted_series=farmass_log_conflicted_series,
                            farmass_log_fired_series=farmass_log_fired_series,
                            control_score_series=control_score_series,
                            uncertain_decision_series=uncertain_decision_series,
                            used_deepthink_series=used_deepthink_series,
                            openvla_inference_times=openvla_inference_times,
                            deepthink_inference_times=deepthink_inference_times,
                            selected_inference_times=selected_inference_times,
                            ee_positions=ee_positions,
                            proprio_states=proprio_states,
                            far_mass_per_slot_series=far_mass_per_slot_series,
                            peak_separation_per_slot_series=peak_separation_per_slot_series,
                            far_mass_x_peak_separation_per_slot_series=far_mass_x_peak_separation_per_slot_series,
                            openvla_actions=openvla_actions,
                            deepthink_actions=deepthink_actions,
                            executed_actions=executed_actions,
                            selected_policy_series=selected_policy_series,
                            reasoning_records=reasoning_records,
                            router_config=router_config,
                        )
                    if args.save_reasoning and reasoning_records:
                        save_reasoning_jsonl(save_dir, reasoning_records)
                    if args.save_video and replay_images:
                        save_rollout_video(
                            replay_images,
                            trial_id,
                            success=bool(success),
                            task_description=task_description,
                            save_dir=str(save_dir),
                        )

                    print(
                        f"Episode done: success={success} steps={len(step_ids)} "
                        f"deepthink_ratio={row['deepthink_ratio']}"
                    )
                    print(f"Running total: successes={total_successes}/{total_episodes}")
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    finally:
        deepthink_worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="libero_10", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--trial-ids", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-eval-steps", type=int, default=None)

    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "rollouts" / "openvla_deepthink_dual"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tmp-dir", default=str(PROJECT_ROOT / "tmp" / "openvla_deepthink"))
    parser.add_argument("--keep-tmp-npz", action="store_true")
    parser.add_argument("--skip-existing-rollouts", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-reasoning", action="store_true")

    parser.add_argument("--openvla-checkpoint", default=None)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--deepthink-python", default=DEFAULT_DEEPTHINK_PYTHON)
    parser.add_argument("--deepthink-server", default="experiments/libero/deepthinkvla_jsonl_server.py")
    parser.add_argument("--deepthink-checkpoint", default=DEFAULT_DEEPTHINK_CHECKPOINT)
    parser.add_argument("--deepthink-repo-root", default=DEFAULT_DEEPTHINK_REPO_ROOT)
    parser.add_argument("--deepthink-device", default="cuda:0")
    parser.add_argument("--deepthink-cuda-visible-devices", default="2")
    parser.add_argument("--deepthink-compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--deepthink-num-images", type=int, default=2)
    parser.add_argument("--deepthink-image-size", type=int, default=224)
    parser.add_argument("--deepthink-max-new-tokens", type=int, default=2048)
    parser.add_argument("--deepthink-masked-cot", action="store_true")
    parser.add_argument("--deepthink-execute-chunk-steps", type=int, default=10)

    parser.add_argument(
        "--router-control-mode",
        "--control-mode",
        dest="router_control_mode",
        default="metric_window_total_variation",
        choices=[
            "metric",
            "metric_window",
            "metric_variance",
            "metric_total_variation",
            "metric_window_total_variation",
            "fixed_interval",
            "random",
            "farmass_log_contrast",
        ],
    )
    parser.add_argument(
        "--uncertainty-metric-name",
        default="far_mass_x_peak_separation",
        choices=[
            "entropy",
            "top3_mass",
            "dist_aware_dispersion",
            "far_mass_x_peak_separation",
            "bimodal_u",
            "second_peak_mass_x_peak_distance",
            "dist_entropy",
        ],
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Defaults by suite: spatial=1.05, object=1.21, goal=0.79, libero_10=0.93.",
    )
    parser.add_argument("--score-threshold-direction", default="gt", choices=["auto", "gt", "lt"])
    parser.add_argument("--tv-window", type=int, default=5)
    parser.add_argument("--fixed-deepthink-interval", type=int, default=5)
    parser.add_argument("--random-deepthink-probability", type=float, default=0.5)
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=10)
    parser.add_argument("--h-hi", type=float, default=1.5)
    parser.add_argument("--h-lo", type=float, default=0.5)
    parser.add_argument("--trigger-mode", default="onset", choices=["onset", "sustain", "refire"])
    parser.add_argument("--refire-max-fires", type=int, default=1)
    parser.add_argument("--refire-interval", type=int, default=5)
    parser.add_argument(
        "--refire-min-score",
        type=float,
        default=None,
        help="Minimum s_t required for conflicted-state refires; defaults to (h_hi + h_lo) / 2.",
    )
    parser.add_argument(
        "--save-rollout-pt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode .pt files with trigger traces and actions.",
    )

    args = parser.parse_args()
    eval_openvla_deepthink(args)


if __name__ == "__main__":
    main()
