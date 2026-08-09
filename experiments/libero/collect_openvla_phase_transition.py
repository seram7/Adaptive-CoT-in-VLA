#!/usr/bin/env python
"""Collect minimal OpenVLA-only LIBERO rollouts for phase-transition analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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
    compute_step_uncertainty_from_action_scores,
    parse_int_filter,
    validate_local_checkpoint,
)
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
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
        "Could not import LIBERO. Install it or set LIBERO_ROOT before running."
    ) from exc


MAX_STEPS = {
    "libero_spatial": 200,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:90]


def prepare_openvla_action(action: np.ndarray) -> np.ndarray:
    action = normalize_gripper_action(action.copy(), binarize=True)
    return invert_gripper_action(action)


def parse_ids(value: str | None, default: range) -> list[int]:
    if value is None or value == "":
        return list(default)
    if "-" in value and "," not in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return parse_int_filter(value) or list(default)


def as_float_tensor(values) -> torch.Tensor:
    if values is None or len(values) == 0:
        return torch.empty(0)
    return torch.tensor(np.asarray(values), dtype=torch.float32)


def as_float_array(values) -> np.ndarray:
    if values is None or len(values) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def append_summary(run_dir: Path, row: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "episode_summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_rollout_pt(
    save_dir: Path,
    payload: dict[str, Any],
    openvla_logits: list[torch.Tensor],
    save_logits: bool,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_logits and openvla_logits:
        payload["openvla_logits"] = torch.stack(openvla_logits, dim=0).float()
    else:
        payload["openvla_logits"] = torch.empty(0)
    pt_path = save_dir / f"task{payload['task_id']:02d}_trial{payload['trial_id']:02d}.pt"
    torch.save(payload, pt_path)
    return pt_path


def collect(args: argparse.Namespace) -> None:
    if args.dataset not in DEFAULT_OPENVLA_CKPTS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    set_seed_everywhere(args.seed)
    checkpoint = args.openvla_checkpoint or DEFAULT_OPENVLA_CKPTS[args.dataset]
    validate_local_checkpoint(checkpoint)

    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name
    run_dir = output_root / args.dataset / run_name

    print(f"Seed: {args.seed}")
    print(f"Task suite: {args.dataset}")
    print(f"Run name: {run_name}")
    print(f"Output root: {output_root}")
    print(f"OpenVLA checkpoint: {checkpoint}")
    print(f"OpenVLA device: {args.openvla_device}")
    print(f"Save logits: {args.save_logits}")

    worker = OpenVLAWorker(
        checkpoint=checkpoint,
        dataset=args.dataset,
        device=args.openvla_device,
        dtype_name=args.torch_dtype,
        center_crop=True,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.dataset]()
    task_ids = parse_ids(args.task_ids, range(task_suite.n_tasks))
    trial_ids = parse_ids(args.trial_ids, range(args.num_trials_per_task))

    total_episodes = 0
    total_successes = 0
    for task_id in tqdm(task_ids, desc="Tasks", unit="task"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=args.env_resolution)
        task_slug = slugify(task_description)

        try:
            for trial_id in tqdm(trial_ids, desc="Trials", leave=False):
                save_dir = output_root / args.dataset / run_name / task_slug / f"trial{trial_id:02d}"
                pt_path = save_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
                if args.skip_existing and pt_path.exists():
                    print(f"Skipping existing {pt_path}")
                    continue

                print(f"\nTask {task_id} trial {trial_id}: {task_description}")
                env.reset()
                obs = env.set_init_state(initial_states[trial_id])

                max_steps = MAX_STEPS.get(args.dataset, 200)
                if args.max_eval_steps is not None:
                    max_steps = min(max_steps, args.max_eval_steps)

                success = 0
                episode_error = None
                replay_images = []
                step_ids = []
                openvla_logits = []
                proprio_states = []
                ee_positions = []
                ee_axisangles = []
                gripper_qpos = []
                raw_openvla_actions = []
                executed_actions = []
                inference_times = []

                entropy_per_slot = []
                top3_mass_per_slot = []
                dist_aware_dispersion_per_slot = []
                far_mass_per_slot = []
                peak_separation_per_slot = []
                far_mass_x_peak_separation_per_slot = []
                bimodal_u_per_slot = []
                dist_entropy_per_slot = []
                second_peak_mass_x_peak_distance_per_slot = []

                entropy_mean = []
                top3_mass_mean = []
                dist_aware_dispersion_mean = []
                far_mass_mean = []
                peak_separation_mean = []
                far_mass_x_peak_separation_mean = []
                bimodal_u_mean = []
                dist_entropy_mean = []
                second_peak_mass_x_peak_distance_mean = []

                t = 0
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
                        proprio_states.append(observation["state"].astype(np.float32))
                        ee_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
                        ee_axisangles.append(
                            np.asarray(
                                quat2axisangle(obs["robot0_eef_quat"].copy()),
                                dtype=np.float32,
                            )
                        )
                        gripper_qpos.append(
                            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
                        )

                        openvla_out = worker.infer(observation, task_description)
                        inference_times.append(float(openvla_out.inference_time))
                        action_uncertainty = compute_step_uncertainty_from_action_scores(
                            openvla_out.action_scores
                        )

                        if args.save_logits:
                            openvla_logits.append(action_uncertainty["logits_7xV"])
                        step_ids.append(t)
                        raw_openvla_actions.append(np.asarray(openvla_out.action, dtype=np.float32))
                        action = prepare_openvla_action(openvla_out.action)
                        executed_actions.append(action.copy())

                        entropy_per_slot.append(action_uncertainty["entropy_per_slot"])
                        top3_mass_per_slot.append(action_uncertainty["top3_mass_per_slot"])
                        dist_aware_dispersion_per_slot.append(
                            action_uncertainty["dist_aware_dispersion_per_slot"]
                        )
                        far_mass_per_slot.append(action_uncertainty["far_mass_per_slot"])
                        peak_separation_per_slot.append(
                            action_uncertainty["peak_separation_per_slot"]
                        )
                        far_mass_x_peak_separation_per_slot.append(
                            action_uncertainty["far_mass_x_peak_separation_per_slot"]
                        )
                        bimodal_u_per_slot.append(action_uncertainty["bimodal_u_per_slot"])
                        dist_entropy_per_slot.append(action_uncertainty["dist_entropy_per_slot"])
                        second_peak_mass_x_peak_distance_per_slot.append(
                            action_uncertainty["second_peak_mass_x_peak_distance_per_slot"]
                        )

                        entropy_mean.append(action_uncertainty["entropy_mean"])
                        top3_mass_mean.append(action_uncertainty["top3_mass_mean"])
                        dist_aware_dispersion_mean.append(
                            action_uncertainty["dist_aware_dispersion_mean"]
                        )
                        far_mass_mean.append(action_uncertainty["far_mass_mean"])
                        peak_separation_mean.append(action_uncertainty["peak_separation_mean"])
                        far_mass_x_peak_separation_mean.append(
                            action_uncertainty["far_mass_x_peak_separation_mean"]
                        )
                        bimodal_u_mean.append(action_uncertainty["bimodal_u_mean"])
                        dist_entropy_mean.append(action_uncertainty["dist_entropy_mean"])
                        second_peak_mass_x_peak_distance_mean.append(
                            action_uncertainty["second_peak_mass_x_peak_distance_mean"]
                        )

                        print(
                            f"Step: {t} inference={openvla_out.inference_time:.4f}s "
                            f"gripper={action[-1]:.1f} "
                            f"far_mass_x_peak_sep={far_mass_x_peak_separation_mean[-1]:.6f}"
                        )
                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            success = 1
                            total_successes += 1
                            break
                        t += 1
                except Exception as exc:
                    episode_error = f"{type(exc).__name__}: {exc}"
                    print(f"Caught episode exception: {episode_error}")

                payload = {
                    "task_suite": args.dataset,
                    "task_name": task_slug,
                    "task_id": int(task_id),
                    "trial_id": int(trial_id),
                    "success": int(success),
                    "error": episode_error,
                    "episode_length": int(len(step_ids)),
                    "step_ids": np.asarray(step_ids, dtype=np.int32),
                    "openvla_checkpoint": str(checkpoint),
                    "proprio_states": as_float_array(proprio_states),
                    "ee_positions": as_float_array(ee_positions),
                    "ee_axisangles": as_float_array(ee_axisangles),
                    "gripper_qpos": as_float_array(gripper_qpos),
                    "raw_openvla_actions": as_float_array(raw_openvla_actions),
                    "openvla_actions": as_float_array(executed_actions),
                    "executed_actions": as_float_array(executed_actions),
                    "openvla_inference_times": as_float_array(inference_times),
                    "inference_times": as_float_array(inference_times),
                    "entropy_per_slot_series": as_float_tensor(entropy_per_slot),
                    "top3_mass_per_slot_series": as_float_tensor(top3_mass_per_slot),
                    "dist_aware_dispersion_per_slot_series": as_float_tensor(
                        dist_aware_dispersion_per_slot
                    ),
                    "far_mass_per_slot_series": as_float_tensor(far_mass_per_slot),
                    "peak_separation_per_slot_series": as_float_tensor(
                        peak_separation_per_slot
                    ),
                    "far_mass_x_peak_separation_per_slot_series": as_float_tensor(
                        far_mass_x_peak_separation_per_slot
                    ),
                    "bimodal_u_per_slot_series": as_float_tensor(bimodal_u_per_slot),
                    "dist_entropy_per_slot_series": as_float_tensor(dist_entropy_per_slot),
                    "second_peak_mass_x_peak_distance_per_slot_series": as_float_tensor(
                        second_peak_mass_x_peak_distance_per_slot
                    ),
                    "entropy_series": as_float_tensor(entropy_mean),
                    "top3_mass_series": as_float_tensor(top3_mass_mean),
                    "dist_aware_dispersion_series": as_float_tensor(
                        dist_aware_dispersion_mean
                    ),
                    "far_mass_series": as_float_tensor(far_mass_mean),
                    "peak_separation_series": as_float_tensor(peak_separation_mean),
                    "far_mass_x_peak_separation_series": as_float_tensor(
                        far_mass_x_peak_separation_mean
                    ),
                    "selected_metric_series": as_float_tensor(
                        far_mass_x_peak_separation_mean
                    ),
                    "bimodal_u_series": as_float_tensor(bimodal_u_mean),
                    "dist_entropy_series": as_float_tensor(dist_entropy_mean),
                    "second_peak_mass_x_peak_distance_series": as_float_tensor(
                        second_peak_mass_x_peak_distance_mean
                    ),
                    "gripper_far_mass_series": as_float_tensor(
                        [row[-1] for row in far_mass_per_slot]
                    ),
                    "gripper_far_mass_x_peak_separation_series": as_float_tensor(
                        [row[-1] for row in far_mass_x_peak_separation_per_slot]
                    ),
                }
                saved_pt = save_rollout_pt(save_dir, payload, openvla_logits, args.save_logits)
                append_summary(
                    run_dir,
                    {
                        "task_suite": args.dataset,
                        "task_name": task_slug,
                        "task_id": int(task_id),
                        "trial_id": int(trial_id),
                        "success": int(success),
                        "error": episode_error,
                        "num_steps": int(len(step_ids)),
                        "mean_far_mass": (
                            float(np.mean(far_mass_mean)) if far_mass_mean else None
                        ),
                        "mean_peak_separation": (
                            float(np.mean(peak_separation_mean)) if peak_separation_mean else None
                        ),
                        "mean_far_mass_x_peak_separation": (
                            float(np.mean(far_mass_x_peak_separation_mean))
                            if far_mass_x_peak_separation_mean
                            else None
                        ),
                        "mean_openvla_inference_time": (
                            float(np.mean(inference_times)) if inference_times else None
                        ),
                        "openvla_checkpoint": str(checkpoint),
                        "save_dir": str(save_dir),
                        "pt_path": str(saved_pt),
                    },
                )
                total_episodes += 1

                if args.save_video and replay_images:
                    save_rollout_video(
                        replay_images,
                        trial_id,
                        success=bool(success),
                        task_description=task_description,
                        save_dir=str(save_dir),
                    )

                print(f"Saved {saved_pt}")
                print(f"Running total: successes={total_successes}/{total_episodes}")
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--num-trials-per-task", type=int, default=5)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--trial-ids", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="/scratch/seram7/Adaptive-CoT-in-VLA/rollouts_42_h100_new",
    )
    parser.add_argument("--run-name", default="openvla_phase_transition_minimal")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--openvla-checkpoint", default=None)
    parser.add_argument("--openvla-device", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
