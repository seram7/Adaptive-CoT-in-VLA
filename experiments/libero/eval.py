# Cell 1: imports
import inspect
import os
import textwrap
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from pathlib import Path
from typing import Dict, List, Optional
import torch.nn.functional as F

from libero_utils import UNNORM_KEYS

import sys

sys.path.insert(0, "/home/seram/ut/project/LIBERO")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
    save_rollot_reasoning,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_processor

from libero.libero import benchmark

print("libero imported : OK")

ACTION_TOKEN_BEGIN_IDX = 31743
N_ACTION_BINS = 256
ACTION_TOKEN_START = ACTION_TOKEN_BEGIN_IDX + 1
ACTION_TOKEN_END = ACTION_TOKEN_START + N_ACTION_BINS


# ---------------------------------------------------------------------------
# Action token range
# ---------------------------------------------------------------------------
# Adjust if needed for your tokenizer.
ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 32000

# ---------------------------------------------------------------------------
# Reasoning prefix extraction
# ---------------------------------------------------------------------------


def extract_cached_reasoning_prefix(decoded_text: str) -> str:
    """Parse the decoded generation to extract a reusable reasoning prefix."""
    text = decoded_text

    if "\nOut: " in text:
        text = text.split("\nOut: ")[-1]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()

    if " MOVE REASONING: " in text:
        return text.split(" MOVE REASONING: ")[0] + " MOVE REASONING: "

    if " GRIPPER POSITION: " in text:
        return text.split(" GRIPPER POSITION: ")[0] + " GRIPPER POSITION: "

    return "TASK:"


# ---------------------------------------------------------------------------
# Uncertainty metric helpers
# ---------------------------------------------------------------------------


def _smooth_probs_1d(probs: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    probs: [N, B]
    returns: [N, B]
    """
    if kernel_size <= 1:
        return probs
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=probs.device, dtype=probs.dtype) / kernel_size
    x = probs.unsqueeze(1)  # [N,1,B]
    x = F.pad(x, (pad, pad), mode="replicate")
    y = F.conv1d(x, weight)
    return y.squeeze(1)


def _local_maxima_mask(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, B]
    """
    left = torch.empty_like(x)
    right = torch.empty_like(x)

    left[:, 0] = -torch.inf
    left[:, 1:] = x[:, :-1]

    right[:, -1] = -torch.inf
    right[:, :-1] = x[:, 1:]

    return (x >= left) & (x >= right)


def _window_mass(probs: torch.Tensor, centers: torch.Tensor, radius: int) -> torch.Tensor:
    """
    probs: [N, B]
    centers: [N]
    """
    B = probs.shape[-1]
    idx = torch.arange(B, device=probs.device).unsqueeze(0)
    mask = (idx - centers.unsqueeze(1)).abs() <= radius
    return (probs * mask).sum(dim=-1)


def _top2_peaks(smoothed_probs: torch.Tensor, min_peak_prob: float = 0.01):
    peak_mask = _local_maxima_mask(smoothed_probs)
    peak_mask = peak_mask & (smoothed_probs >= min_peak_prob)

    peak_vals = torch.where(
        peak_mask,
        smoothed_probs,
        torch.full_like(smoothed_probs, -torch.inf),
    )

    top2_vals, top2_idx = torch.topk(peak_vals, k=2, dim=-1)
    peak1_val = top2_vals[:, 0]
    peak2_val = top2_vals[:, 1]
    peak1_idx = top2_idx[:, 0]
    peak2_idx = top2_idx[:, 1]
    has_peak2 = torch.isfinite(peak2_val)

    return peak1_idx, peak2_idx, peak1_val, peak2_val, has_peak2


def compute_step_uncertainty_from_action_scores(
    action_scores,
    smooth_kernel: int = 5,
    near_radius: int = 2,
    far_radius: int = 20,
    bimodal_tau: int = 50,
    min_peak_prob: float = 0.01,
) -> Dict[str, object]:
    """
    action_scores: list/tuple of length 7
        each element shape [1, vocab_size] or [vocab_size]

    returns dict with per-slot [7] metrics and mean scalars.
    """
    step_logits = torch.stack(
        [x.squeeze(0).float().cpu() for x in action_scores], dim=0
    )  # [7, vocab_size]

    action_logits = step_logits[:, ACTION_TOKEN_START:ACTION_TOKEN_END]  # [7, 256]
    probs = torch.softmax(action_logits, dim=-1)  # [7, 256]

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [7]
    top3_mass = torch.topk(probs, k=3, dim=-1).values.sum(dim=-1)  # [7]

    _, B = probs.shape
    idx = torch.arange(B, dtype=probs.dtype).unsqueeze(0)  # [1, B]

    # 1) distance-aware dispersion (variance in bin space)
    mu = (probs * idx).sum(dim=-1)  # [7]
    dist_aware_dispersion = (probs * (idx - mu.unsqueeze(1)) ** 2).sum(dim=-1)  # [7]

    # Smoothed peak structure
    s_probs = _smooth_probs_1d(probs, kernel_size=smooth_kernel)  # [7, 256]
    map_idx = s_probs.argmax(dim=-1)  # [7]

    far_mask = (torch.arange(B).unsqueeze(0) - map_idx.unsqueeze(1)).abs() > far_radius
    far_mask = far_mask.to(probs.dtype)
    far_mass = (probs * far_mask).sum(dim=-1)  # [7]

    peak1_idx, peak2_idx, _, _, has_peak2 = _top2_peaks(s_probs, min_peak_prob=min_peak_prob)
    peak_separation = (peak1_idx - peak2_idx).abs().to(probs.dtype)
    peak_separation = torch.where(has_peak2, peak_separation, torch.zeros_like(peak_separation))

    # 2) far_mass x peak_separation
    far_mass_x_peak_separation = far_mass * peak_separation

    # 3) bimodal_u
    dist_from_map = (torch.arange(B).unsqueeze(0) - map_idx.unsqueeze(1)).abs().to(probs.dtype)
    bimodal_u = (probs * torch.clamp(dist_from_map, max=float(bimodal_tau))).sum(dim=-1)

    # 4) second_peak_mass x peak distance
    second_peak_mass = _window_mass(probs, peak2_idx, radius=near_radius)
    second_peak_mass = torch.where(has_peak2, second_peak_mass, torch.zeros_like(second_peak_mass))
    second_peak_mass_x_peak_distance = second_peak_mass * peak_separation

    return {
        "entropy_per_slot": entropy.numpy(),
        "top3_mass_per_slot": top3_mass.numpy(),
        "dist_aware_dispersion_per_slot": dist_aware_dispersion.numpy(),
        "far_mass_x_peak_separation_per_slot": far_mass_x_peak_separation.numpy(),
        "bimodal_u_per_slot": bimodal_u.numpy(),
        "second_peak_mass_x_peak_distance_per_slot": second_peak_mass_x_peak_distance.numpy(),
        "entropy_mean": float(entropy.mean().item()),
        "top3_mass_mean": float(top3_mass.mean().item()),
        "dist_aware_dispersion_mean": float(dist_aware_dispersion.mean().item()),
        "far_mass_x_peak_separation_mean": float(far_mass_x_peak_separation.mean().item()),
        "bimodal_u_mean": float(bimodal_u.mean().item()),
        "second_peak_mass_x_peak_distance_mean": float(second_peak_mass_x_peak_distance.mean().item()),
        "logits_7xV": step_logits,
    }


def compute_windowed_avg_total_variation(series, window: int = 5) -> float:
    if series is None or len(series) < 2:
        return 0.0
    arr = np.asarray(series[-window:], dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.nanmean(np.abs(np.diff(arr))))


# ---------------------------------------------------------------------------
# Metric series utilities
# ---------------------------------------------------------------------------


def compute_running_total_variation(series: List[float]) -> float:
    """
    Running total variation up to the current step:
        TV_t = sum_{i=1..t} |x_i - x_{i-1}|
    """
    if series is None or len(series) < 2:
        return 0.0
    arr = np.asarray(series, dtype=np.float64)
    return float(np.abs(np.diff(arr)).sum())


def compute_relative_change(series: List[float]) -> float:
    """
    Relative change using the last two values.
    """
    if series is None or len(series) < 2:
        return 0.0
    prev = float(series[-2])
    cur = float(series[-1])
    return (cur - prev) / (abs(prev) + 1e-6)


# ---------------------------------------------------------------------------
# FrozenPromptController
# ---------------------------------------------------------------------------


class FrozenPromptController:
    """
    Controls whether the reasoning prefix should stay frozen or reset.

    control_mode:
        - "none": always update from latest decoded text
        - "time": fixed-length freezing
        - "metric": use current metric value
        - "metric_variance": use relative change of the metric
        - "metric_total_variation": use running total variation of the metric
        - "metric_window_total_variation": use windowed average total variation
    """

    def __init__(
        self,
        max_freezing_time: int = 5,
        base_prefix: str = " TASK:",
        control_mode: str = "metric",
        score_threshold: float = 0.0,
        metric_name: str = "second_peak_mass_x_peak_distance",
        high_score_means_uncertain: bool = True,
        tv_window: int = 5,
    ):
        self.max_freezing_time = max_freezing_time
        self.base_prefix = base_prefix
        self.frozen_prefix = base_prefix
        self.time_frozen = 0

        self.control_mode = control_mode
        self.score_threshold = score_threshold
        self.metric_name = metric_name
        self.high_score_means_uncertain = high_score_means_uncertain
        self.tv_window = tv_window

        self.fixed = False

        # NEW: counters
        self.num_prefix_updates = 0  # decoded prefix written into frozen_prefix
        self.num_prefix_resets = 0  # reset back to base_prefix
        self.num_prefix_changes = 0  # prefix string actually changed
        self.num_decisions = 0  # number of update_from_decoded calls

    def get_prefix_for_step(self) -> str:
        if self.control_mode == "time":
            if self.time_frozen <= 0:
                old_prefix = self.frozen_prefix
                self.frozen_prefix = self.base_prefix
                self.time_frozen = self.max_freezing_time
                if self.frozen_prefix != old_prefix:
                    self.num_prefix_changes += 1
                self.num_prefix_resets += 1
            self.time_frozen -= 1
        return self.frozen_prefix

    def _is_uncertain(self, score: float) -> bool:
        if self.high_score_means_uncertain:
            return score > self.score_threshold
        return score < self.score_threshold

    def _set_prefix(self, new_prefix: str, is_reset: bool = False, is_update: bool = False):
        old_prefix = self.frozen_prefix
        self.frozen_prefix = new_prefix

        if is_reset:
            self.num_prefix_resets += 1
        if is_update:
            self.num_prefix_updates += 1
        if new_prefix != old_prefix:
            self.num_prefix_changes += 1

    def update_from_decoded(self, decoded_text: str, scores: Optional[List[float]] = None) -> None:
        self.num_decisions += 1

        if self.control_mode == "time":
            if self.time_frozen == self.max_freezing_time - 1:
                new_prefix = extract_cached_reasoning_prefix(decoded_text)
                self._set_prefix(new_prefix, is_update=True)
            return

        if self.control_mode == "none":
            new_prefix = extract_cached_reasoning_prefix(decoded_text)
            self._set_prefix(new_prefix, is_update=True)
            return

        if scores is None or len(scores) == 0:
            raw_score = 0.0
        elif self.control_mode == "metric":
            raw_score = float(scores[-1])
        elif self.control_mode == "metric_variance":
            raw_score = compute_relative_change(scores)
        elif self.control_mode == "metric_total_variation":
            raw_score = compute_running_total_variation(scores)
        elif self.control_mode == "metric_window_total_variation":
            raw_score = compute_windowed_avg_total_variation(scores, window=self.tv_window)
        else:
            raw_score = float(scores[-1])

        uncertain = self._is_uncertain(raw_score)

        if uncertain:
            self._set_prefix(self.base_prefix, is_reset=True)
            self.fixed = False
            print(
                f"[{self.metric_name} | {self.control_mode}] "
                f"uncertainty high (score={raw_score:.6f}) -> reset to base"
            )
        else:
            new_prefix = extract_cached_reasoning_prefix(decoded_text)
            self._set_prefix(new_prefix, is_update=True)
            self.fixed = True
            print(
                f"[{self.metric_name} | {self.control_mode}] "
                f"uncertainty low (score={raw_score:.6f}) -> freeze/update prefix"
            )

    def get_stats(self) -> Dict[str, int]:
        return {
            "num_prefix_updates": int(self.num_prefix_updates),
            "num_prefix_resets": int(self.num_prefix_resets),
            "num_prefix_changes": int(self.num_prefix_changes),
            "num_decisions": int(self.num_decisions),
        }

    def reset(self) -> None:
        self.frozen_prefix = self.base_prefix
        self.time_frozen = 0
        self.fixed = False
        self.num_prefix_updates = 0
        self.num_prefix_resets = 0
        self.num_prefix_changes = 0
        self.num_decisions = 0


# ---------------------------------------------------------------------------
# Rollout saving
# ---------------------------------------------------------------------------


def save_rollout_pt(
    save_dir,
    task_name,
    task_id,
    trial_id,
    success,
    step_ids,
    rollout_logits,
    entropy_series,
    top3_mass_series,
    dist_aware_dispersion_series=None,
    far_mass_x_peak_separation_series=None,
    bimodal_u_series=None,
    second_peak_mass_x_peak_distance_series=None,
    selected_metric_series=None,
    inference_times=None,
    prefix_stats=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(rollout_logits) == 0:
        logits = torch.empty(0)
    else:
        logits = torch.stack(rollout_logits, dim=0).float()

    def _to_tensor_or_empty(x):
        if x is None or len(x) == 0:
            return torch.empty(0)
        return torch.tensor(x, dtype=torch.float32)

    payload = {
        "task_name": str(task_name),
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "success": int(success),
        "step_ids": np.asarray(step_ids, dtype=np.int32),
        "logits": logits,
        "entropy_series": _to_tensor_or_empty(entropy_series),
        "top3_mass_series": _to_tensor_or_empty(top3_mass_series),
        "dist_aware_dispersion_series": _to_tensor_or_empty(dist_aware_dispersion_series),
        "far_mass_x_peak_separation_series": _to_tensor_or_empty(far_mass_x_peak_separation_series),
        "bimodal_u_series": _to_tensor_or_empty(bimodal_u_series),
        "second_peak_mass_x_peak_distance_series": _to_tensor_or_empty(
            second_peak_mass_x_peak_distance_series
        ),
        "selected_metric_series": _to_tensor_or_empty(selected_metric_series),
        "inference_times": (
            np.asarray(inference_times, dtype=np.float32) if inference_times is not None else None
        ),
        "prefix_stats": prefix_stats if prefix_stats is not None else {},
    }

    pt_path = save_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
    torch.save(payload, pt_path)
    return pt_path


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def eval_libero_uncertainty(
    dataset: str = "libero_spatial",
    num_trials_per_task: int = 20,
    prompt_control_mode: str = "metric",
    frozen_prompt_max_freezing_time: int = 5,
    score_threshold: float = 0.0,
    uncertainty_metric_name: str = "second_peak_mass_x_peak_distance",
    tv_window: int = 5,
    frozen_prompt_base_prefix: str = " TASK:",
):
    """
    prompt_control_mode:
        - time
        - none
        - metric
        - metric_variance
        - metric_total_variation
        - metric_window_total_variation

    uncertainty_metric_name:
        - entropy
        - top3_mass
        - dist_aware_dispersion
        - far_mass_x_peak_separation
        - bimodal_u
        - second_peak_mass_x_peak_distance
    """

    checkpoint = f"leepanic/ecot-{dataset.replace('_', '-')}-r32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )

    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    vla.norm_stats[dataset] = UNNORM_KEYS[f"{dataset}_no_noops"]
    vla.norm_stats.update(UNNORM_KEYS)

    task_suite_name = dataset
    cfg = GenerateConfig(
        pretrained_checkpoint=f"leepanic/ecot-{dataset.replace('_', '-')}-r32",
        unnorm_key=task_suite_name,
        center_crop=True,
        num_trials_per_task=num_trials_per_task,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")

    if uncertainty_metric_name == "top3_mass":
        high_score_means_uncertain = False
    else:
        high_score_means_uncertain = True

    if prompt_control_mode == "none":
        use_frozen_prompt_controller = False
    else:
        use_frozen_prompt_controller = True
    print(f"FrozenPromptController: {'enabled' if use_frozen_prompt_controller else 'disabled'}")
    print(
        f"control_mode={prompt_control_mode}, "
        f"metric={uncertainty_metric_name}, "
        f"threshold={score_threshold}"
    )

    resize_size = 224
    total_episodes, total_successes = 0, 0
    num_steps_wait = 10

    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks", unit="task"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        episode_inference_stats = []

        for episode_idx in tqdm(range(num_trials_per_task)):
            print(f"\nTask: {task_description}")

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            if use_frozen_prompt_controller:
                controller = FrozenPromptController(
                    max_freezing_time=frozen_prompt_max_freezing_time,
                    base_prefix=frozen_prompt_base_prefix,
                    control_mode=prompt_control_mode,
                    score_threshold=score_threshold,
                    metric_name=uncertainty_metric_name,
                    high_score_means_uncertain=high_score_means_uncertain,
                    tv_window=tv_window,
                )
            else:
                prefix_text = frozen_prompt_base_prefix

            t = 0
            replay_images = []
            replay_reasoning = []
            inference_times = []

            if task_suite_name == "libero_spatial":
                max_steps = 200
            elif task_suite_name == "libero_object":
                max_steps = 280
            elif task_suite_name == "libero_goal":
                max_steps = 300
            elif task_suite_name == "libero_10":
                max_steps = 520
            elif task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 200

            print(f"Starting episode {task_episodes + 1}...")
            rollout_logits = []
            step_ids = []

            entropy_series = []
            top3_mass_series = []
            dist_aware_dispersion_series = []
            far_mass_x_peak_separation_series = []
            bimodal_u_series = []
            second_peak_mass_x_peak_distance_series = []
            selected_metric_series = []

            task_description_slug = task_description.replace(" ", "_")
            if prompt_control_mode == "none":
                save_dir = f"./rollouts/{task_suite_name}/ECoT_plain/{task_description_slug}/trial{episode_idx:02d}"
            if prompt_control_mode == "time":
                save_dir = (
                    f"./rollouts/{task_suite_name}/ECoT_frozen{frozen_prompt_max_freezing_time}"
                    f"/{task_description_slug}/trial{episode_idx:02d}"
                )
            else:
                if "window" in prompt_control_mode:
                    save_dir = (
                        f"./rollouts/{task_suite_name}/{uncertainty_metric_name}_{prompt_control_mode}_{score_threshold}_w{tv_window}"
                        f"/{task_description_slug}/trial{episode_idx:02d}"
                    )
                else:
                    save_dir = (
                        f"./rollouts/{task_suite_name}/{uncertainty_metric_name}_{prompt_control_mode}_{score_threshold}"
                        f"/{task_description_slug}/trial{episode_idx:02d}"
                    )
            success = 0
            done = False

            while t < max_steps + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
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

                    if use_frozen_prompt_controller:
                        prefix_text = controller.get_prefix_for_step()

                    inference_time, action, generated_ids, action_scores = get_action(
                        cfg,
                        vla,
                        observation,
                        task_description,
                        processor=processor,
                        max_new_tokens=1024,
                        prefix_text=prefix_text,
                    )
                    inference_times.append(inference_time)

                    generated_text = processor.batch_decode(generated_ids)[0]
                    reasoning_text = "##S##" + prefix_text + "##E##" + generated_text
                    replay_reasoning.append(reasoning_text)

                    action_uncertainty = compute_step_uncertainty_from_action_scores(action_scores)

                    rollout_logits.append(action_uncertainty["logits_7xV"])
                    step_ids.append(t)

                    entropy_series.append(action_uncertainty["entropy_mean"])
                    top3_mass_series.append(action_uncertainty["top3_mass_mean"])
                    dist_aware_dispersion_series.append(action_uncertainty["dist_aware_dispersion_mean"])
                    far_mass_x_peak_separation_series.append(
                        action_uncertainty["far_mass_x_peak_separation_mean"]
                    )
                    bimodal_u_series.append(action_uncertainty["bimodal_u_mean"])
                    second_peak_mass_x_peak_distance_series.append(
                        action_uncertainty["second_peak_mass_x_peak_distance_mean"]
                    )

                    metric_key = f"{uncertainty_metric_name}_mean"
                    selected_metric_score = float(action_uncertainty[metric_key])
                    selected_metric_series.append(selected_metric_score)

                    if use_frozen_prompt_controller:
                        controller.update_from_decoded(
                            generated_text,
                            scores=selected_metric_series,
                        )

                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)

                    print(f"Step: {t}")
                    print(f"Inference time: {inference_time:.4f} seconds")
                    print(f"Action: {action}")

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        success = 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    break

            if use_frozen_prompt_controller:
                prefix_stats = controller.get_stats()
            else:
                prefix_stats = {
                    "num_prefix_updates": 0,
                    "num_prefix_resets": 0,
                    "num_prefix_changes": 0,
                    "num_decisions": 0,
                }

            save_rollout_pt(
                save_dir=save_dir,
                task_name=task_description_slug,
                task_id=task_id,
                trial_id=episode_idx,
                success=success,
                step_ids=step_ids,
                rollout_logits=rollout_logits,
                entropy_series=entropy_series,
                top3_mass_series=top3_mass_series,
                dist_aware_dispersion_series=dist_aware_dispersion_series,
                far_mass_x_peak_separation_series=far_mass_x_peak_separation_series,
                bimodal_u_series=bimodal_u_series,
                second_peak_mass_x_peak_distance_series=second_peak_mass_x_peak_distance_series,
                selected_metric_series=selected_metric_series,
                inference_times=inference_times,
                prefix_stats=prefix_stats,
            )

            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                save_dir=save_dir,
            )
            save_rollot_reasoning(
                replay_reasoning,
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                save_dir=save_dir,
            )

            episode_inference_stats.append(
                {
                    "task_id": task_id,
                    "trial_id": episode_idx,
                    "task_description": task_description,
                    "success": success,
                    "num_inference_steps": len(step_ids),
                    "total_inference_time": sum(inference_times),
                    "mean_inference_time": np.mean(inference_times) if inference_times else 0.0,
                    "max_inference_time": max(inference_times) if inference_times else 0.0,
                    "inference_times": inference_times,
                    "num_prefix_updates": prefix_stats["num_prefix_updates"],
                    "num_prefix_resets": prefix_stats["num_prefix_resets"],
                    "num_prefix_changes": prefix_stats["num_prefix_changes"],
                    "num_prefix_decisions": prefix_stats["num_decisions"],
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LIBERO uncertainty-controlled evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="libero_spatial",
        help="Name of the LIBERO dataset to evaluate on (e.g., 'libero_spatial', 'libero_object', 'libero_goal', 'libero_10', '(not) libero_90')",
    )
    parser.add_argument(
        "--num-trials-per-task",
        type=int,
        default=20,
        help="Number of trials to run per task",
    )
    parser.add_argument(
        "--prompt-control-mode",
        type=str,
        default="none",
        choices=[
            "none",  # ECoT
            "time",  # Freeze prefix prompt for a fixed number of steps after each update
            "metric",
            "metric_variance",
            "metric_total_variation",
            "metric_window_total_variation",
        ],
        help="Control mode for prompt freezing/resetting",
    )
    parser.add_argument(
        "--frozen-prompt-max-freezing-time",
        type=int,
        default=5,
        help="Steps to hold cached prefix in time mode",
    )
    parser.add_argument(
        "--uncertainty-metric-name",
        type=str,
        default="second_peak_mass_x_peak_distance",
        choices=[
            "entropy",
            "top3_mass",
            "dist_aware_dispersion",
            "far_mass_x_peak_separation",
            "bimodal_u",
            "second_peak_mass_x_peak_distance",
        ],
        help="Which uncertainty metric to use for prompt control",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Threshold for uncertainty-based control",
    )
    parser.add_argument(
        "--tv-window",
        type=int,
        default=5,
        help="Window size for metric_window_total_variation",
    )
    parser.add_argument(
        "--frozen-prompt-base-prefix",
        type=str,
        default=" TASK:",
        help="Initial/fallback prefix string",
    )

    args = parser.parse_args()

    eval_libero_uncertainty(
        dataset=args.dataset,
        num_trials_per_task=args.num_trials_per_task,
        frozen_prompt_max_freezing_time=args.frozen_prompt_max_freezing_time,
        frozen_prompt_base_prefix=args.frozen_prompt_base_prefix,
        prompt_control_mode=args.prompt_control_mode,
        score_threshold=args.score_threshold,
        uncertainty_metric_name=args.uncertainty_metric_name,
        tv_window=args.tv_window,
    )
