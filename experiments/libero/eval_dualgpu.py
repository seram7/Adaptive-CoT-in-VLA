"""Dual-GPU OpenVLA/ECoT router evaluation for LIBERO.

The router always queries OpenVLA first, computes uncertainty from OpenVLA
action-token logits, and only queries ECoT on a second device when the control
score crosses the configured threshold.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
if os.environ.get("LIBERO_ROOT"):
    sys.path.insert(0, os.environ["LIBERO_ROOT"])

from experiments.libero.libero_utils import UNNORM_KEYS  # noqa: E402
try:  # noqa: E402
    from libero.libero import benchmark
except ImportError as exc:  # noqa: E402
    raise ImportError(
        "Could not import LIBERO. Install it with `pip install -e /path/to/LIBERO` "
        "or set LIBERO_ROOT=/path/to/LIBERO before running eval_dualgpu.py."
    ) from exc

try:  # noqa: E402
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
        save_rollot_reasoning,
        save_rollout_video,
    )
except ImportError as exc:  # noqa: E402
    raise ImportError(
        "Could not import LIBERO environment utilities. Install LIBERO and make sure "
        "`import libero` works before running eval_dualgpu.py."
    ) from exc
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import (  # noqa: E402
    PrismaticImageProcessor,
    PrismaticProcessor,
)

try:  # noqa: E402
    from experiments.libero.uncertainty_video import save_uncertainty_rollout_video
except ImportError:  # noqa: E402
    save_uncertainty_rollout_video = None


OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

DEFAULT_OPENVLA_CKPTS = {
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
    "libero_10": "openvla/openvla-7b-finetuned-libero-10",
}

DEFAULT_ECOT_CKPTS = {
    "libero_spatial": "leepanic/ecot-libero-spatial-r32",
    "libero_object": "leepanic/ecot-libero-object-r32",
    "libero_goal": "leepanic/ecot-libero-goal-r32",
    "libero_10": "leepanic/ecot-libero-10-r32",
}

ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 32000


def _smooth_probs_1d(probs: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return probs
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, device=probs.device, dtype=probs.dtype) / kernel_size
    x = probs.unsqueeze(1)
    x = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(x, weight).squeeze(1)


def _local_maxima_mask(x: torch.Tensor) -> torch.Tensor:
    left = torch.empty_like(x)
    right = torch.empty_like(x)
    left[:, 0] = -torch.inf
    left[:, 1:] = x[:, :-1]
    right[:, -1] = -torch.inf
    right[:, :-1] = x[:, 1:]
    return (x >= left) & (x >= right)


def _window_mass(probs: torch.Tensor, centers: torch.Tensor, radius: int) -> torch.Tensor:
    bins = probs.shape[-1]
    idx = torch.arange(bins, device=probs.device).unsqueeze(0)
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
    peak1_idx = top2_idx[:, 0]
    peak2_idx = top2_idx[:, 1]
    peak2_val = top2_vals[:, 1]
    has_peak2 = torch.isfinite(peak2_val)
    return peak1_idx, peak2_idx, has_peak2


def compute_step_uncertainty_from_action_scores(
    action_scores,
    smooth_kernel: int = 5,
    near_radius: int = 2,
    far_radius: int = 20,
    bimodal_tau: int = 50,
    min_peak_prob: float = 0.01,
) -> Dict[str, object]:
    step_logits = torch.stack([x.squeeze(0).float().cpu() for x in action_scores], dim=0)
    action_logits = step_logits[:, ACTION_TOKEN_START:ACTION_TOKEN_END]
    probs = torch.softmax(action_logits, dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    top3_mass = torch.topk(probs, k=3, dim=-1).values.sum(dim=-1)

    _, bins = probs.shape
    idx = torch.arange(bins, dtype=probs.dtype).unsqueeze(0)
    mu = (probs * idx).sum(dim=-1)
    dist_aware_dispersion = (probs * (idx - mu.unsqueeze(1)) ** 2).sum(dim=-1)

    smoothed_probs = _smooth_probs_1d(probs, kernel_size=smooth_kernel)
    map_idx = smoothed_probs.argmax(dim=-1)

    far_mask = (torch.arange(bins).unsqueeze(0) - map_idx.unsqueeze(1)).abs() > far_radius
    far_mass = (probs * far_mask.to(probs.dtype)).sum(dim=-1)

    peak1_idx, peak2_idx, has_peak2 = _top2_peaks(
        smoothed_probs, min_peak_prob=min_peak_prob
    )
    peak_separation = (peak1_idx - peak2_idx).abs().to(probs.dtype)
    peak_separation = torch.where(has_peak2, peak_separation, torch.zeros_like(peak_separation))

    far_mass_x_peak_separation = far_mass * peak_separation
    dist_from_map = (torch.arange(bins).unsqueeze(0) - map_idx.unsqueeze(1)).abs().to(probs.dtype)
    bimodal_u = (probs * torch.clamp(dist_from_map, max=float(bimodal_tau))).sum(dim=-1)

    second_peak_mass = _window_mass(probs, peak2_idx, radius=near_radius)
    second_peak_mass = torch.where(has_peak2, second_peak_mass, torch.zeros_like(second_peak_mass))
    second_peak_mass_x_peak_distance = second_peak_mass * peak_separation

    dist = (torch.arange(bins).unsqueeze(0) - map_idx.unsqueeze(1)).abs().to(probs.dtype)
    far_dist = torch.where(dist > far_radius, dist, torch.zeros_like(dist))
    dist_entropy = (probs * far_dist).sum(dim=-1)

    return {
        "entropy_per_slot": entropy.numpy(),
        "top3_mass_per_slot": top3_mass.numpy(),
        "dist_aware_dispersion_per_slot": dist_aware_dispersion.numpy(),
        "far_mass_x_peak_separation_per_slot": far_mass_x_peak_separation.numpy(),
        "bimodal_u_per_slot": bimodal_u.numpy(),
        "dist_entropy_per_slot": dist_entropy.numpy(),
        "second_peak_mass_x_peak_distance_per_slot": (
            second_peak_mass_x_peak_distance.numpy()
        ),
        "entropy_mean": float(entropy.mean().item()),
        "top3_mass_mean": float(top3_mass.mean().item()),
        "dist_aware_dispersion_mean": float(dist_aware_dispersion.mean().item()),
        "far_mass_x_peak_separation_mean": float(far_mass_x_peak_separation.mean().item()),
        "bimodal_u_mean": float(bimodal_u.mean().item()),
        "dist_entropy_mean": float(dist_entropy.mean().item()),
        "second_peak_mass_x_peak_distance_mean": float(
            second_peak_mass_x_peak_distance.mean().item()
        ),
        "logits_7xV": step_logits,
    }


def compute_running_total_variation(series: List[float]) -> float:
    if series is None or len(series) < 2:
        return 0.0
    arr = np.asarray(series, dtype=np.float64)
    return float(np.abs(np.diff(arr)).mean())


def compute_relative_change(series: List[float]) -> float:
    if series is None or len(series) < 2:
        return 0.0
    prev = float(series[-2])
    cur = float(series[-1])
    return (cur - prev) / (abs(prev) + 1e-6)


def compute_windowed_avg(series: List[float], window: int = 5) -> float:
    if series is None or len(series) < window:
        return 0.0
    return float(np.nanmean(np.asarray(series[-window:], dtype=np.float64)))


def compute_windowed_avg_total_variation(series: List[float], window: int = 5) -> float:
    if series is None or len(series) < 2:
        return 0.0
    arr = np.asarray(series[-window:], dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.nanmean(np.abs(np.diff(arr))))


def compute_control_score(control_mode: str, scores: Optional[List[float]], tv_window: int) -> float:
    if scores is None or len(scores) == 0:
        return 0.0
    if control_mode == "metric":
        return float(scores[-1])
    if control_mode == "metric_window":
        return compute_windowed_avg(scores, window=tv_window)
    if control_mode == "metric_variance":
        return compute_relative_change(scores)
    if control_mode == "metric_total_variation":
        return compute_running_total_variation(scores)
    if control_mode == "metric_window_total_variation":
        return compute_windowed_avg_total_variation(scores, window=tv_window)
    return float(scores[-1])


def _summary_contains_episode(summary_path: Path, task_id: int, trial_id: int) -> bool:
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


def episode_already_exists(
    task_suite_name: str,
    run_name: str,
    task_description_slug: str,
    task_id: int,
    trial_id: int,
    extra_rollout_roots=None,
):
    rollout_roots = [Path(f"./rollouts/{task_suite_name}")]
    if extra_rollout_roots:
        rollout_roots.extend(Path(root) for root in extra_rollout_roots)

    for root in rollout_roots:
        run_dir = root / run_name
        episode_dir = run_dir / task_description_slug / f"trial{trial_id:02d}"
        pt_path = episode_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
        if pt_path.exists():
            return True, str(pt_path)

        summary_path = run_dir / "episode_summary.jsonl"
        if _summary_contains_episode(summary_path, task_id=task_id, trial_id=trial_id):
            return True, str(summary_path)

    return False, None


def parse_int_filter(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@dataclass
class WorkerOutput:
    inference_time: float
    action: np.ndarray
    generated_ids: torch.Tensor
    action_scores: List[torch.Tensor]
    decoded_text: str = ""


def register_openvla_auto_classes() -> None:
    registrations = (
        (AutoConfig.register, ("openvla", OpenVLAConfig)),
        (AutoImageProcessor.register, (OpenVLAConfig, PrismaticImageProcessor)),
        (AutoProcessor.register, (OpenVLAConfig, PrismaticProcessor)),
        (AutoModelForVision2Seq.register, (OpenVLAConfig, OpenVLAForActionPrediction)),
    )
    for register_fn, args in registrations:
        try:
            register_fn(*args)
        except ValueError:
            pass


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_vla_model(checkpoint: str, dataset: str, device: torch.device, dtype: torch.dtype):
    register_openvla_auto_classes()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if not hasattr(model, "norm_stats") or model.norm_stats is None:
        model.norm_stats = {}
    model.norm_stats[dataset] = UNNORM_KEYS[f"{dataset}_no_noops"]
    model.norm_stats.update(UNNORM_KEYS)
    return model, processor


def validate_local_checkpoint(checkpoint: str) -> None:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return

    required_files = ["config.json", "model.safetensors.index.json", "preprocessor_config.json"]
    missing_required = [name for name in required_files if not (checkpoint_path / name).exists()]
    if missing_required:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} is missing required files: {missing_required}"
        )

    with (checkpoint_path / "model.safetensors.index.json").open("r", encoding="utf-8") as f:
        index = json.load(f)
    expected_shards = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [name for name in expected_shards if not (checkpoint_path / name).exists()]
    if missing_shards:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} is missing model shards: {missing_shards}"
        )


def prepare_image(obs, center_crop: bool) -> Image.Image:
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    if not center_crop:
        return image

    crop_scale = 0.9
    crop_ratio = float(np.sqrt(crop_scale))
    width, height = image.size
    crop_width = max(1, int(round(width * crop_ratio)))
    crop_height = max(1, int(round(height * crop_ratio)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((224, 224), Image.BILINEAR).convert("RGB")


def build_openvla_prompt(checkpoint: str, task_label: str) -> str:
    if "openvla-v01" in checkpoint:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def build_ecot_prompt(task_label: str, prefix_text: str) -> str:
    return (
        f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
        f"{task_label.lower()}? ASSISTANT:{prefix_text}"
    )


def ensure_generation_prompt_token(inputs):
    input_ids = inputs["input_ids"]
    if torch.all(input_ids[:, -1] == 29871):
        return inputs

    inputs["input_ids"] = torch.cat(
        (
            input_ids,
            torch.full(
                (input_ids.shape[0], 1),
                29871,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
        ),
        dim=1,
    )
    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
    return inputs


def append_generation_prompt_token(inputs):
    input_ids = inputs["input_ids"]
    inputs["input_ids"] = torch.cat(
        (
            input_ids,
            torch.full(
                (input_ids.shape[0], 1),
                29871,
                dtype=input_ids.dtype,
                device=input_ids.device,
            ),
        ),
        dim=1,
    )
    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
    return inputs


def slice_final_action_tokens(generated_sequence, scores, action_dim: int, eos_token_id):
    ends_with_eos = (
        eos_token_id is not None
        and generated_sequence.numel() >= action_dim + 1
        and int(generated_sequence[-1].item()) == int(eos_token_id)
    )
    action_token_slice = slice(-(action_dim + 1), -1) if ends_with_eos else slice(-action_dim, None)
    action_token_ids = generated_sequence[action_token_slice]
    action_scores = list(scores[action_token_slice])
    if len(action_token_ids) != action_dim or len(action_scores) != action_dim:
        raise RuntimeError(
            f"Expected {action_dim} action tokens/scores, got "
            f"{len(action_token_ids)} tokens and {len(action_scores)} scores."
        )
    return action_token_ids, action_scores


def decode_action_from_token_ids(model, action_token_ids: torch.Tensor, unnorm_key: str) -> np.ndarray:
    predicted_action_token_ids = action_token_ids.detach().cpu().numpy()
    discretized_actions = int(model.vocab_size) - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1,
        a_min=0,
        a_max=model.bin_centers.shape[0] - 1,
    )
    normalized_actions = model.bin_centers[discretized_actions]

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return action.astype(np.float32)


class OpenVLAWorker:
    def __init__(
        self,
        checkpoint: str,
        dataset: str,
        device: str,
        dtype_name: str = "auto",
        center_crop: bool = True,
    ):
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.unnorm_key = dataset
        self.device = torch.device(device)
        self.dtype = resolve_torch_dtype(dtype_name, self.device)
        self.center_crop = center_crop
        print(f"Loading OpenVLA on {self.device}: {checkpoint}")
        self.model, self.processor = load_vla_model(checkpoint, dataset, self.device, self.dtype)

    @torch.inference_mode()
    def infer(self, observation: Dict, task_label: str) -> WorkerOutput:
        image = prepare_image(observation, self.center_crop)
        prompt = build_openvla_prompt(self.checkpoint, task_label)
        input_dtype = next(self.model.parameters()).dtype
        inputs = self.processor(prompt, image).to(self.device, dtype=input_dtype)
        inputs = ensure_generation_prompt_token(inputs)

        action_dim = self.model.get_action_dim(self.unnorm_key)
        start_time = time.perf_counter()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=action_dim + 1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            use_cache=True,
        )
        inference_time = time.perf_counter() - start_time

        action_token_ids, action_scores = slice_final_action_tokens(
            generated_sequence=generated.sequences[0],
            scores=generated.scores,
            action_dim=action_dim,
            eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
        )
        action = decode_action_from_token_ids(self.model, action_token_ids, self.unnorm_key)
        return WorkerOutput(
            inference_time=inference_time,
            action=action,
            generated_ids=generated.sequences,
            action_scores=action_scores,
        )


class ECoTWorker:
    def __init__(
        self,
        checkpoint: str,
        dataset: str,
        device: str,
        dtype_name: str = "auto",
        center_crop: bool = True,
        max_new_tokens: int = 1024,
        prefix_text: str = " TASK:",
    ):
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.unnorm_key = dataset
        self.device = torch.device(device)
        self.dtype = resolve_torch_dtype(dtype_name, self.device)
        self.center_crop = center_crop
        self.max_new_tokens = max_new_tokens
        self.prefix_text = prefix_text
        print(f"Loading ECoT on {self.device}: {checkpoint}")
        self.model, self.processor = load_vla_model(checkpoint, dataset, self.device, self.dtype)

    @torch.inference_mode()
    def infer(self, observation: Dict, task_label: str) -> WorkerOutput:
        image = prepare_image(observation, self.center_crop)
        prompt = build_ecot_prompt(task_label, self.prefix_text)
        input_dtype = next(self.model.parameters()).dtype
        inputs = self.processor(prompt, image).to(self.device, dtype=input_dtype)
        inputs = append_generation_prompt_token(inputs)
        input_ids = inputs.pop("input_ids")

        action_dim = self.model.get_action_dim(self.unnorm_key)
        start_time = time.perf_counter()
        generated = self.model.generate(
            input_ids=input_ids,
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        inference_time = time.perf_counter() - start_time

        action_token_ids, action_scores = slice_final_action_tokens(
            generated_sequence=generated.sequences[0],
            scores=generated.scores,
            action_dim=action_dim,
            eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
        )
        action = decode_action_from_token_ids(self.model, action_token_ids, self.unnorm_key)
        decoded_text = self.processor.batch_decode(generated.sequences)[0]
        return WorkerOutput(
            inference_time=inference_time,
            action=action,
            generated_ids=generated.sequences,
            action_scores=action_scores,
            decoded_text=decoded_text,
        )


def threshold_direction_to_uncertain(
    score_threshold_direction: str,
    uncertainty_metric_name: str,
):
    if score_threshold_direction not in {"auto", "gt", "lt"}:
        raise ValueError("score_threshold_direction must be one of: auto, gt, lt")
    if score_threshold_direction == "gt":
        return True
    if score_threshold_direction == "lt":
        return False
    return uncertainty_metric_name != "top3_mass"


def build_dualgpu_run_name(
    router_control_mode: str,
    uncertainty_metric_name: str,
    threshold_suffix: str,
    tv_window: int,
    fixed_ecot_interval: int = 5,
) -> str:
    if router_control_mode == "fixed_interval":
        return f"dualgpu_fixed_interval_{fixed_ecot_interval}"
    if "window" in router_control_mode:
        return f"dualgpu_{uncertainty_metric_name}_{router_control_mode}_{threshold_suffix}_w{tv_window}"
    return f"dualgpu_{uncertainty_metric_name}_{router_control_mode}_{threshold_suffix}"


def build_episode_save_dir(
    task_suite_name: str,
    run_name: str,
    task_description_slug: str,
    episode_idx: int,
) -> str:
    return (
        f"./rollouts/{task_suite_name}/{run_name}/"
        f"{task_description_slug}/trial{episode_idx:02d}"
    )


def save_dualgpu_rollout_pt(
    save_dir,
    task_name,
    task_id,
    trial_id,
    success,
    step_ids,
    openvla_logits,
    selected_metric_series,
    control_score_series,
    uncertain_decision_series,
    used_ecot_series,
    openvla_inference_times,
    ecot_inference_times,
    selected_inference_times,
    openvla_actions,
    ecot_actions,
    executed_actions,
    selected_policy_series,
    ecot_reasoning_series,
    router_config,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _to_tensor_or_empty(values):
        if values is None or len(values) == 0:
            return torch.empty(0)
        return torch.tensor(values, dtype=torch.float32)

    def _to_array_or_empty(values):
        if values is None or len(values) == 0:
            return np.empty((0,), dtype=np.float32)
        return np.asarray(values, dtype=np.float32)

    payload = {
        "task_name": str(task_name),
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "success": int(success),
        "step_ids": np.asarray(step_ids, dtype=np.int32),
        "openvla_logits": (
            torch.stack(openvla_logits, dim=0).float() if len(openvla_logits) else torch.empty(0)
        ),
        "openvla_metric_series": _to_tensor_or_empty(selected_metric_series),
        "selected_metric_series": _to_tensor_or_empty(selected_metric_series),
        "control_score_series": _to_tensor_or_empty(control_score_series),
        "uncertain_decision_series": _to_tensor_or_empty(uncertain_decision_series),
        "used_ecot_series": _to_tensor_or_empty(used_ecot_series),
        "openvla_inference_times": _to_array_or_empty(openvla_inference_times),
        "ecot_inference_times": _to_array_or_empty(ecot_inference_times),
        "selected_inference_times": _to_array_or_empty(selected_inference_times),
        "openvla_actions": _to_array_or_empty(openvla_actions),
        "ecot_actions": _to_array_or_empty(ecot_actions),
        "executed_actions": _to_array_or_empty(executed_actions),
        "selected_policy_series": list(selected_policy_series),
        "ecot_reasoning": list(ecot_reasoning_series),
        "router_config": dict(router_config),
    }

    pt_path = save_dir / f"task{task_id:02d}_trial{trial_id:02d}.pt"
    torch.save(payload, pt_path)
    return pt_path


def append_dualgpu_episode_summary_jsonl(
    save_dir,
    task_suite_name,
    task_name,
    task_id,
    trial_id,
    success,
    step_ids,
    selected_metric_series,
    control_score_series,
    used_ecot_series,
    openvla_inference_times,
    ecot_inference_times,
    selected_inference_times,
    router_config,
):
    save_dir = Path(save_dir)
    run_dir = save_dir.parents[1] if len(save_dir.parents) >= 2 else save_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    def _mean_or_none(values):
        if values is None or len(values) == 0:
            return None
        return float(np.mean(values))

    row = {
        "task_suite": str(task_suite_name),
        "task_name": str(task_name),
        "task_id": int(task_id),
        "trial_id": int(trial_id),
        "success": int(success),
        "num_steps": int(len(step_ids)),
        "ecot_ratio": _mean_or_none(used_ecot_series),
        "num_ecot_steps": int(np.sum(used_ecot_series)) if used_ecot_series else 0,
        "mean_openvla_metric": _mean_or_none(selected_metric_series),
        "mean_selected_metric": _mean_or_none(selected_metric_series),
        "mean_control_score": _mean_or_none(control_score_series),
        "mean_openvla_inference_time": _mean_or_none(openvla_inference_times),
        "mean_ecot_inference_time": _mean_or_none(ecot_inference_times),
        "mean_selected_inference_time": _mean_or_none(selected_inference_times),
        "metric": router_config["uncertainty_metric_name"],
        "router_control_mode": router_config["router_control_mode"],
        "score_threshold": float(router_config["score_threshold"]),
        "score_threshold_direction": router_config["score_threshold_direction"],
        "tv_window": int(router_config["tv_window"]),
        "fixed_ecot_interval": int(router_config.get("fixed_ecot_interval", 0)),
        "openvla_checkpoint": router_config["openvla_checkpoint"],
        "ecot_checkpoint": router_config["ecot_checkpoint"],
        "openvla_device": router_config["openvla_device"],
        "ecot_device": router_config["ecot_device"],
        "save_dir": str(save_dir),
    }

    summary_path = run_dir / "episode_summary.jsonl"
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return summary_path


def eval_libero_dualgpu(
    dataset: str = "libero_goal",
    num_trials_per_task: int = 20,
    openvla_checkpoint: Optional[str] = None,
    ecot_checkpoint: Optional[str] = None,
    openvla_device: str = "cuda:0",
    ecot_device: str = "cuda:1",
    torch_dtype: str = "auto",
    router_control_mode: str = "metric_window_total_variation",
    uncertainty_metric_name: str = "far_mass_x_peak_separation",
    score_threshold: float = 0.0,
    score_threshold_direction: str = "auto",
    tv_window: int = 5,
    fixed_ecot_interval: int = 5,
    ecot_prefix_text: str = " TASK:",
    ecot_max_new_tokens: int = 1024,
    seed: int = 42,
    save_video: bool = False,
    save_reasoning: bool = False,
    save_uncertainty_video: bool = False,
    uncertainty_video_fps: int = 10,
    save_rollout_pt_file: bool = True,
    skip_existing_rollouts: bool = False,
    skip_existing_rollout_roots=None,
    task_ids: Optional[List[int]] = None,
    trial_ids: Optional[List[int]] = None,
    max_eval_steps: Optional[int] = None,
):
    if dataset not in DEFAULT_OPENVLA_CKPTS:
        raise ValueError(f"Unsupported dataset for default ckpt lookup: {dataset}")
    if router_control_mode == "fixed_interval" and fixed_ecot_interval <= 0:
        raise ValueError("fixed_ecot_interval must be positive when using fixed_interval")

    set_seed_everywhere(seed)
    openvla_checkpoint = openvla_checkpoint or DEFAULT_OPENVLA_CKPTS[dataset]
    ecot_checkpoint = ecot_checkpoint or DEFAULT_ECOT_CKPTS[dataset]
    validate_local_checkpoint(openvla_checkpoint)
    validate_local_checkpoint(ecot_checkpoint)
    high_score_means_uncertain = threshold_direction_to_uncertain(
        score_threshold_direction,
        uncertainty_metric_name,
    )
    threshold_operator = ">" if high_score_means_uncertain else "<"
    threshold_suffix = (
        f"{score_threshold}"
        if score_threshold_direction == "auto"
        else f"{score_threshold}_{score_threshold_direction}"
    )
    run_name = build_dualgpu_run_name(
        router_control_mode=router_control_mode,
        uncertainty_metric_name=uncertainty_metric_name,
        threshold_suffix=threshold_suffix,
        tv_window=tv_window,
        fixed_ecot_interval=fixed_ecot_interval,
    )

    print(f"Seed: {seed}")
    print(f"Task suite: {dataset}")
    print(f"Run name: {run_name}")
    print(f"OpenVLA device: {openvla_device}")
    print(f"ECoT device: {ecot_device}")
    print(f"OpenVLA checkpoint: {openvla_checkpoint}")
    print(f"ECoT checkpoint: {ecot_checkpoint}")
    print(
        f"Router: metric={uncertainty_metric_name}, mode={router_control_mode}, "
        f"threshold={score_threshold}, uncertain_when=score {threshold_operator} threshold"
    )
    if router_control_mode == "fixed_interval":
        print(f"Fixed ECoT interval: every {fixed_ecot_interval} control steps")
    if task_ids is not None:
        print(f"Task filter: {task_ids}")
    if trial_ids is not None:
        print(f"Trial filter: {trial_ids}")
    if max_eval_steps is not None:
        print(f"Max eval steps override: {max_eval_steps}")

    openvla_worker = OpenVLAWorker(
        checkpoint=openvla_checkpoint,
        dataset=dataset,
        device=openvla_device,
        dtype_name=torch_dtype,
        center_crop=True,
    )
    ecot_worker = ECoTWorker(
        checkpoint=ecot_checkpoint,
        dataset=dataset,
        device=ecot_device,
        dtype_name=torch_dtype,
        center_crop=True,
        max_new_tokens=ecot_max_new_tokens,
        prefix_text=ecot_prefix_text,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[dataset]()
    num_tasks_in_suite = task_suite.n_tasks
    num_steps_wait = 10

    task_id_set = set(task_ids) if task_ids is not None else None
    trial_id_set = set(trial_ids) if trial_ids is not None else None
    task_iter = [
        task_id
        for task_id in range(num_tasks_in_suite)
        if task_id_set is None or task_id in task_id_set
    ]

    router_config = {
        "openvla_checkpoint": str(openvla_checkpoint),
        "ecot_checkpoint": str(ecot_checkpoint),
        "openvla_device": str(openvla_device),
        "ecot_device": str(ecot_device),
        "torch_dtype": str(torch_dtype),
        "router_control_mode": str(router_control_mode),
        "uncertainty_metric_name": str(uncertainty_metric_name),
        "score_threshold": float(score_threshold),
        "score_threshold_direction": str(score_threshold_direction),
        "high_score_means_uncertain": bool(high_score_means_uncertain),
        "tv_window": int(tv_window),
        "fixed_ecot_interval": int(fixed_ecot_interval),
        "ecot_prefix_text": str(ecot_prefix_text),
        "ecot_max_new_tokens": int(ecot_max_new_tokens),
    }

    total_episodes, total_successes = 0, 0
    for task_id in tqdm(task_iter, desc="Tasks", unit="task"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        task_description_slug = task_description.replace(" ", "_")

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(range(num_trials_per_task), desc="Trials", leave=False):
            if trial_id_set is not None and episode_idx not in trial_id_set:
                continue

            save_dir = build_episode_save_dir(
                task_suite_name=dataset,
                run_name=run_name,
                task_description_slug=task_description_slug,
                episode_idx=episode_idx,
            )
            print(f"\nTask: {task_description}")

            if skip_existing_rollouts:
                exists, existing_path = episode_already_exists(
                    task_suite_name=dataset,
                    run_name=run_name,
                    task_description_slug=task_description_slug,
                    task_id=task_id,
                    trial_id=episode_idx,
                    extra_rollout_roots=skip_existing_rollout_roots,
                )
                if exists:
                    print(
                        f"Skipping existing episode task={task_id} trial={episode_idx}: "
                        f"{existing_path}"
                    )
                    continue

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            if dataset == "libero_spatial":
                max_steps = 200
            elif dataset == "libero_object":
                max_steps = 280
            elif dataset == "libero_goal":
                max_steps = 300
            elif dataset == "libero_10":
                max_steps = 520
            elif dataset == "libero_90":
                max_steps = 400
            else:
                max_steps = 200
            if max_eval_steps is not None:
                max_steps = min(max_steps, max_eval_steps)

            replay_images = []
            replay_reasoning = []
            step_ids = []
            openvla_logits = []

            selected_metric_series = []
            control_score_series = []
            uncertain_decision_series = []
            used_ecot_series = []
            openvla_inference_times = []
            ecot_inference_times = []
            selected_inference_times = []
            openvla_actions = []
            ecot_actions = []
            executed_actions = []
            selected_policy_series = []
            ecot_reasoning_series = []

            t = 0
            success = 0
            done = False

            while t < max_steps + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue

                    img = get_libero_image(obs, 224)
                    if save_video or save_reasoning or save_uncertainty_video:
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

                    openvla_out = openvla_worker.infer(observation, task_description)
                    openvla_inference_times.append(openvla_out.inference_time)

                    action_uncertainty = compute_step_uncertainty_from_action_scores(
                        openvla_out.action_scores
                    )
                    if save_rollout_pt_file:
                        openvla_logits.append(action_uncertainty["logits_7xV"])

                    metric_key = f"{uncertainty_metric_name}_mean"
                    selected_metric_score = float(action_uncertainty[metric_key])
                    selected_metric_series.append(selected_metric_score)
                    if router_control_mode == "fixed_interval":
                        control_step_idx = len(step_ids)
                        is_uncertain = control_step_idx % fixed_ecot_interval == 0
                        control_score = float(is_uncertain)
                    else:
                        control_score = compute_control_score(
                            router_control_mode,
                            selected_metric_series,
                            tv_window,
                        )
                        if high_score_means_uncertain:
                            is_uncertain = control_score > score_threshold
                        else:
                            is_uncertain = control_score < score_threshold
                    control_score_series.append(control_score)
                    uncertain_decision_series.append(float(is_uncertain))

                    ecot_out = None
                    if is_uncertain:
                        ecot_out = ecot_worker.infer(observation, task_description)
                        action = ecot_out.action.copy()
                        selected_policy = "ecot"
                        selected_time = openvla_out.inference_time + ecot_out.inference_time
                        ecot_inference_times.append(ecot_out.inference_time)
                        reasoning_text = (
                            "##S##"
                            + ecot_prefix_text
                            + "##E##"
                            + ecot_out.decoded_text
                        )
                        ecot_reasoning = ecot_out.decoded_text
                        if router_control_mode == "fixed_interval":
                            print(
                                f"[router] fixed_interval step={len(step_ids)} "
                                f"every={fixed_ecot_interval} -> ECoT"
                            )
                        else:
                            print(
                                f"[router] uncertain score={control_score:.6f} "
                                f"{threshold_operator} {score_threshold:.6f} -> ECoT"
                            )
                    else:
                        action = openvla_out.action.copy()
                        selected_policy = "openvla"
                        selected_time = openvla_out.inference_time
                        ecot_inference_times.append(0.0)
                        reasoning_text = "OpenVLA direct action"
                        ecot_reasoning = ""
                        if router_control_mode == "fixed_interval":
                            print(
                                f"[router] fixed_interval step={len(step_ids)} "
                                f"every={fixed_ecot_interval} -> OpenVLA"
                            )
                        else:
                            print(
                                f"[router] certain score={control_score:.6f}, "
                                f"threshold={score_threshold:.6f} -> OpenVLA"
                            )

                    step_ids.append(t)
                    used_ecot_series.append(float(is_uncertain))
                    selected_inference_times.append(selected_time)
                    selected_policy_series.append(selected_policy)
                    ecot_reasoning_series.append(ecot_reasoning)
                    openvla_actions.append(openvla_out.action.copy())
                    if ecot_out is None:
                        ecot_actions.append(np.full_like(openvla_out.action, np.nan))
                    else:
                        ecot_actions.append(ecot_out.action.copy())
                    if save_reasoning or save_uncertainty_video:
                        replay_reasoning.append(reasoning_text)

                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    executed_actions.append(action.copy())

                    print(f"Step: {t}")
                    print(f"OpenVLA inference time: {openvla_out.inference_time:.4f} seconds")
                    if ecot_out is not None:
                        print(f"ECoT inference time: {ecot_out.inference_time:.4f} seconds")
                    print(f"Selected policy: {selected_policy}")
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

            append_dualgpu_episode_summary_jsonl(
                save_dir=save_dir,
                task_suite_name=dataset,
                task_name=task_description_slug,
                task_id=task_id,
                trial_id=episode_idx,
                success=success,
                step_ids=step_ids,
                selected_metric_series=selected_metric_series,
                control_score_series=control_score_series,
                used_ecot_series=used_ecot_series,
                openvla_inference_times=openvla_inference_times,
                ecot_inference_times=[
                    x for x, used in zip(ecot_inference_times, used_ecot_series) if used
                ],
                selected_inference_times=selected_inference_times,
                router_config=router_config,
            )

            if save_rollout_pt_file:
                save_dualgpu_rollout_pt(
                    save_dir=save_dir,
                    task_name=task_description_slug,
                    task_id=task_id,
                    trial_id=episode_idx,
                    success=success,
                    step_ids=step_ids,
                    openvla_logits=openvla_logits,
                    selected_metric_series=selected_metric_series,
                    control_score_series=control_score_series,
                    uncertain_decision_series=uncertain_decision_series,
                    used_ecot_series=used_ecot_series,
                    openvla_inference_times=openvla_inference_times,
                    ecot_inference_times=ecot_inference_times,
                    selected_inference_times=selected_inference_times,
                    openvla_actions=openvla_actions,
                    ecot_actions=ecot_actions,
                    executed_actions=executed_actions,
                    selected_policy_series=selected_policy_series,
                    ecot_reasoning_series=ecot_reasoning_series,
                    router_config=router_config,
                )

            task_episodes += 1
            total_episodes += 1

            if save_video:
                save_rollout_video(
                    replay_images,
                    episode_idx,
                    success=done,
                    task_description=task_description,
                    save_dir=save_dir,
                )
            if save_uncertainty_video:
                if save_uncertainty_rollout_video is None:
                    raise ImportError(
                        "--save-uncertainty-video requires experiments/libero/uncertainty_video.py "
                        "and its optional video dependencies, including imageio and Pillow."
                    )
                save_uncertainty_rollout_video(
                    rollout_images=replay_images,
                    step_ids=step_ids,
                    selected_metric_series=control_score_series,
                    cot_used_series=used_ecot_series,
                    uncertain_decision_series=uncertain_decision_series,
                    save_dir=save_dir,
                    episode_idx=episode_idx,
                    success=done,
                    task_description=task_description,
                    metric_name=f"{uncertainty_metric_name} {router_control_mode}",
                    threshold=score_threshold,
                    high_score_means_uncertain=high_score_means_uncertain,
                    fps=uncertainty_video_fps,
                    rollout_reasoning=replay_reasoning,
                )
            if save_reasoning:
                save_rollot_reasoning(
                    replay_reasoning,
                    replay_images,
                    episode_idx,
                    success=done,
                    task_description=task_description,
                    save_dir=save_dir,
                )

            print(
                f"Episode task={task_id} trial={episode_idx} success={success} "
                f"ecot_ratio={np.mean(used_ecot_series) if used_ecot_series else 0.0:.3f}"
            )

        print(
            f"Task {task_id} done: successes={task_successes}/{task_episodes} "
            f"running_total={total_successes}/{total_episodes}"
        )


def main():
    parser = argparse.ArgumentParser(description="LIBERO dual-GPU OpenVLA/ECoT router evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="libero_goal",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    parser.add_argument("--openvla-checkpoint", type=str, default=None)
    parser.add_argument("--ecot-checkpoint", type=str, default=None)
    parser.add_argument("--openvla-device", type=str, default="cuda:0")
    parser.add_argument("--ecot-device", type=str, default="cuda:1")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--router-control-mode",
        type=str,
        default="metric_window_total_variation",
        choices=[
            "metric",
            "metric_window",
            "metric_variance",
            "metric_total_variation",
            "metric_window_total_variation",
            "fixed_interval",
        ],
    )
    parser.add_argument(
        "--uncertainty-metric-name",
        type=str,
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
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--score-threshold-direction",
        type=str,
        default="auto",
        choices=["auto", "gt", "lt"],
    )
    parser.add_argument("--tv-window", type=int, default=5)
    parser.add_argument("--fixed-ecot-interval", type=int, default=5)
    parser.add_argument("--ecot-prefix-text", type=str, default=" TASK:")
    parser.add_argument("--ecot-max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-reasoning", action="store_true")
    parser.add_argument("--save-uncertainty-video", action="store_true")
    parser.add_argument("--uncertainty-video-fps", type=int, default=10)
    parser.add_argument(
        "--save-rollout-pt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-existing-rollouts", action="store_true")
    parser.add_argument("--skip-existing-rollout-root", action="append", default=[])
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--trial-ids", type=str, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    args = parser.parse_args()

    eval_libero_dualgpu(
        dataset=args.dataset,
        num_trials_per_task=args.num_trials_per_task,
        openvla_checkpoint=args.openvla_checkpoint,
        ecot_checkpoint=args.ecot_checkpoint,
        openvla_device=args.openvla_device,
        ecot_device=args.ecot_device,
        torch_dtype=args.torch_dtype,
        router_control_mode=args.router_control_mode,
        uncertainty_metric_name=args.uncertainty_metric_name,
        score_threshold=args.score_threshold,
        score_threshold_direction=args.score_threshold_direction,
        tv_window=args.tv_window,
        fixed_ecot_interval=args.fixed_ecot_interval,
        ecot_prefix_text=args.ecot_prefix_text,
        ecot_max_new_tokens=args.ecot_max_new_tokens,
        seed=args.seed,
        save_video=args.save_video,
        save_reasoning=args.save_reasoning,
        save_uncertainty_video=args.save_uncertainty_video,
        uncertainty_video_fps=args.uncertainty_video_fps,
        save_rollout_pt_file=args.save_rollout_pt,
        skip_existing_rollouts=args.skip_existing_rollouts,
        skip_existing_rollout_roots=args.skip_existing_rollout_root,
        task_ids=parse_int_filter(args.task_ids),
        trial_ids=parse_int_filter(args.trial_ids),
        max_eval_steps=args.max_eval_steps,
    )


if __name__ == "__main__":
    main()
