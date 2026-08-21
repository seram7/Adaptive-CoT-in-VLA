"""Action-token uncertainty metrics and causal CoT routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


TEMPORAL_MODE_TO_METRIC = {
    "step_mode_farmass_ablation": "step_mode_farmass",
    "overlap_proxy_farmass": "overlap_proxy_farmass",
    "overlap_continuous_drift": "overlap_continuous_drift",
}
THRESHOLD_MODES = {"entropy_wtv", "farmass_wtv", *TEMPORAL_MODE_TO_METRIC}


def _smooth_probs(probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return probs
    if kernel_size % 2 == 0:
        raise ValueError("smooth_kernel must be odd")
    pad = kernel_size // 2
    weight = torch.ones(1, 1, kernel_size, dtype=probs.dtype, device=probs.device)
    weight /= float(kernel_size)
    values = F.pad(probs.unsqueeze(1), (pad, pad), mode="replicate")
    return F.conv1d(values, weight).squeeze(1)


def _local_maxima(values: torch.Tensor) -> torch.Tensor:
    left = torch.empty_like(values)
    right = torch.empty_like(values)
    left[:, 0] = -torch.inf
    left[:, 1:] = values[:, :-1]
    right[:, -1] = -torch.inf
    right[:, :-1] = values[:, 1:]
    return (values >= left) & (values >= right)


def compute_uncertainty(
    action_logits: torch.Tensor,
    *,
    smooth_kernel: int = 5,
    far_radius: int = 20,
    min_peak_probability: float = 0.01,
) -> dict[str, object]:
    """Compute the LIBERO-compatible entropy and FarMass metrics.

    Args:
        action_logits: Tensor shaped ``[slots, 256]`` or ``[1, slots, 256]``.
            A slot is one scalar action dimension at one action-horizon index.

    The FarMass metric is the mean, over slots, of probability mass farther
    than ``far_radius`` bins from the smoothed MAP bin multiplied by the
    distance between the two strongest local probability peaks.
    """

    logits = torch.as_tensor(action_logits).detach().float().cpu()
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {tuple(logits.shape)}")
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[-1] != 256:
        raise ValueError(f"Expected action logits [slots, 256], got {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    smoothed = _smooth_probs(probs, smooth_kernel)
    map_index = smoothed.argmax(dim=-1)
    bin_index = torch.arange(probs.shape[-1]).unsqueeze(0)
    distance_from_map = (bin_index - map_index.unsqueeze(1)).abs()
    far_mass = (probs * (distance_from_map > far_radius)).sum(dim=-1)

    peak_values = torch.where(
        _local_maxima(smoothed) & (smoothed >= min_peak_probability),
        smoothed,
        torch.full_like(smoothed, -torch.inf),
    )
    top_values, top_indices = torch.topk(peak_values, k=2, dim=-1)
    has_second_peak = torch.isfinite(top_values[:, 1])
    peak_separation = (top_indices[:, 0] - top_indices[:, 1]).abs().to(probs.dtype)
    peak_separation = torch.where(
        has_second_peak, peak_separation, torch.zeros_like(peak_separation)
    )
    farmass = far_mass * peak_separation

    return {
        "entropy_mean": float(entropy.mean().item()),
        "farmass_mean": float(farmass.mean().item()),
        "entropy_per_slot": entropy.numpy(),
        "far_mass_per_slot": far_mass.numpy(),
        "peak_separation_per_slot": peak_separation.numpy(),
        "farmass_per_slot": farmass.numpy(),
    }


def rebin_proxy_logits(
    action_logits: torch.Tensor,
    *,
    horizon: int,
    slots_per_step: int,
    bins: int = 100,
) -> np.ndarray:
    """Convert 256 ordered action-token proxy logits to probability bins.

    OpenVLA's token order is reversed relative to normalized action value, so
    probabilities are flipped before mass-preserving rebinning.  The returned
    bins are equal-width normalized-action bins, not empirical data quantiles.
    """

    logits = torch.as_tensor(action_logits).detach().float().cpu()
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"Expected batch size one, got {tuple(logits.shape)}")
        logits = logits[0]
    expected_slots = int(horizon) * int(slots_per_step)
    if logits.shape != (expected_slots, 256):
        raise ValueError(
            f"Expected proxy logits {(expected_slots, 256)}, got {tuple(logits.shape)}"
        )
    if bins < 2:
        raise ValueError("bins must be at least two")

    probs = torch.softmax(logits, dim=-1).flip(-1)
    target = torch.div(torch.arange(256) * bins, 256, rounding_mode="floor")
    target = target.clamp_max(bins - 1).unsqueeze(0).expand(expected_slots, -1)
    rebinned = torch.zeros(expected_slots, bins, dtype=probs.dtype)
    rebinned.scatter_add_(-1, target, probs)
    return rebinned.reshape(horizon, slots_per_step, bins).numpy()


def _outside_central_mass(
    reference: np.ndarray,
    candidate: np.ndarray,
    credible_mass: float,
) -> np.ndarray:
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be in (0, 1)")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must have matching [items, bins] shapes")
    tail = (1.0 - credible_mass) / 2.0
    cdf = np.cumsum(reference, axis=-1)
    low = np.argmax(cdf >= tail, axis=-1)
    high = np.argmax(cdf >= 1.0 - tail, axis=-1)
    indices = np.arange(reference.shape[-1])[None, :]
    outside = (indices < low[:, None]) | (indices > high[:, None])
    return np.sum(candidate * outside, axis=-1)


def temporal_mode_farmass(
    proxy_probs: np.ndarray,
    *,
    credible_mass: float = 0.9,
) -> dict[str, float]:
    """Metric 1: compare h=0 and h=1 proxy modes within one chunk."""

    # Copy before normalizing: callers retain this chunk for the next overlap
    # query, so mutating it here would change the future metric input.
    probs = np.array(proxy_probs, dtype=np.float64, copy=True)
    if probs.ndim != 3 or probs.shape[0] < 2:
        raise ValueError("proxy_probs must have shape [horizon>=2, slots, bins]")
    first, second = probs[0], probs[1]
    first /= np.sum(first, axis=-1, keepdims=True)
    second /= np.sum(second, axis=-1, keepdims=True)
    mode_distance = np.abs(np.argmax(second, axis=-1) - np.argmax(first, axis=-1))
    mode_distance = mode_distance / float(probs.shape[-1] - 1)
    outside_mass = _outside_central_mass(first, second, credible_mass)
    score = mode_distance * outside_mass
    return {
        "step_mode_farmass": float(np.mean(score)),
        "step_mode_distance": float(np.mean(mode_distance)),
        "step_outside_mass": float(np.mean(outside_mass)),
    }


def overlap_proxy_farmass(
    previous_proxy_probs: np.ndarray,
    current_proxy_probs: np.ndarray,
    *,
    executed_steps: int,
    credible_mass: float = 0.9,
) -> dict[str, float]:
    """Metric 2: distribution disagreement for aligned future timesteps."""

    # These arrays are cached across policy queries.  Keep metric evaluation
    # side-effect free while normalizing the aligned distributions.
    previous = np.array(previous_proxy_probs, dtype=np.float64, copy=True)
    current = np.array(current_proxy_probs, dtype=np.float64, copy=True)
    if previous.ndim != 3 or current.ndim != 3 or previous.shape[1:] != current.shape[1:]:
        raise ValueError("proxy probabilities must have compatible [horizon, slots, bins] shapes")
    overlap = min(previous.shape[0] - executed_steps, current.shape[0])
    if overlap <= 0:
        raise ValueError("No prediction-horizon overlap")
    reference = previous[executed_steps : executed_steps + overlap].reshape(-1, previous.shape[-1])
    candidate = current[:overlap].reshape(-1, current.shape[-1])
    reference /= np.sum(reference, axis=-1, keepdims=True)
    candidate /= np.sum(candidate, axis=-1, keepdims=True)
    wasserstein = np.sum(
        np.abs(np.cumsum(reference, axis=-1) - np.cumsum(candidate, axis=-1)),
        axis=-1,
    ) / float(previous.shape[-1] - 1)
    outside_mass = _outside_central_mass(reference, candidate, credible_mass)
    score = wasserstein * outside_mass
    return {
        "overlap_proxy_farmass": float(np.mean(score)),
        "overlap_proxy_wasserstein": float(np.mean(wasserstein)),
        "overlap_proxy_outside_mass": float(np.mean(outside_mass)),
        "overlap_steps": int(overlap),
    }


def overlap_continuous_drift(
    previous_actions: np.ndarray,
    current_actions: np.ndarray,
    *,
    executed_steps: int,
) -> dict[str, float]:
    """Metric 3: normalized continuous-action drift on aligned predictions."""

    previous = np.asarray(previous_actions, dtype=np.float64)
    current = np.asarray(current_actions, dtype=np.float64)
    if previous.ndim != 2 or current.ndim != 2 or previous.shape[1] != current.shape[1]:
        raise ValueError("actions must have compatible [horizon, action_dim] shapes")
    overlap = min(previous.shape[0] - executed_steps, current.shape[0])
    if overlap <= 0:
        raise ValueError("No prediction-horizon overlap")
    delta = np.abs(
        previous[executed_steps : executed_steps + overlap] - current[:overlap]
    )
    per_dimension = np.mean(delta, axis=0)
    top_k = min(4, per_dimension.size)
    return {
        "overlap_continuous_drift": float(np.mean(delta)),
        "overlap_continuous_top4": float(np.mean(np.sort(per_dimension)[-top_k:])),
        "overlap_steps": int(overlap),
    }


def windowed_total_variation(values: Sequence[float], window: int = 5) -> float:
    if window < 2:
        raise ValueError("window must be at least two")
    if values is None or len(values) < 2:
        return 0.0
    recent = np.asarray(values[-window:], dtype=np.float64)
    return float(np.nanmean(np.abs(np.diff(recent))))


def threshold_for_rate(values: Sequence[float], target_rate: float) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("inf")
    rate = float(np.clip(target_rate, 0.0, 1.0))
    return float(np.quantile(clean, 1.0 - rate))


@dataclass
class Router:
    mode: str
    cot_ratio: float = 0.2
    threshold: float | None = None
    tv_window: int = 5
    seed: int = 0
    entropy_history: list[float] = field(default_factory=list)
    farmass_history: list[float] = field(default_factory=list)
    query_index: int = 0

    def __post_init__(self) -> None:
        valid = {
            "baseline",
            "fixed",
            "random",
            "entropy_wtv",
            "farmass_wtv",
            "pilot",
            *TEMPORAL_MODE_TO_METRIC,
        }
        if self.mode not in valid:
            raise ValueError(f"Unknown router mode {self.mode!r}; choose from {sorted(valid)}")
        if not 0.0 <= self.cot_ratio <= 1.0:
            raise ValueError("cot_ratio must be in [0, 1]")
        if self.mode in THRESHOLD_MODES and self.threshold is None:
            raise ValueError(f"{self.mode} requires a threshold")
        self.rng = np.random.default_rng(self.seed)

    def update(self, metrics: dict[str, object]) -> dict[str, object]:
        self.entropy_history.append(float(metrics["entropy_mean"]))
        self.farmass_history.append(float(metrics["farmass_mean"]))
        entropy_wtv = windowed_total_variation(self.entropy_history, self.tv_window)
        farmass_wtv = windowed_total_variation(self.farmass_history, self.tv_window)

        if self.mode in {"baseline", "pilot"}:
            use_zr0 = False
        elif self.mode == "fixed":
            interval = max(1, round(1.0 / self.cot_ratio)) if self.cot_ratio > 0 else 10**18
            # Match the Adaptive-CoT LIBERO evaluator: query 0, N, 2N, ...
            # are the fixed-interval reasoning queries.
            use_zr0 = self.query_index % interval == 0
        elif self.mode == "random":
            use_zr0 = bool(self.rng.random() < self.cot_ratio)
        elif self.mode == "entropy_wtv":
            use_zr0 = len(self.entropy_history) >= 2 and entropy_wtv >= float(self.threshold)
        elif self.mode == "farmass_wtv":
            use_zr0 = len(self.farmass_history) >= 2 and farmass_wtv >= float(self.threshold)
        else:
            metric_name = TEMPORAL_MODE_TO_METRIC[self.mode]
            value = metrics.get(metric_name)
            # The two overlap metrics are undefined for query zero.  None is
            # deliberately neither routed nor coerced to zero.
            use_zr0 = bool(
                value is not None
                and np.isfinite(float(value))
                and float(value) >= float(self.threshold)
            )

        result = {
            "query_index": self.query_index,
            "entropy_wtv": entropy_wtv,
            "farmass_wtv": farmass_wtv,
            "use_zr0": use_zr0,
            "routing_metric": TEMPORAL_MODE_TO_METRIC.get(self.mode, self.mode),
            "routing_value": (
                metrics.get(TEMPORAL_MODE_TO_METRIC[self.mode])
                if self.mode in TEMPORAL_MODE_TO_METRIC
                else entropy_wtv
                if self.mode == "entropy_wtv"
                else farmass_wtv
                if self.mode == "farmass_wtv"
                else None
            ),
        }
        self.query_index += 1
        return result
