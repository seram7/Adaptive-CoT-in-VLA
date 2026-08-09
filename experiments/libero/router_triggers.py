"""Causal router trigger helpers for LIBERO OpenVLA/DeepThink experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def compute_windowed_avg_total_variation(
    series: Sequence[float],
    window: int = 5,
) -> float:
    if series is None or len(series) < 2:
        return 0.0
    arr = np.asarray(series[-window:], dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return float(np.nanmean(np.abs(np.diff(arr))))


def compute_farmass_log_contrast(
    m_series: Sequence[float],
    short_window: int = 3,
    long_window: int = 10,
    eps: float = 1e-6,
    std_floor: float = 1e-3,
) -> tuple[float, float]:
    """Return causal z_t and short-vs-long contrast s_t for the last item."""

    if short_window <= 0:
        raise ValueError("short_window must be positive")
    if long_window <= short_window:
        raise ValueError("long_window must be larger than short_window")
    if m_series is None or len(m_series) == 0:
        return 0.0, 0.0

    arr = np.asarray(m_series, dtype=np.float64)
    z = np.log(np.maximum(arr, 0.0) + eps)
    z_t = float(z[-1])
    if len(z) < long_window:
        return z_t, 0.0

    cur = float(np.mean(z[-short_window:]))
    prev = float(np.mean(z[-long_window:-short_window]))
    sd = float(np.std(z[-long_window:], ddof=0))
    score = (cur - prev) / max(sd, std_floor)
    return z_t, float(score)


@dataclass
class FarmassLogContrastTrigger:
    short_window: int = 3
    long_window: int = 10
    h_hi: float = 1.5
    h_lo: float = 0.5
    refire_max_fires: int = 1
    refire_interval: int = 5
    refire_min_score: float | None = None
    conflicted: bool = False
    conflict_age: int = 0
    conflict_fire_count: int = 0

    def update(self, m_series: Sequence[float]) -> tuple[float, float, bool, bool]:
        """Update trigger state and return z_t, s_t, conflicted, fired."""

        z_t, s_t = compute_farmass_log_contrast(
            m_series,
            short_window=self.short_window,
            long_window=self.long_window,
        )
        fired = False
        if len(m_series) < self.long_window:
            return z_t, s_t, self.conflicted, fired

        if not self.conflicted and s_t > self.h_hi:
            self.conflicted = True
            self.conflict_age = 0
            self.conflict_fire_count = 1
            fired = True
        elif self.conflicted:
            self.conflict_age += 1
            if s_t < self.h_lo:
                self.conflicted = False
                self.conflict_age = 0
                self.conflict_fire_count = 0
            else:
                min_score = (
                    float(self.refire_min_score)
                    if self.refire_min_score is not None
                    else 0.5 * (float(self.h_hi) + float(self.h_lo))
                )
                can_refire = (
                    self.refire_max_fires > self.conflict_fire_count
                    and self.refire_interval > 0
                    and self.conflict_age % self.refire_interval == 0
                    and s_t > min_score
                )
                if can_refire:
                    self.conflict_fire_count += 1
                    fired = True

        return z_t, s_t, self.conflicted, fired


def simulate_farmass_log_contrast(
    m_series: Sequence[float],
    short_window: int = 3,
    long_window: int = 10,
    h_hi: float = 1.5,
    h_lo: float | None = None,
) -> dict[str, np.ndarray]:
    if h_lo is None:
        h_lo = h_hi / 3.0
    trigger = FarmassLogContrastTrigger(
        short_window=short_window,
        long_window=long_window,
        h_hi=h_hi,
        h_lo=h_lo,
    )

    z_values: list[float] = []
    s_values: list[float] = []
    conflicted_values: list[float] = []
    fired_values: list[float] = []
    prefix: list[float] = []
    for m in m_series:
        prefix.append(float(m))
        z_t, s_t, conflicted, fired = trigger.update(prefix)
        z_values.append(z_t)
        s_values.append(s_t)
        conflicted_values.append(float(conflicted))
        fired_values.append(float(fired))

    return {
        "z": np.asarray(z_values, dtype=np.float64),
        "s": np.asarray(s_values, dtype=np.float64),
        "conflicted": np.asarray(conflicted_values, dtype=np.float64),
        "fired": np.asarray(fired_values, dtype=np.float64),
    }


def threshold_for_firing_rate(values: Sequence[float], firing_rate: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("inf")
    firing_rate = float(np.clip(firing_rate, 0.0, 1.0))
    return float(np.quantile(arr, 1.0 - firing_rate))
