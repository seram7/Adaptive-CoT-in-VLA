#!/usr/bin/env python
"""Sanity checks for the causal farmass_log_contrast router."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.router_triggers import (  # noqa: E402
    compute_windowed_avg_total_variation,
    simulate_farmass_log_contrast,
)


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def as_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def find_openvla_collect_root() -> Path:
    candidates = [
        PROJECT_ROOT / "rollouts_collect_v1",
        PROJECT_ROOT / "rollouts_42_h100_new",
        Path("/scratch/seram7/Adaptive-CoT-in-VLA/rollouts_42_h100_new"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        found = list(root.glob("libero_spatial/openvla_phase_transition_minimal_t0_9_tr0_4_v2"))
        if found:
            return root
    raise FileNotFoundError("Could not find rollouts_collect_v1 or openvla_phase_transition_minimal_t0_9_tr0_4_v2")


def synthetic_series(seed: int = 7) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    y = 0.04 + rng.normal(0.0, 0.001, size=90)
    pulses = [(22, 30), (58, 66)]
    for start, end in pulses:
        y[start:end] += 0.20
    return np.clip(y, 1e-5, None), pulses


def local_maxima(values: np.ndarray, threshold: float, merge_gap: int = 5) -> np.ndarray:
    raw = []
    for i in range(1, len(values) - 1):
        if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            raw.append(i)
    if not raw:
        return np.empty((0,), dtype=np.int32)

    merged = []
    cluster = [raw[0]]
    for idx in raw[1:]:
        if idx - cluster[-1] <= merge_gap:
            cluster.append(idx)
            continue
        merged.append(max(cluster, key=lambda j: values[j]))
        cluster = [idx]
    merged.append(max(cluster, key=lambda j: values[j]))
    return np.asarray(merged, dtype=np.int32)


def run_synthetic() -> dict:
    series, pulses = synthetic_series()
    wtv = np.asarray(
        [compute_windowed_avg_total_variation(series[: i + 1], window=5) for i in range(len(series))],
        dtype=np.float64,
    )
    wtv_threshold = 0.025
    wtv_fires = local_maxima(wtv, threshold=wtv_threshold)
    log_result = simulate_farmass_log_contrast(
        series,
        short_window=3,
        long_window=10,
        h_hi=1.5,
        h_lo=1.5 / 3.0,
    )
    log_fires = np.flatnonzero(log_result["fired"] > 0.5)
    peaks = [int((start + end - 1) / 2) for start, end in pulses]
    return {
        "pulse_segments": pulses,
        "pulse_peaks": peaks,
        "wtv_threshold": wtv_threshold,
        "wtv_fires": wtv_fires.tolist(),
        "wtv_fire_count": int(len(wtv_fires)),
        "log_contrast_h_hi": 1.5,
        "log_contrast_h_lo": 1.5 / 3.0,
        "log_contrast_fires": log_fires.tolist(),
        "log_contrast_fire_count": int(len(log_fires)),
        "log_fires_before_peaks": [bool(fire < peak) for fire, peak in zip(log_fires.tolist(), peaks)],
    }


def load_m_series(pt_path: Path) -> np.ndarray:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    for key in ("far_mass_x_peak_separation_arm_mean_series", "farmass_log_m_series"):
        if key in payload:
            arr = as_numpy(payload[key])
            if arr.size:
                return arr
    if "far_mass_x_peak_separation_per_slot_series" in payload:
        slots = as_numpy(payload["far_mass_x_peak_separation_per_slot_series"])
        if slots.ndim == 2 and slots.shape[1] >= 6:
            return np.nanmean(slots[:, :6], axis=1)
    if "far_mass_x_peak_separation_series" in payload:
        return as_numpy(payload["far_mass_x_peak_separation_series"])
    if "openvla_metric_series" in payload:
        return as_numpy(payload["openvla_metric_series"])
    if "selected_metric_series" in payload:
        return as_numpy(payload["selected_metric_series"])
    raise KeyError(f"No far_mass series found in {pt_path}")


def run_offline(root: Path, h_values: list[float], max_files_per_suite: int | None = None) -> list[dict]:
    rows = []
    for suite in SUITES:
        run_dirs = sorted((root / suite).glob("openvla_phase_transition_minimal_t0_9_tr0_4_v2"))
        if not run_dirs:
            continue
        pt_files = sorted(run_dirs[0].glob("**/task*_trial*.pt"))
        if max_files_per_suite is not None:
            pt_files = pt_files[:max_files_per_suite]
        suite_series = []
        for pt_path in pt_files:
            try:
                series = load_m_series(pt_path)
            except Exception:
                continue
            if series.size:
                suite_series.append(series)
        total_steps = int(sum(len(s) for s in suite_series))
        for h_hi in h_values:
            fire_steps = 0
            conflicted_steps = 0
            for series in suite_series:
                result = simulate_farmass_log_contrast(
                    series,
                    short_window=3,
                    long_window=10,
                    h_hi=h_hi,
                    h_lo=h_hi / 3.0,
                )
                fire_steps += int(np.sum(result["fired"] > 0.5))
                conflicted_steps += int(np.sum(result["conflicted"] > 0.5))
            rows.append(
                {
                    "suite": suite,
                    "h_hi": float(h_hi),
                    "h_lo": float(h_hi / 3.0),
                    "episodes": int(len(suite_series)),
                    "steps": total_steps,
                    "fire_rate_onset": fire_steps / total_steps if total_steps else 0.0,
                    "conflicted_rate_sustain": conflicted_steps / total_steps if total_steps else 0.0,
                    "fires": fire_steps,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "analysis_phase_transition"))
    parser.add_argument("--max-files-per-suite", type=int, default=None)
    parser.add_argument("--h-values", default="0.75,1.0,1.25,1.5,2.0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h_values = [float(x) for x in args.h_values.split(",") if x.strip()]

    synth = run_synthetic()
    collect_root = find_openvla_collect_root()
    offline_rows = run_offline(collect_root, h_values, args.max_files_per_suite)

    json_path = output_dir / "farmass_log_contrast_sanity.json"
    csv_path = output_dir / "farmass_log_contrast_offline_rates.csv"
    md_path = output_dir / "farmass_log_contrast_sanity.md"

    json_path.write_text(
        json.dumps(
            {
                "synthetic": synth,
                "collect_root": str(collect_root),
                "offline": offline_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "suite",
                "h_hi",
                "h_lo",
                "episodes",
                "steps",
                "fire_rate_onset",
                "conflicted_rate_sustain",
                "fires",
            ],
        )
        writer.writeheader()
        writer.writerows(offline_rows)

    lines = [
        "# farmass_log_contrast sanity",
        "",
        "## Synthetic pulse check",
        "",
        f"- pulse segments: {synth['pulse_segments']}",
        f"- pulse peaks: {synth['pulse_peaks']}",
        f"- metric_window_total_variation fires: {synth['wtv_fires']} count={synth['wtv_fire_count']}",
        f"- farmass_log_contrast onset fires: {synth['log_contrast_fires']} count={synth['log_contrast_fire_count']}",
        f"- log fires before pulse peaks: {synth['log_fires_before_peaks']}",
        "",
        "## Offline OpenVLA-only rates",
        "",
        f"- source root: `{collect_root}`",
        "",
        "| suite | h_hi | onset fire rate | sustain conflicted rate | episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in offline_rows:
        lines.append(
            f"| {row['suite']} | {row['h_hi']:.3f} | {row['fire_rate_onset']:.4f} | "
            f"{row['conflicted_rate_sustain']:.4f} | {row['episodes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"synthetic": synth, "offline_rows": offline_rows}, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
