#!/usr/bin/env python3
"""Calibrate per-replan PI0.5 Far-Mass thresholds with step cooldowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robotwin.evaluate_pi05_zr0 import SamplingFarMassRouter
from experiments.robotwin.evaluate_pi05_zr0 import STEP_THRESHOLD_MODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["demo_clean", "demo_randomized"])
    parser.add_argument("--target-ratio", type=float, default=0.2)
    parser.add_argument("--cooldown-steps", type=int, required=True)
    parser.add_argument("--replan-stride", type=int, required=True)
    parser.add_argument("--force-first-query", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sequences_from_directory(path: Path) -> list[list[tuple[float, int]]]:
    sequences: list[list[tuple[float, int]]] = []
    for episode_path in sorted(path.glob("episode_*.json")):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        sequence: list[tuple[float, int]] = []
        for record in episode.get("records", []):
            value = record.get(STEP_THRESHOLD_MODE)
            if value is None or not np.isfinite(float(value)):
                continue
            sequence.append((float(value), int(record.get("executed_actions", 0))))
        if sequence:
            sequences.append(sequence)
    return sequences


def realized_ratio(
    sequences: list[list[tuple[float, int]]],
    threshold: float,
    *,
    target_ratio: float,
    cooldown_steps: int,
    force_first_query: bool,
) -> float:
    routed = 0
    queries = 0
    for episode_index, sequence in enumerate(sequences):
        router = SamplingFarMassRouter(
            STEP_THRESHOLD_MODE,
            cot_ratio=target_ratio,
            threshold=threshold,
            tv_window=2,
            seed=episode_index,
            cooldown_steps=cooldown_steps,
            force_first_query=force_first_query,
        )
        for value, executed_steps in sequence:
            decision = router.update(None, step_uncertainty=value)
            routed += int(bool(decision["use_zr0"]))
            queries += 1
            router.record_execution(executed_steps)
    return routed / max(1, queries)


def choose_threshold(
    sequences: list[list[tuple[float, int]]],
    *,
    target_ratio: float,
    cooldown_steps: int,
    force_first_query: bool,
) -> tuple[float, float, int]:
    values = np.asarray(
        [value for sequence in sequences for value, _ in sequence],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("No eligible per-step Far-Mass values")
    unique = np.unique(values)
    candidates = np.concatenate(
        [
            np.asarray([np.nextafter(unique[0], -np.inf)]),
            unique,
            np.asarray([np.nextafter(unique[-1], np.inf)]),
        ]
    )
    evaluations = []
    for threshold in candidates:
        ratio = realized_ratio(
            sequences,
            float(threshold),
            target_ratio=target_ratio,
            cooldown_steps=cooldown_steps,
            force_first_query=force_first_query,
        )
        evaluations.append((abs(ratio - target_ratio), -threshold, threshold, ratio))
    _, _, threshold, ratio = min(evaluations)
    return float(threshold), float(ratio), int(values.size)


def main() -> None:
    args = parse_args()
    payload: dict = {}
    for condition in args.conditions:
        payload[condition] = {}
        for task in args.tasks:
            source = args.pilot_root / condition / task / "pilot"
            sequences = sequences_from_directory(source)
            if not sequences:
                raise ValueError(f"No usable pilot sequences in {source}")
            threshold, calibration_ratio, eligible_samples = choose_threshold(
                sequences,
                target_ratio=args.target_ratio,
                cooldown_steps=args.cooldown_steps,
                force_first_query=args.force_first_query,
            )
            raw_values = [value for sequence in sequences for value, _ in sequence]
            payload[condition][task] = {
                STEP_THRESHOLD_MODE: threshold,
                "threshold_stats": {
                    STEP_THRESHOLD_MODE: {
                        "episodes": len(sequences),
                        "queries": sum(len(sequence) for sequence in sequences),
                        "eligible_samples": eligible_samples,
                        "target_ratio": args.target_ratio,
                        "calibration_ratio": calibration_ratio,
                        "cooldown_steps": args.cooldown_steps,
                        "replan_stride": args.replan_stride,
                        "force_first_query": args.force_first_query,
                        "raw_metric_mean": float(np.mean(raw_values)),
                        "raw_metric_std": float(np.std(raw_values)),
                        "pilot_dir": str(source),
                    }
                },
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
