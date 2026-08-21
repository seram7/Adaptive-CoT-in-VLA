#!/usr/bin/env python3
"""Create expert-feasible manifests for the Motus PI0.5 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robotwin.pi05_task_registry import get_pi05_task
from experiments.robotwin.robotwin_env import RoboTwinHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-block", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_spec = get_pi05_task(args.task)
    existing: dict = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("task") != args.task or existing.get("task_config") != args.task_config:
            raise ValueError(f"Manifest identity mismatch: {args.output}")
    episodes = list(existing.get("episodes", []))
    if len(episodes) >= args.episodes:
        print(f"{args.output} already contains {len(episodes)} episodes")
        return

    harness = RoboTwinHarness(
        args.robotwin_root,
        args.task,
        args.task_config,
        step_limit=task_spec.step_limit,
    )
    additions = harness.find_feasible_seeds(
        args.episodes - len(episodes),
        start_seed=100000 * (1 + args.seed_block),
        skip_seeds=(item["seed"] for item in episodes),
    )
    episodes.extend(additions)
    payload = {
        "task": args.task,
        "task_config": args.task_config,
        "seed_block": args.seed_block,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(episodes)} expert-validated seeds to {args.output}")


if __name__ == "__main__":
    main()
