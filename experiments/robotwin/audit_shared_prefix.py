#!/usr/bin/env python3
"""Assert common-prefix inference equality across fixed/random/adaptive arms."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any


DEFAULT_ARMS = (
    "fixed",
    "random",
    "adaptive_sampling_farmass_replan_max",
)


def _episodes(path: Path) -> dict[str, dict[str, Any]]:
    return {
        item.name: json.loads(item.read_text(encoding="utf-8"))
        for item in sorted(path.glob("episode_*.json"))
    }


def audit_pair(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    if (left["task"], left["task_config"], left["seed"]) != (
        right["task"],
        right["task_config"],
        right["seed"],
    ):
        raise AssertionError("Attempted to compare different episodes")
    left_records = left["records"]
    right_records = right["records"]
    for query_index, (lhs, rhs) in enumerate(zip(left_records, right_records, strict=False)):
        if lhs["query_index"] != query_index or rhs["query_index"] != query_index:
            raise AssertionError(f"Non-contiguous query index at {query_index}")
        if lhs["observation_hash"] != rhs["observation_hash"]:
            raise AssertionError(
                f"Observation diverged before routing decision at query {query_index}"
            )
        if lhs["pi05_action_seed"] != rhs["pi05_action_seed"]:
            raise AssertionError(f"PI0.5 execution seed mismatch at query {query_index}")
        if lhs["pi05_action_hash"] != rhs["pi05_action_hash"]:
            raise AssertionError(f"PI0.5 action mismatch at query {query_index}")

        if lhs["selected_policy"] != rhs["selected_policy"]:
            return query_index
        for field in (
            "selected_action_hash",
            "executed_action_hash",
            "executed_actions",
        ):
            if lhs[field] != rhs[field]:
                raise AssertionError(
                    f"Common-prefix {field} mismatch at query {query_index}: "
                    f"{lhs[field]} != {rhs[field]}"
                )

    if len(left_records) != len(right_records):
        raise AssertionError("One arm terminated before any routing decision diverged")
    if (left["success"], left["sim_steps"]) != (right["success"], right["sim_steps"]):
        raise AssertionError("Identical routing trajectory produced a different episode outcome")
    return None


def audit_condition_root(condition_root: Path, arms: tuple[str, ...]) -> dict[str, Any]:
    payloads = {arm: _episodes(condition_root / arm) for arm in arms}
    names = set(payloads[arms[0]])
    for arm in arms[1:]:
        if set(payloads[arm]) != names:
            raise AssertionError(f"Episode set differs for {arm}")
    if not names:
        raise ValueError(f"No episode JSON files under {condition_root}")

    pairs: dict[str, Any] = {}
    for left_arm, right_arm in combinations(arms, 2):
        divergences = []
        identical = 0
        for name in sorted(names):
            divergence = audit_pair(payloads[left_arm][name], payloads[right_arm][name])
            if divergence is None:
                identical += 1
            else:
                divergences.append(divergence)
        pairs[f"{left_arm}__{right_arm}"] = {
            "episodes": len(names),
            "identical_full_trajectories": identical,
            "first_divergence_query_min": min(divergences) if divergences else None,
            "first_divergence_query_max": max(divergences) if divergences else None,
        }
    return {"condition_root": str(condition_root.resolve()), "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_condition_root(args.condition_root, tuple(args.arms))
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
