#!/usr/bin/env python3
"""Summarize OpenVLA + DeepThinkVLA LIBERO rollout JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BASE_THRESHOLDS = {
    "entropy": {
        "libero_spatial": 0.26,
        "libero_object": 0.24,
        "libero_goal": 0.23,
        "libero_10": 0.21,
    },
    "far_mass_x_peak_separation": {
        "libero_spatial": 1.05,
        "libero_object": 1.21,
        "libero_goal": 0.79,
        "libero_10": 0.93,
    },
}
FAMILY_LABELS = {
    "deepthink": "2gpu",
    "deepthink_router_chunk1": "2gpu",
    "deepthink_router_chunk5": "2gpu",
    "deepthink_farmass_split2g": "2gpu-original",
    "deepthink_farmass_split2g_rerun": "2gpu-rerun",
    "deepthink_single40gb": "shared40gb",
    "deepthink_farmass_shared40gb_rerun": "shared40gb-rerun",
    "deepthink_router_entropy_cot02_shared40gb_chunk1": "shared40gb-cot0.2",
    "deepthink_router_entropy_cot02_shared40gb_chunk5": "shared40gb-cot0.2",
    "libero10_cotratio_50ep_dual2gpu_chunk1": "2gpu-target-cot",
}


@dataclass
class Run:
    path: Path
    family: str
    suite: str
    rows: list[dict[str, Any]]
    raw_rows: int
    invalid_rows: int
    duplicate_rows: int
    conflicting_duplicate_outcomes: int
    config: dict[str, Any]

    @property
    def episode_map(self) -> dict[tuple[int, int], int]:
        return {
            (int(row["task_id"]), int(row["trial_id"])): int(bool(row.get("success")))
            for row in self.rows
        }

    @property
    def chunk(self) -> int | None:
        value = self.config.get("deepthink_execute_chunk_steps")
        return int(value) if value is not None else None

    @property
    def mode(self) -> str | None:
        return self.config.get("router_control_mode")

    @property
    def metric(self) -> str | None:
        return self.config.get("uncertainty_metric_name")

    @property
    def threshold(self) -> float | None:
        value = self.config.get("score_threshold")
        return float(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/seram7/codes/Adaptive-CoT-in-VLA/rollouts_42_h100_new"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--mcnemar", type=Path, default=None)
    return parser.parse_args()


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def add_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    """Add Holm-adjusted p-values within each comparison family."""
    comparison_types = sorted({str(row["comparison_type"]) for row in rows})
    for comparison_type in comparison_types:
        group = [row for row in rows if row["comparison_type"] == comparison_type]
        ordered = sorted(group, key=lambda row: float(row["exact_mcnemar_p"]))
        running_max = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * float(row["exact_mcnemar_p"]))
            running_max = max(running_max, adjusted)
            row["holm_p"] = running_max


def summary_paths(root: Path) -> list[Path]:
    # The known layout is at most family/suite/run/episode_summary.jsonl. Avoid a
    # recursive walk through tens of thousands of rollout images.
    return sorted(
        set(root.glob("*/*/episode_summary.jsonl"))
        | set(root.glob("*/*/*/episode_summary.jsonl"))
    )


def load_run(root: Path, path: Path) -> Run | None:
    parsed: list[dict[str, Any]] = []
    invalid_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                parsed.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                invalid_rows += 1
    if not parsed:
        return None

    by_episode: dict[tuple[int, int], dict[str, Any]] = {}
    outcomes: dict[tuple[int, int], set[int]] = {}
    for row in parsed:
        key = (int(row.get("task_id", -1)), int(row.get("trial_id", -1)))
        outcomes.setdefault(key, set()).add(int(bool(row.get("success"))))
        by_episode[key] = row

    rel = path.relative_to(root)
    family = rel.parts[0]
    first = parsed[0]
    suite = str(first.get("task_suite") or (rel.parts[1] if len(rel.parts) > 3 else family))
    return Run(
        path=path,
        family=family,
        suite=suite,
        rows=list(by_episode.values()),
        raw_rows=len(parsed),
        invalid_rows=invalid_rows,
        duplicate_rows=len(parsed) - len(by_episode),
        conflicting_duplicate_outcomes=sum(len(values) > 1 for values in outcomes.values()),
        config=dict(first.get("router_config") or {}),
    )


def is_close(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and math.isclose(left, right, abs_tol=1e-9)


def run_name(run: Run) -> str:
    return run.path.parent.name


def target_cot(run: Run) -> float | None:
    match = re.search(r"targetcot(\d+)p(\d+)", run_name(run))
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def category(run: Run) -> str:
    if run.mode == "deepthink_plain":
        return "deepthink-only"
    if run.family in SUITES:
        return "openvla-calibration"
    if run.mode == "random":
        return "random-target-cot"
    if run.family == "libero10_cotratio_50ep_dual2gpu_chunk1":
        return "metric-target-cot"
    if run.mode == "metric_window_total_variation":
        return "hybrid-metric"
    return "other"


def is_canonical(run: Run) -> bool:
    if run.family in {"deepthink_router_chunk1", "deepthink_router_chunk5"}:
        return True
    if run.family == "deepthink" and run.mode == "metric_window_total_variation":
        return True
    if run.family == "deepthink_farmass_split2g_rerun":
        return True
    return False


def task_counts(run: Run) -> str:
    counts = Counter(int(row.get("task_id", -1)) for row in run.rows)
    return " ".join(f"{task_id}:{counts[task_id]}" for task_id in sorted(counts))


def step_weighted_mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    usable = [row for row in rows if row.get(key) is not None and int(row.get("num_steps") or 0) > 0]
    total_steps = sum(int(row["num_steps"]) for row in usable)
    if total_steps == 0:
        return None
    return sum(float(row[key]) * int(row["num_steps"]) for row in usable) / total_steps


def nonnull_mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def inventory_row(root: Path, run: Run) -> dict[str, Any]:
    n = len(run.rows)
    successes = sum(int(bool(row.get("success"))) for row in run.rows)
    ci_low, ci_high = wilson_interval(successes, n)
    total_steps = sum(int(row.get("num_steps") or 0) for row in run.rows)
    deepthink_steps = sum(int(row.get("num_deepthink_steps") or 0) for row in run.rows)
    openvla_latency = step_weighted_mean(run.rows, "mean_openvla_inference_time")
    selected_latency = step_weighted_mean(run.rows, "mean_selected_inference_time")
    deepthink_request_latency = nonnull_mean(run.rows, "mean_deepthink_inference_time")
    base = BASE_THRESHOLDS.get(run.metric or "", {}).get(run.suite)
    return {
        "category": category(run),
        "canonical": is_canonical(run),
        "family": run.family,
        "runtime": FAMILY_LABELS.get(run.family, "n/a"),
        "suite": run.suite,
        "chunk_size": run.chunk,
        "routing_mode": run.mode,
        "metric": run.metric,
        "metric_aggregate": "window_total_variation" if run.mode == "metric_window_total_variation" else run.mode,
        "window": run.config.get("tv_window"),
        "threshold_direction": run.config.get("score_threshold_direction"),
        "threshold": run.threshold,
        "is_base_threshold": is_close(run.threshold, base),
        "target_cot": target_cot(run),
        "random_probability": run.config.get("random_deepthink_probability"),
        "raw_jsonl_rows": run.raw_rows,
        "rollouts": n,
        "expected_rollouts": 500,
        "complete": n == 500 and all(count == 50 for count in Counter(int(row.get("task_id", -1)) for row in run.rows).values()),
        "successes": successes,
        "success_rate": successes / n if n else math.nan,
        "wilson95_low": ci_low,
        "wilson95_high": ci_high,
        "mean_selected_latency_s_per_step": selected_latency,
        "mean_openvla_latency_s_per_step": openvla_latency,
        "mean_deepthink_request_latency_s": deepthink_request_latency,
        "deepthink_overhead_s_per_step": (
            selected_latency - openvla_latency
            if selected_latency is not None and openvla_latency is not None
            else None
        ),
        "selected_inference_throughput_hz": 1 / selected_latency if selected_latency else None,
        "deepthink_step_ratio": deepthink_steps / total_steps if total_steps else math.nan,
        "error_episodes": sum(row.get("error") is not None for row in run.rows),
        "duplicate_jsonl_rows": run.duplicate_rows,
        "conflicting_duplicate_outcomes": run.conflicting_duplicate_outcomes,
        "invalid_jsonl_rows": run.invalid_rows,
        "task_counts": task_counts(run),
        "summary_path": str(run.path),
        "relative_summary_path": str(run.path.relative_to(root)),
    }


def find_run(
    runs: Iterable[Run],
    family: str,
    suite: str,
    metric: str,
    threshold: float,
) -> Run:
    matches = [
        run
        for run in runs
        if run.family == family
        and run.suite == suite
        and run.metric == metric
        and is_close(run.threshold, threshold)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {family}/{suite}/{metric}/{threshold}, got {len(matches)}")
    return matches[0]


def openvla_baseline(runs: Iterable[Run], suite: str) -> Run:
    # These high-threshold 2-GPU entropy runs have zero DeepThinkVLA steps and
    # therefore provide a 500-episode OpenVLA-only baseline with matching IDs.
    threshold = {
        "libero_spatial": 0.95,
        "libero_object": 1.11,
        "libero_goal": 0.69,
        "libero_10": 0.83,
    }[suite]
    return find_run(runs, "deepthink", suite, "entropy", threshold)


def base_run(runs: Iterable[Run], chunk: int, suite: str, metric: str) -> Run:
    threshold = BASE_THRESHOLDS[metric][suite]
    if chunk == 1:
        family = "deepthink_router_chunk1"
    elif chunk == 5:
        family = "deepthink_router_chunk5"
    elif metric == "entropy":
        family = "deepthink"
    else:
        family = "deepthink_farmass_split2g_rerun"
    return find_run(runs, family, suite, metric, threshold)


def mcnemar_row(comparison_type: str, label: str, method: Run, reference: Run) -> dict[str, Any]:
    method_map = method.episode_map
    reference_map = reference.episode_map
    common = sorted(set(method_map) & set(reference_map))
    method_win = sum(method_map[key] == 1 and reference_map[key] == 0 for key in common)
    reference_win = sum(method_map[key] == 0 and reference_map[key] == 1 for key in common)
    method_latency = step_weighted_mean(method.rows, "mean_selected_inference_time")
    reference_latency = step_weighted_mean(reference.rows, "mean_selected_inference_time")
    return {
        "comparison_type": comparison_type,
        "comparison": label,
        "suite": method.suite,
        "paired_rollouts": len(common),
        "method_success_reference_failure": method_win,
        "method_failure_reference_success": reference_win,
        "discordant_pairs": method_win + reference_win,
        "exact_mcnemar_p": exact_mcnemar(method_win, reference_win),
        "method_rate": sum(method_map[key] for key in common) / len(common),
        "reference_rate": sum(reference_map[key] for key in common) / len(common),
        "method_selected_latency_s_per_step": method_latency,
        "reference_selected_latency_s_per_step": reference_latency,
        "latency_ratio": (
            method_latency / reference_latency
            if method_latency is not None and reference_latency is not None
            else None
        ),
        "method_path": str(method.path),
        "reference_path": str(reference.path),
    }


def fmt_rate(row: dict[str, Any]) -> str:
    return (
        f"{int(row['successes'])}/{int(row['rollouts'])} "
        f"({100 * float(row['success_rate']):.1f}%, "
        f"95% CI {100 * float(row['wilson95_low']):.1f}-{100 * float(row['wilson95_high']):.1f}%)"
    )


def fmt_latency(row: dict[str, Any], include_ratio: bool = True) -> str:
    value = row.get("mean_selected_latency_s_per_step")
    if value in {None, ""}:
        return "n/a"
    text = f"{1000 * float(value):.0f} ms/step"
    ratio = row.get("selected_latency_ratio_vs_openvla")
    if include_ratio and ratio not in {None, ""}:
        text += f" ({float(ratio):.2f}x)"
    return text


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_report(
    root: Path,
    runs: list[Run],
    inventory: list[dict[str, Any]],
    mcnemar: list[dict[str, Any]],
    snapshot: str,
) -> str:
    lines = [
        "# OpenVLA + DeepThinkVLA LIBERO Rollout Report",
        "",
        f"Snapshot: `{snapshot}`",
        "",
        "## Scope and counting",
        "",
        f"- Source: `{root}` ({len(runs)} non-empty `episode_summary.jsonl` files).",
        "- A rollout is one unique `(task_id, trial_id)` pair. Repeated JSONL rows are deduplicated by keeping the last record for that pair.",
        "- Success intervals are two-sided Wilson 95% confidence intervals.",
        "- The main chunk comparison uses the 2-GPU router lineage. Shared-40GB and target-CoT runs are listed separately because execution/model settings differ.",
        "- `complete` means 500 unique rollouts with 50 rollouts for each of the 10 tasks.",
        "",
        "## Router definition",
        "",
        "All metric-gated runs in this directory use `metric_window_total_variation`, window 5, direction `gt`:",
        "",
        "1. OpenVLA produces a token distribution for each of the seven action dimensions.",
        "2. The per-step metric is the mean across those seven dimensions: token entropy, or `far_mass * peak_separation`.",
        "3. The control score is the mean absolute first difference over the last five per-step metric values.",
        "4. DeepThinkVLA is selected when that control score is greater than the threshold, and its queued action chunk has length 1, 5, or 10.",
        "",
        "Latency is the step-weighted mean model-inference time. It includes OpenVLA inference on every environment step plus DeepThinkVLA inference only when a new action chunk is requested; cached actions in the rest of a chunk add no new DeepThink inference call. It excludes environment simulation and file I/O.",
        "",
        "No completed metric-gated JSONL in this snapshot uses raw metric, window mean, relative change, or full-history total variation aggregation.",
        "",
        "## Baselines",
        "",
        "The OpenVLA-only rows are 2-GPU entropy runs whose recorded DeepThink step ratio is exactly zero. DeepThink-only chunk 1/5 rows come from `plain_chunk*`.",
        "",
    ]

    baseline_rows = []
    inv_by_path = {row["summary_path"]: row for row in inventory}
    for suite in SUITES:
        baseline = openvla_baseline(runs, suite)
        baseline_row = inv_by_path[str(baseline.path)]
        baseline_rows.append((suite, "OpenVLA only", "-", fmt_rate(baseline_row), fmt_latency(baseline_row)))
        for chunk in (1, 5):
            plain = next(
                run
                for run in runs
                if run.family == "deepthink" and run.suite == suite and run.mode == "deepthink_plain" and run.chunk == chunk
            )
            plain_row = inv_by_path[str(plain.path)]
            baseline_rows.append((suite, "DeepThinkVLA only", chunk, fmt_rate(plain_row), fmt_latency(plain_row)))
    lines.extend(markdown_table(["Suite", "Policy", "Chunk", "Success", "Latency vs OpenVLA"], baseline_rows))

    lines.extend(
        [
            "",
            "## Preset base thresholds",
            "",
            "This is the cleanest chunk-size comparison: the suite-specific preset threshold is held fixed within each metric.",
            "",
        ]
    )
    base_rows = []
    for suite in SUITES:
        for metric in BASE_THRESHOLDS:
            threshold = BASE_THRESHOLDS[metric][suite]
            for chunk in (1, 5, 10):
                run = base_run(runs, chunk, suite, metric)
                row = inv_by_path[str(run.path)]
                base_rows.append(
                    (
                        suite,
                        chunk,
                        "entropy" if metric == "entropy" else "far-mass x peak-sep",
                        threshold,
                        fmt_rate(row),
                        fmt_latency(row),
                        f"{100 * float(row['deepthink_step_ratio']):.1f}%",
                    )
                )
    lines.extend(markdown_table(["Suite", "Chunk", "Metric", "Threshold", "Success", "Latency vs OpenVLA", "DeepThink steps"], base_rows))

    lines.extend(
        [
            "",
            "### Latency-aware reading",
            "",
            "At the preset thresholds, chunk 10 has the lowest selected latency and equal-or-higher point-estimate success in all eight suite/metric comparisons. Its latency is 279-318 ms/step (1.15-1.31x OpenVLA), versus chunk 5 at 286-329 ms/step (1.18-1.36x) and chunk 1 at 404-505 ms/step (1.66-2.08x). This is chunk amortization, not a faster DeepThinkVLA call: one roughly one-second DeepThink inference supplies more cached actions.",
            "",
            "Thus chunk 10 is the point-estimate latency/success Pareto choice at the preset thresholds. The success-rate differences between chunks are still not statistically robust, as shown by the paired tests below.",
        ]
    )

    lines.extend(
        [
            "",
            "## Paired test against OpenVLA",
            "",
            "McNemar uses the same `(task_id, trial_id)` across the 500-episode OpenVLA-only and preset-threshold runs. `W-L` is hybrid-success/OpenVLA-failure versus hybrid-failure/OpenVLA-success. These are preselected base thresholds, avoiding best-threshold post-selection in this test.",
            "",
        ]
    )
    baseline_tests = [row for row in mcnemar if row["comparison_type"] == "preset-vs-openvla"]
    mcnemar_rows = []
    for row in baseline_tests:
        mcnemar_rows.append(
            (
                row["suite"],
                row["comparison"].split(" vs ")[0],
                row["paired_rollouts"],
                f"{row['method_success_reference_failure']}-{row['method_failure_reference_success']}",
                f"{row['exact_mcnemar_p']:.3g}",
                f"{row['holm_p']:.3g}",
            )
        )
    lines.extend(markdown_table(["Suite", "Hybrid", "Paired n", "W-L", "Exact p", "Holm p"], mcnemar_rows))

    lines.extend(
        [
            "",
            "CI comparison against the OpenVLA-only proxy: Spatial's preset entropy rows and chunk-1 far-mass row overlap the OpenVLA CI slightly; Spatial far-mass chunk 5/10 and every Object, Goal, and LIBERO-10 preset row do not overlap it. The paired test detects additional differences because it uses episode-level outcomes. Holm p-values adjust across the 24 preset-vs-OpenVLA tests.",
            "",
            "## Paired chunk-size comparison",
            "",
            "These tests compare preset thresholds directly across chunk sizes. Positive delta means the first chunk in the comparison has the higher success rate. Holm adjustment is across these 24 chunk comparisons.",
            "",
        ]
    )
    chunk_tests = [row for row in mcnemar if row["comparison_type"] == "chunk-size"]
    chunk_rows = []
    for row in chunk_tests:
        method_latency_ms = 1000 * float(row["method_selected_latency_s_per_step"])
        reference_latency_ms = 1000 * float(row["reference_selected_latency_s_per_step"])
        chunk_rows.append(
            (
                row["suite"],
                row["comparison"],
                f"{100 * (float(row['method_rate']) - float(row['reference_rate'])):+.1f} pp",
                f"{method_latency_ms:.0f} vs {reference_latency_ms:.0f} ms ({100 * (float(row['latency_ratio']) - 1):+.0f}%)",
                f"{row['method_success_reference_failure']}-{row['method_failure_reference_success']}",
                f"{row['exact_mcnemar_p']:.3g}",
                f"{row['holm_p']:.3g}",
            )
        )
    lines.extend(markdown_table(["Suite", "Comparison", "Success delta", "Latency", "W-L", "Exact p", "Holm p"], chunk_rows))
    lines.extend(
        [
            "",
            "All preset chunk-size Wilson intervals overlap. Two paired comparisons are nominally below 0.05 before correction (Spatial far-mass chunk 5 vs 1, and LIBERO-10 far-mass chunk 10 vs 1), but none remains significant after Holm correction. The present data therefore do not support a robust global chunk-size winner at the preset thresholds.",
            "",
            "## Best observed canonical settings",
            "",
            "These are descriptive maxima over the tested thresholds, not confirmatory estimates; chunk 10 has a much denser far-mass sweep than chunk 1/5.",
            "",
        ]
    )
    best_rows = []
    canonical_metric = [row for row in inventory if row["canonical"] and row["category"] == "hybrid-metric"]
    for suite in SUITES:
        for chunk in (1, 5, 10):
            candidates = [row for row in canonical_metric if row["suite"] == suite and row["chunk_size"] == chunk]
            best = max(candidates, key=lambda row: (float(row["success_rate"]), -float(row["deepthink_step_ratio"])))
            best_rows.append(
                (
                    suite,
                    chunk,
                    "entropy" if best["metric"] == "entropy" else "far-mass x peak-sep",
                    best["threshold"],
                    fmt_rate(best),
                    fmt_latency(best),
                    f"{100 * float(best['deepthink_step_ratio']):.1f}%",
                )
            )
    lines.extend(markdown_table(["Suite", "Chunk", "Metric", "Threshold", "Success", "Latency vs OpenVLA", "DeepThink steps"], best_rows))

    lines.extend(
        [
            "",
            "## Full canonical threshold inventory",
            "",
            "The following rows are the main 2-GPU lineage. Chunk 10 far-mass uses the comprehensive `split2g_rerun` directory; the overlapping earlier `split2g` directory is retained only in the CSV inventory.",
            "",
        ]
    )
    canonical = canonical_metric
    canonical.sort(
        key=lambda row: (
            SUITES.index(str(row["suite"])),
            int(row["chunk_size"]),
            str(row["metric"]),
            float(row["threshold"]),
        )
    )
    for suite in SUITES:
        lines.extend([f"### {suite}", ""])
        suite_rows = []
        for row in canonical:
            if row["suite"] != suite:
                continue
            suite_rows.append(
                (
                    row["chunk_size"],
                    "entropy" if row["metric"] == "entropy" else "far-mass x peak-sep",
                    row["threshold"],
                    f"{row['rollouts']}{'' if row['complete'] else '*'}",
                    fmt_rate(row),
                    fmt_latency(row),
                    f"{100 * float(row['deepthink_step_ratio']):.1f}%",
                )
            )
        lines.extend(markdown_table(["Chunk", "Metric", "Threshold", "n", "Success", "Latency vs OpenVLA", "DeepThink steps"], suite_rows))
        lines.append("")

    supplemental = [
        row
        for row in inventory
        if row["category"] == "hybrid-metric" and not row["canonical"]
    ]
    incomplete = [row for row in inventory if row["category"] in {"hybrid-metric", "metric-target-cot", "random-target-cot"} and not row["complete"]]
    duplicate_total = sum(int(row["duplicate_jsonl_rows"]) for row in inventory)
    conflict_total = sum(int(row["conflicting_duplicate_outcomes"]) for row in inventory)
    invalid_total = sum(int(row["invalid_jsonl_rows"]) for row in inventory)
    lines.extend(
        [
            "## Additional runs and data quality",
            "",
            f"- The CSV contains {len(inventory)} total configurations: {len(canonical)} canonical metric rows, {len(supplemental)} non-canonical metric rows, plus DeepThink-only, OpenVLA calibration, and target-CoT/random runs.",
            f"- Deduplication removed {duplicate_total} repeated JSONL rows. {conflict_total} episode keys had conflicting duplicate success outcomes; the last JSONL record is authoritative. Invalid JSONL rows: {invalid_total}.",
            "- Conflicting outcomes occur only in LIBERO-10 far-mass chunk-10 threshold 1.13: 9 keys in `deepthink_farmass_split2g_rerun` and 14 keys in `deepthink_farmass_shared40gb_rerun`. Treat those two rates as provisional because no saved `.pt` file is available to resolve the competing JSONL outcomes.",
            f"- Incomplete/current configurations: {len(incomplete)}. They are marked `complete=False` in the CSV and `*` where shown above.",
            "- `deepthink_farmass_split2g` is an overlapping earlier chunk-10 run; `deepthink_farmass_split2g_rerun` is used for the canonical table.",
            "- Shared-40GB results can differ from 2-GPU results at identical settings, so they are not pooled as if they were extra iid trials.",
            "- The active LIBERO-10 target-CoT sweep is a moving snapshot and is not mixed into the canonical threshold table.",
            "",
        ]
    )
    if incomplete:
        incomplete_rows = [
            (
                row["family"],
                row["suite"],
                row["chunk_size"],
                row["routing_mode"],
                row["metric"],
                row["threshold"],
                row["target_cot"],
                row["rollouts"],
            )
            for row in incomplete
        ]
        lines.extend(markdown_table(["Family", "Suite", "Chunk", "Mode", "Metric", "Threshold", "Target CoT", "n"], incomplete_rows))
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- At the preset thresholds, chunk-size effects are suite- and metric-dependent rather than monotonic. Far-mass benefits most clearly from chunk 5/10 on Spatial and LIBERO-10; entropy changes are smaller except on LIBERO-10.",
            "- Latency changes the operational choice: chunk 10 weakly dominates chunk 1/5 at the preset point estimates, with the lowest per-step inference time and equal-or-higher success in every suite/metric pair.",
            "- Lower far-mass thresholds route far more steps to DeepThinkVLA and produce the highest observed rates in the exhaustive chunk-10 sweep. Those maxima are descriptive and threshold-selected, so they should not be presented as an unbiased confirmatory comparison.",
            "- Wilson intervals answer uncertainty for each individual rate. The paired McNemar rows use stronger episode-level information and directly test preset hybrid routing against OpenVLA-only.",
            "- CI overlap is a conservative visual heuristic, not itself a formal test. Use the exact McNemar p-values for paired claims.",
            "",
            "## Artifacts",
            "",
            "- `openvla_deepthink_performance_inventory.csv`: every configuration, rollout count, rate, Wilson CI, selected/OpenVLA/DeepThink latency, throughput, routing load, completeness, and source path.",
            "- `openvla_deepthink_mcnemar.csv`: paired preset-threshold comparisons against OpenVLA-only and paired chunk-size comparisons.",
            "- Generator: `analysis_phase_transition/summarize_openvla_deepthink_rollouts.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = args.report or root / "openvla_deepthink_performance_report.md"
    inventory_path = args.inventory or root / "openvla_deepthink_performance_inventory.csv"
    mcnemar_path = args.mcnemar or root / "openvla_deepthink_mcnemar.csv"

    runs = [run for path in summary_paths(root) if (run := load_run(root, path)) is not None]
    inventory = [inventory_row(root, run) for run in runs]
    inventory_by_path = {str(row["summary_path"]): row for row in inventory}
    baseline_latencies = {
        suite: inventory_by_path[str(openvla_baseline(runs, suite).path)]["mean_selected_latency_s_per_step"]
        for suite in SUITES
    }
    for row in inventory:
        baseline_latency = baseline_latencies.get(str(row["suite"]))
        selected_latency = row["mean_selected_latency_s_per_step"]
        row["openvla_baseline_selected_latency_s_per_step"] = baseline_latency
        row["selected_latency_ratio_vs_openvla"] = (
            float(selected_latency) / float(baseline_latency)
            if selected_latency is not None and baseline_latency is not None
            else None
        )
    inventory.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["suite"]),
            -1 if row["chunk_size"] is None else int(row["chunk_size"]),
            str(row["metric"]),
            -math.inf if row["threshold"] is None else float(row["threshold"]),
            str(row["relative_summary_path"]),
        )
    )

    comparisons = []
    for suite in SUITES:
        reference = openvla_baseline(runs, suite)
        for chunk in (1, 5, 10):
            for metric in BASE_THRESHOLDS:
                method = base_run(runs, chunk, suite, metric)
                short_metric = "entropy" if metric == "entropy" else "far-mass"
                comparisons.append(
                    mcnemar_row(
                        "preset-vs-openvla",
                        f"chunk{chunk}-{short_metric} vs OpenVLA",
                        method,
                        reference,
                    )
                )

    for suite in SUITES:
        for metric in BASE_THRESHOLDS:
            chunk_runs = {chunk: base_run(runs, chunk, suite, metric) for chunk in (1, 5, 10)}
            short_metric = "entropy" if metric == "entropy" else "far-mass"
            for method_chunk, reference_chunk in ((5, 1), (10, 1), (10, 5)):
                comparisons.append(
                    mcnemar_row(
                        "chunk-size",
                        f"chunk{method_chunk} vs chunk{reference_chunk} ({short_metric})",
                        chunk_runs[method_chunk],
                        chunk_runs[reference_chunk],
                    )
                )

    add_holm_adjustment(comparisons)

    snapshot = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    write_csv(inventory_path, inventory)
    write_csv(mcnemar_path, comparisons)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(build_report(root, runs, inventory, comparisons, snapshot), encoding="utf-8")

    print(f"Wrote {report}")
    print(f"Wrote {inventory_path} ({len(inventory)} rows)")
    print(f"Wrote {mcnemar_path} ({len(comparisons)} rows)")


if __name__ == "__main__":
    main()
