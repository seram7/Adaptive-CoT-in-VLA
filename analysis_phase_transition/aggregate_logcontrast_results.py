#!/usr/bin/env python
"""Aggregate OpenVLA+DeepThink farmass_log_contrast experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis_phase_transition"))

from analyze_critical_segments import (  # noqa: E402
    SUITES,
    evaluate_fire_set,
    label_all,
    load_episodes,
)


def torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def mean_or_nan(values) -> float:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else math.nan


def discover_run_names(root: Path) -> list[str]:
    names = set()
    for suite in SUITES:
        suite_dir = root / suite
        if not suite_dir.exists():
            continue
        for child in suite_dir.iterdir():
            if child.is_dir():
                names.add(child.name)
    return sorted(names)


def describe_run(payload: dict, run_name: str) -> dict:
    cfg = payload.get("router_config", {}) or {}
    mode = str(cfg.get("router_control_mode", "unknown"))
    metric = str(cfg.get("uncertainty_metric_name", "unknown"))
    if mode == "farmass_log_contrast":
        method = "farmass_log_contrast"
        threshold = float(cfg.get("h_hi", math.nan))
        trigger_mode = str(cfg.get("trigger_mode", "onset"))
    elif mode == "metric_window_total_variation":
        method = "baseline_wtv"
        threshold = float(cfg.get("score_threshold", math.nan))
        trigger_mode = ""
    else:
        method = mode
        threshold = float(cfg.get("score_threshold", math.nan))
        trigger_mode = ""
    return {
        "run_name": run_name,
        "method": method,
        "metric": metric,
        "threshold": threshold,
        "trigger_mode": trigger_mode,
        "short_window": cfg.get("short_window", ""),
        "long_window": cfg.get("long_window", ""),
    }


def summarize_run(root: Path, run_name: str) -> list[dict]:
    rows = []
    for suite in SUITES:
        files = sorted((root / suite / run_name).glob("**/task*_trial*.pt"))
        if not files:
            continue
        payloads = []
        for path in files:
            try:
                payload = torch_load(path)
            except Exception as exc:
                print(f"Skipping {path}: {type(exc).__name__}: {exc}")
                continue
            payloads.append((path, payload))
        if not payloads:
            continue
        desc = describe_run(payloads[0][1], run_name)
        rows.append(
            {
                "suite": suite,
                **desc,
                "episodes": len(payloads),
                "success_rate": mean_or_nan([p.get("success", 0) for _, p in payloads]),
                "cot_ratio": mean_or_nan(
                    [
                        np.mean(to_numpy(p.get("used_deepthink_series")))
                        for _, p in payloads
                        if to_numpy(p.get("used_deepthink_series")).size
                    ]
                ),
                "num_fires_mean": mean_or_nan(
                    [
                        np.sum(to_numpy(p.get("farmass_log_fired_series")))
                        if desc["method"] == "farmass_log_contrast"
                        else np.sum(to_numpy(p.get("uncertain_decision_series")))
                        for _, p in payloads
                    ]
                ),
                "mean_openvla_latency": mean_or_nan(
                    [
                        np.mean(to_numpy(p.get("openvla_inference_times")))
                        for _, p in payloads
                        if to_numpy(p.get("openvla_inference_times")).size
                    ]
                ),
                "mean_deepthink_latency": mean_or_nan(
                    [
                        np.mean(to_numpy(p.get("deepthink_inference_times"))[to_numpy(p.get("deepthink_inference_times")) > 0])
                        for _, p in payloads
                        if np.any(to_numpy(p.get("deepthink_inference_times")) > 0)
                    ]
                ),
            }
        )
    return rows


def fire_arrays_for_run(episodes, method: str) -> dict[Path, np.ndarray]:
    fires_by_path = {}
    for ep in episodes:
        payload = torch_load(ep.path)
        if method == "farmass_log_contrast":
            fires = to_numpy(payload.get("farmass_log_fired_series")).astype(bool)
        else:
            fires = to_numpy(payload.get("uncertain_decision_series")).astype(bool)
        if len(fires) < len(ep.actions):
            padded = np.zeros(len(ep.actions), dtype=bool)
            padded[: len(fires)] = fires
            fires = padded
        fires_by_path[ep.path] = fires[: len(ep.actions)]
    return fires_by_path


def critical_metrics_for_run(root: Path, run_name: str) -> list[dict]:
    episodes, skipped = load_episodes(root, run_name)
    if not episodes:
        return []
    labels, _infos, _clip_len, _metadata = label_all(
        episodes,
        warmup=10,
        clip_candidates=(25, 20, 15, 10, 7, 5),
    )
    payload = torch_load(episodes[0].path)
    desc = describe_run(payload, run_name)
    fires_by_path = fire_arrays_for_run(episodes, desc["method"])
    rows = []
    for split in ("all", "success", "failure"):
        metrics = evaluate_fire_set(episodes, labels, fires_by_path, split)
        for suite in SUITES:
            suite_eps = [ep for ep in episodes if ep.suite == suite]
            if not suite_eps:
                continue
            suite_labels = {ep.path: labels[ep.path] for ep in suite_eps}
            suite_fires = {ep.path: fires_by_path[ep.path] for ep in suite_eps}
            suite_metrics = evaluate_fire_set(suite_eps, suite_labels, suite_fires, split)
            rows.append(
                {
                    "suite": suite,
                    **desc,
                    "split": split,
                    **suite_metrics,
                }
            )
        rows.append({"suite": "ALL", **desc, "split": split, **metrics})
    if skipped:
        print(f"{run_name}: skipped {len(skipped)} files during critical labeling")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, suite in zip(axes, SUITES):
        suite_rows = [r for r in rows if r["suite"] == suite]
        for method, marker in (("baseline_wtv", "s"), ("farmass_log_contrast", "o")):
            method_rows = sorted(
                [r for r in suite_rows if r["method"] == method],
                key=lambda x: (float(x["cot_ratio"]), float(x["threshold"])),
            )
            if not method_rows:
                continue
            ax.plot(
                [r["cot_ratio"] for r in method_rows],
                [r["success_rate"] for r in method_rows],
                marker=marker,
                label=method,
            )
        ax.set_title(suite)
        ax.set_xlabel("CoT ratio")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Success rate")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = output_dir / "logcontrast_success_vs_cot_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown(path: Path, summary_rows: list[dict], critical_rows: list[dict]) -> None:
    lines = ["# Log-Contrast Experiment Summary", ""]
    if not summary_rows:
        lines.append("No completed `.pt` files found yet.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "| suite | method | threshold | episodes | success | cot_ratio | openvla latency | deepthink latency |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary_rows, key=lambda r: (r["suite"], r["method"], float(r["threshold"]))):
        lines.append(
            f"| {row['suite']} | {row['method']} | {float(row['threshold']):.3f} | "
            f"{int(row['episodes'])} | {float(row['success_rate']):.3f} | "
            f"{float(row['cot_ratio']):.3f} | {float(row['mean_openvla_latency']):.3f} | "
            f"{float(row['mean_deepthink_latency']):.3f} |"
        )
    if critical_rows:
        lines.extend(["", "## Critical Segment Coverage", ""])
        lines.extend(
            [
                "| suite | method | threshold | split | coverage | waste | fires/segment |",
                "|---|---|---:|---|---:|---:|---:|",
            ]
        )
        for row in sorted(
            [r for r in critical_rows if r["suite"] != "ALL" and r["split"] == "all"],
            key=lambda r: (r["suite"], r["method"], float(r["threshold"])),
        ):
            lines.append(
                f"| {row['suite']} | {row['method']} | {float(row['threshold']):.3f} | "
                f"{row['split']} | {float(row['segment_coverage']):.3f} | "
                f"{float(row['waste_rate']):.3f} | {float(row['fires_per_segment']):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT / "rollouts_logcontrast"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-critical", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    run_names = discover_run_names(root)
    summary_rows: list[dict] = []
    critical_rows: list[dict] = []
    for run_name in run_names:
        summary_rows.extend(summarize_run(root, run_name))
        if not args.skip_critical:
            critical_rows.extend(critical_metrics_for_run(root, run_name))

    summary_csv = output_dir / "logcontrast_summary.csv"
    critical_csv = output_dir / "logcontrast_critical_segments.csv"
    md_path = output_dir / "logcontrast_summary.md"
    write_csv(summary_csv, summary_rows)
    write_csv(critical_csv, critical_rows)
    fig_path = plot_summary(output_dir, summary_rows)
    write_markdown(md_path, summary_rows, critical_rows)

    print(f"runs={len(run_names)} summary_rows={len(summary_rows)} critical_rows={len(critical_rows)}")
    print(f"Saved {summary_csv}")
    print(f"Saved {critical_csv}")
    if fig_path:
        print(f"Saved {fig_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
