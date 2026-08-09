#!/usr/bin/env python
"""Analyze OpenVLA phase-transition critical segments from saved rollout .pt files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CANDIDATES = ("C0_wtv_m", "C1_wtv_log_m", "C2_short_long_log", "C3_schmitt", "C4_far_mass_profile")
TARGET_RATES = (0.05, 0.10, 0.20, 0.30)
KINDS = ("S1", "S2", "S3", "S4")


@dataclass
class Segment:
    kind: str
    start: int
    end: int
    event: int | None = None


@dataclass
class Episode:
    suite: str
    task_id: int
    trial_id: int
    success: bool
    path: Path
    step_ids: np.ndarray
    actions: np.ndarray
    ee_positions: np.ndarray
    proprio_states: np.ndarray
    far_mass_slots: np.ndarray
    peak_separation_slots: np.ndarray
    far_mass_x_peak_slots: np.ndarray


def torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_episodes(root: Path, run_name: str) -> tuple[list[Episode], list[str]]:
    episodes: list[Episode] = []
    skipped: list[str] = []
    for suite in SUITES:
        for path in sorted((root / suite / run_name).glob("**/task*_trial*.pt")):
            try:
                payload = torch_load(path)
            except Exception as exc:
                skipped.append(f"{path}: load failed: {type(exc).__name__}: {exc}")
                continue

            actions = to_numpy(payload.get("executed_actions")).astype(np.float32)
            ee_positions = to_numpy(payload.get("ee_positions")).astype(np.float32)
            proprio_states = to_numpy(payload.get("proprio_states")).astype(np.float32)
            far_mass = to_numpy(payload.get("far_mass_per_slot_series")).astype(np.float32)
            peak_sep = to_numpy(payload.get("peak_separation_per_slot_series")).astype(np.float32)
            far_x_peak = to_numpy(
                payload.get("far_mass_x_peak_separation_per_slot_series")
            ).astype(np.float32)

            if actions.ndim != 2 or actions.shape[-1] != 7:
                skipped.append(f"{path}: missing executed_actions with shape (T, 7)")
                continue
            if ee_positions.ndim != 2 or ee_positions.shape[-1] != 3:
                skipped.append(f"{path}: missing ee_positions with shape (T, 3)")
                continue
            if far_x_peak.ndim != 2 or far_x_peak.shape[-1] != 7:
                skipped.append(
                    f"{path}: missing far_mass_x_peak_separation_per_slot_series with shape (T, 7)"
                )
                continue
            if far_mass.ndim != 2 or far_mass.shape[-1] != 7:
                skipped.append(f"{path}: missing far_mass_per_slot_series with shape (T, 7)")
                continue
            n = min(len(actions), len(ee_positions), len(far_x_peak), len(far_mass))
            if n == 0:
                skipped.append(f"{path}: empty episode")
                continue

            episodes.append(
                Episode(
                    suite=str(payload.get("task_suite", suite)),
                    task_id=int(payload.get("task_id", -1)),
                    trial_id=int(payload.get("trial_id", -1)),
                    success=bool(int(payload.get("success", 0))),
                    path=path,
                    step_ids=to_numpy(payload.get("step_ids"))[:n].astype(np.int32),
                    actions=actions[:n],
                    ee_positions=ee_positions[:n],
                    proprio_states=proprio_states[:n],
                    far_mass_slots=far_mass[:n],
                    peak_separation_slots=peak_sep[:n],
                    far_mass_x_peak_slots=far_x_peak[:n],
                )
            )
    return episodes, skipped


def smooth_centered(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or window <= 1:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def gripper_events(actions: np.ndarray, merge_window: int = 3) -> tuple[list[int], list[int], np.ndarray]:
    g = np.where(actions[:, -1] >= 0, 1, -1).astype(np.int8)
    raw = np.flatnonzero(g[1:] != g[:-1]) + 1
    merged: list[int] = []
    for idx in raw:
        if merged and idx - merged[-1] <= merge_window:
            continue
        merged.append(int(idx))
    closes = [idx for idx in merged if g[idx] == 1 and g[idx - 1] == -1]
    opens = [idx for idx in merged if g[idx] == -1 and g[idx - 1] == 1]
    return closes, opens, g


def last_fast_start(speed: np.ndarray, event: int, clip_len: int) -> int:
    if event <= 0:
        return 0
    threshold = float(np.percentile(speed, 60))
    candidates = np.flatnonzero(speed[:event] > threshold)
    start = int(candidates[-1]) if len(candidates) else max(0, event - clip_len)
    return max(start, event - clip_len)


def normalized_dtw_prefix_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, m = len(a), len(b)
    dp = np.full((n, m), np.inf, dtype=np.float64)
    steps = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            cost = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                dp[i, j] = cost
                steps[i, j] = 1.0
                continue
            choices = []
            if i > 0:
                choices.append((dp[i - 1, j], steps[i - 1, j]))
            if j > 0:
                choices.append((dp[i, j - 1], steps[i, j - 1]))
            if i > 0 and j > 0:
                choices.append((dp[i - 1, j - 1], steps[i - 1, j - 1]))
            prev_cost, prev_steps = min(choices, key=lambda x: x[0])
            dp[i, j] = prev_cost + cost
            steps[i, j] = prev_steps + 1.0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = min(i, m - 1)
        out[i] = dp[i, j] / max(steps[i, j], 1.0)
    return out


def normalized_dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(normalized_dtw_prefix_distances(a, b)[-1])


def build_success_refs(episodes: list[Episode]) -> dict[tuple[str, int], dict[str, object]]:
    grouped: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for ep in episodes:
        if ep.success:
            grouped[(ep.suite, ep.task_id)].append(ep)
    refs: dict[tuple[str, int], dict[str, object]] = {}
    for key, eps in grouped.items():
        if len(eps) < 3:
            refs[key] = {"episodes": eps, "threshold": None, "pair_distances": []}
            continue
        distances = []
        for i in range(len(eps)):
            for j in range(i + 1, len(eps)):
                distances.append(normalized_dtw_distance(eps[i].ee_positions, eps[j].ee_positions))
        refs[key] = {
            "episodes": eps,
            "threshold": float(np.percentile(distances, 95)) if distances else None,
            "pair_distances": distances,
        }
    return refs


def label_episode(
    ep: Episode,
    refs: dict[tuple[str, int], dict[str, object]],
    warmup: int,
    clip_len: int,
) -> tuple[list[Segment], dict[str, object]]:
    speed = smooth_centered(np.linalg.norm(ep.actions[:, :3], axis=1), window=5)
    closes, opens, _ = gripper_events(ep.actions)
    g_close = closes[0] if closes else None
    g_open = next((x for x in opens if g_close is None or x > g_close), None)

    segments: list[Segment] = []
    if g_close is not None:
        s1_start = last_fast_start(speed, g_close, clip_len)
        segments.append(Segment("S1", s1_start, g_close, event=g_close))
        segments.append(Segment("S2", g_close, min(len(speed) - 1, g_close + 10), event=g_close))
    if g_open is not None:
        s3_start = last_fast_start(speed, g_open, clip_len)
        segments.append(Segment("S3", s3_start, g_open, event=g_open))

    s4_status = "not_failure"
    if not ep.success:
        ref = refs.get((ep.suite, ep.task_id), {"episodes": [], "threshold": None})
        if ref["threshold"] is None or len(ref["episodes"]) < 3:
            s4_status = f"skipped_success_refs_{len(ref['episodes'])}"
        else:
            dists = []
            for success_ep in ref["episodes"]:
                dists.append(normalized_dtw_prefix_distances(ep.ee_positions, success_ep.ee_positions))
            min_dist = np.min(np.stack(dists, axis=0), axis=0)
            hits = np.flatnonzero(min_dist > float(ref["threshold"]))
            if len(hits):
                start = int(hits[0])
                segments.append(Segment("S4", start, min(len(speed) - 1, start + 15), event=start))
                s4_status = "labeled"
            else:
                s4_status = "no_divergence"

    clipped = []
    for seg in segments:
        start = max(int(seg.start), warmup)
        end = min(int(seg.end), len(speed) - 1)
        if end >= start:
            clipped.append(Segment(seg.kind, start, end, seg.event))

    info = {
        "g_close": g_close,
        "g_open": g_open,
        "num_closes": len(closes),
        "num_opens": len(opens),
        "s4_status": s4_status,
        "speed": speed,
    }
    return clipped, info


def label_all(
    episodes: list[Episode],
    warmup: int,
    clip_candidates: tuple[int, ...],
) -> tuple[dict[Path, list[Segment]], dict[Path, dict[str, object]], int, dict[str, object]]:
    refs = build_success_refs(episodes)
    best = None
    for clip_len in clip_candidates:
        labels = {}
        infos = {}
        critical = 0
        total = 0
        for ep in episodes:
            segs, info = label_episode(ep, refs, warmup=warmup, clip_len=clip_len)
            labels[ep.path] = segs
            infos[ep.path] = info
            mask = np.zeros(len(ep.actions), dtype=bool)
            for seg in segs:
                mask[seg.start : seg.end + 1] = True
            eligible = np.zeros(len(ep.actions), dtype=bool)
            eligible[warmup:] = True
            critical += int(np.sum(mask & eligible))
            total += int(np.sum(eligible))
        ratio = critical / total if total else 0.0
        candidate = (abs(ratio - 0.225), clip_len, ratio, labels, infos)
        if 0.15 <= ratio <= 0.30:
            best = candidate
            break
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, clip_len, ratio, labels, infos = best
    metadata = {"critical_ratio": ratio, "refs": refs}
    return labels, infos, clip_len, metadata


def causal_window_tv(series: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros(len(series), dtype=np.float64)
    for t in range(len(series)):
        arr = series[max(0, t - window + 1) : t + 1]
        if len(arr) >= 2:
            out[t] = float(np.mean(np.abs(np.diff(arr))))
    return out


def c2_short_long_log(m: np.ndarray) -> np.ndarray:
    z = np.log(np.maximum(m, 0.0) + 1e-6)
    out = np.zeros(len(z), dtype=np.float64)
    for t in range(9, len(z)):
        short = np.mean(z[t - 2 : t + 1])
        long = np.mean(z[t - 9 : t - 2])
        denom = max(float(np.std(z[t - 9 : t + 1])), 1e-3)
        out[t] = max(0.0, (short - long) / denom)
    return out


def c4_profile_distance(far_mass_slots: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    arm = np.maximum(far_mass_slots[:, :6].astype(np.float64), 0.0)
    out = np.zeros(len(arm), dtype=np.float64)
    ref = None
    for t, vec in enumerate(arm):
        norm = np.linalg.norm(vec)
        prof = vec / norm if norm > 1e-12 else np.zeros_like(vec)
        if ref is not None:
            denom = max(np.linalg.norm(prof) * np.linalg.norm(ref), 1e-12)
            out[t] = 1.0 - float(np.dot(prof, ref) / denom)
        ref = prof.copy() if ref is None else alpha * prof + (1.0 - alpha) * ref
    return np.maximum(out, 0.0)


def candidate_scores(ep: Episode) -> dict[str, np.ndarray]:
    m = np.mean(ep.far_mass_x_peak_slots[:, :6], axis=1).astype(np.float64)
    z = np.log(np.maximum(m, 0.0) + 1e-6)
    c2 = c2_short_long_log(m)
    return {
        "C0_wtv_m": causal_window_tv(m, window=5),
        "C1_wtv_log_m": causal_window_tv(z, window=5),
        "C2_short_long_log": c2,
        "C3_schmitt": c2,
        "C4_far_mass_profile": c4_profile_distance(ep.far_mass_slots),
    }


def schmitt_fires(score: np.ndarray, h_hi: float, warmup: int) -> np.ndarray:
    fires = np.zeros(len(score), dtype=bool)
    active = False
    h_lo = h_hi / 3.0
    for t, value in enumerate(score):
        if t < warmup:
            continue
        if active:
            if value <= h_lo:
                active = False
        elif value >= h_hi:
            fires[t] = True
            active = True
    return fires


def tune_schmitt_threshold(scores: list[np.ndarray], warmup: int, target_rate: float) -> float:
    vals = np.concatenate([s[warmup:] for s in scores if len(s) > warmup])
    if len(vals) == 0:
        return math.inf
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        return hi
    total = sum(max(0, len(s) - warmup) for s in scores)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        fires = sum(int(np.sum(schmitt_fires(s, mid, warmup))) for s in scores)
        rate = fires / total if total else 0.0
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return hi


def threshold_fires(score: np.ndarray, threshold: float, warmup: int) -> np.ndarray:
    fires = score >= threshold
    fires[:warmup] = False
    return fires


def segment_mask(length: int, segments: list[Segment]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for seg in segments:
        mask[seg.start : seg.end + 1] = True
    return mask


def evaluate_fire_set(episodes, labels, fires_by_path, split: str) -> dict[str, float]:
    selected = []
    for ep in episodes:
        if split == "success" and not ep.success:
            continue
        if split == "failure" and ep.success:
            continue
        selected.append(ep)

    all_segments = []
    segment_hits = []
    segment_fire_counts = []
    lead_s1 = []
    total_fires = 0
    wasted_fires = 0
    kind_total = Counter()
    kind_hit = Counter()

    for ep in selected:
        segs = labels[ep.path]
        fires = fires_by_path[ep.path]
        crit_mask = segment_mask(len(ep.actions), segs)
        total_fires += int(np.sum(fires))
        wasted_fires += int(np.sum(fires & ~crit_mask))
        for seg in segs:
            all_segments.append(seg)
            kind_total[seg.kind] += 1
            count = int(np.sum(fires[seg.start : seg.end + 1]))
            hit = count > 0
            segment_hits.append(hit)
            segment_fire_counts.append(count)
            if hit:
                kind_hit[seg.kind] += 1
                if seg.kind == "S1" and seg.event is not None:
                    first_fire = int(np.flatnonzero(fires[seg.start : seg.end + 1])[0] + seg.start)
                    lead_s1.append(max(0, int(seg.event) - first_fire))

    covered_counts = [c for c in segment_fire_counts if c > 0]
    result = {
        "num_episodes": float(len(selected)),
        "num_segments": float(len(all_segments)),
        "num_fires": float(total_fires),
        "segment_coverage": float(np.mean(segment_hits)) if segment_hits else math.nan,
        "waste_rate": float(wasted_fires / total_fires) if total_fires else math.nan,
        "fires_per_segment": float(np.mean(covered_counts)) if covered_counts else math.nan,
        "lead_within_S1": float(np.median(lead_s1)) if lead_s1 else math.nan,
    }
    for kind in KINDS:
        total = kind_total[kind]
        result[f"segment_coverage_{kind}"] = kind_hit[kind] / total if total else math.nan
        result[f"num_segments_{kind}"] = float(total)
    return result


def write_inventory(out_dir: Path, episodes, labels, infos, clip_len, critical_ratio, skipped):
    rows = []
    kind_counts = Counter()
    split_kind_counts = defaultdict(Counter)
    per_ep_counts = []
    for ep in episodes:
        segs = labels[ep.path]
        per_ep_counts.append(len(segs))
        for seg in segs:
            kind_counts[seg.kind] += 1
            split_kind_counts["success" if ep.success else "failure"][seg.kind] += 1
        rows.append(
            {
                "suite": ep.suite,
                "task_id": ep.task_id,
                "trial_id": ep.trial_id,
                "success": int(ep.success),
                "num_segments": len(segs),
                "segment_kinds": ";".join(seg.kind for seg in segs),
                "g_close": infos[ep.path]["g_close"],
                "g_open": infos[ep.path]["g_open"],
                "num_closes": infos[ep.path]["num_closes"],
                "num_opens": infos[ep.path]["num_opens"],
                "s4_status": infos[ep.path]["s4_status"],
                "path": str(ep.path),
            }
        )
    with (out_dir / "critical_segment_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Critical Segment Inventory",
        "",
        "## Format Check",
        "",
        "- Action is 7D: translation deltas in dims 0:3, rotation command in dims 3:6, gripper in dim 6.",
        "- `normalize_gripper_action` maps the last action dim from [0, 1] to [-1, +1] and binarizes with `np.sign`.",
        "- `invert_gripper_action` flips the sign; final executed LIBERO convention is treated as `+1=close`, `-1=open`.",
        "- `proprio_states` are stored as `[eef_pos(3), quat2axisangle(eef_quat)(3), robot0_gripper_qpos(2)]`.",
        "- `quat2axisangle` returns a unit axis scaled by the rotation angle in radians.",
        "- Segment speed uses a smoothed norm of executed action dims 0:3; `z_t` and S4 DTW use stored `ee_positions`.",
        "",
        "## Label Inventory",
        "",
        f"- Episodes loaded: {len(episodes)}",
        f"- Skipped files: {len(skipped)}",
        f"- Clip length selected: {clip_len}",
        f"- Critical step ratio after warmup exclusion: {critical_ratio:.3%}",
        f"- Segment count per episode: {dict(Counter(per_ep_counts))}",
        f"- Segment counts by kind: {dict(kind_counts)}",
        f"- Success segment counts: {dict(split_kind_counts['success'])}",
        f"- Failure segment counts: {dict(split_kind_counts['failure'])}",
    ]
    if skipped:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {item}" for item in skipped[:50])
    (out_dir / "critical_segment_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_examples(out_dir: Path, episodes, labels, infos):
    example_dir = out_dir / "critical_segment_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    colors = {"S1": "#4c78a8", "S2": "#f58518", "S3": "#54a24b", "S4": "#e45756"}
    for suite in SUITES:
        suite_eps = [ep for ep in episodes if ep.suite == suite]
        chosen = [ep for ep in suite_eps if ep.success][:1] + [ep for ep in suite_eps if not ep.success][:2]
        for ep in chosen:
            info = infos[ep.path]
            speed = np.asarray(info["speed"])
            z = ep.ee_positions[:, 2]
            _, _, g = gripper_events(ep.actions)
            far_mass = np.mean(ep.far_mass_slots[:, :6], axis=1)
            x = np.arange(len(speed))

            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            series = [(speed, "speed"), (z, "z"), (g, "gripper"), (far_mass, "far_mass arm mean")]
            for ax, (values, ylabel) in zip(axes, series):
                ax.plot(x, values, color="#222222", linewidth=1.2)
                ax.set_ylabel(ylabel)
                for seg in labels[ep.path]:
                    ax.axvspan(seg.start, seg.end, color=colors.get(seg.kind, "#999999"), alpha=0.22)
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("step")
            title = f"{suite} task{ep.task_id:02d} trial{ep.trial_id:02d} success={int(ep.success)}"
            axes[0].set_title(title)
            handles = [
                plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.22, label=k)
                for k in KINDS
            ]
            axes[0].legend(handles=handles, loc="upper right", ncol=4, fontsize=8)
            fig.tight_layout()
            fig.savefig(
                example_dir / f"{suite}_task{ep.task_id:02d}_trial{ep.trial_id:02d}_segments.png",
                dpi=160,
            )
            plt.close(fig)


def run_metrics(out_dir: Path, episodes, labels, warmup: int):
    scores_by_path = {ep.path: candidate_scores(ep) for ep in episodes}
    rows = []
    figure_data = defaultdict(lambda: defaultdict(dict))

    for suite in SUITES:
        suite_eps = [ep for ep in episodes if ep.suite == suite]
        if not suite_eps:
            continue
        total_eligible = sum(max(0, len(ep.actions) - warmup) for ep in suite_eps)
        for candidate in CANDIDATES:
            suite_scores = [scores_by_path[ep.path][candidate] for ep in suite_eps]
            for target_rate in TARGET_RATES:
                if candidate == "C3_schmitt":
                    threshold = tune_schmitt_threshold(suite_scores, warmup, target_rate)
                    fires_by_path = {
                        ep.path: schmitt_fires(scores_by_path[ep.path][candidate], threshold, warmup)
                        for ep in suite_eps
                    }
                else:
                    vals = np.concatenate(
                        [
                            scores_by_path[ep.path][candidate][warmup:]
                            for ep in suite_eps
                            if len(ep.actions) > warmup
                        ]
                    )
                    threshold = float(np.quantile(vals, 1.0 - target_rate)) if len(vals) else math.inf
                    fires_by_path = {
                        ep.path: threshold_fires(scores_by_path[ep.path][candidate], threshold, warmup)
                        for ep in suite_eps
                    }
                actual_rate = (
                    sum(int(np.sum(fires_by_path[ep.path])) for ep in suite_eps) / total_eligible
                    if total_eligible
                    else math.nan
                )
                for split in ("all", "success", "failure"):
                    metrics = evaluate_fire_set(suite_eps, labels, fires_by_path, split=split)
                    metrics["actual_firing_rate"] = actual_rate
                    metrics["threshold"] = threshold
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "suite": suite,
                                "candidate": candidate,
                                "target_firing_rate": target_rate,
                                "split": split,
                                "metric": metric,
                                "value": value,
                            }
                        )
                    if split == "all":
                        figure_data[suite][candidate][target_rate] = {
                            "segment_coverage": metrics["segment_coverage"],
                            "waste_rate": metrics["waste_rate"],
                        }

    csv_path = out_dir / "critical_segment_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["suite", "candidate", "target_firing_rate", "split", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    for suite, by_candidate in figure_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for candidate, by_rate in by_candidate.items():
            xs = sorted(by_rate)
            axes[0].plot(
                xs,
                [by_rate[x]["segment_coverage"] for x in xs],
                marker="o",
                label=candidate,
            )
            axes[1].plot(xs, [by_rate[x]["waste_rate"] for x in xs], marker="o", label=candidate)
        axes[0].set_title("Segment Coverage")
        axes[1].set_title("Waste Rate")
        for ax in axes:
            ax.set_xlabel("target firing rate")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("rate")
        axes[1].legend(loc="best", fontsize=8)
        fig.suptitle(suite)
        fig.tight_layout()
        fig.savefig(out_dir / f"{suite}_critical_segment_coverage_waste.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/scratch/seram7/Adaptive-CoT-in-VLA/rollouts_42_h100_new",
    )
    parser.add_argument(
        "--run-name",
        default="openvla_phase_transition_minimal_t0_9_tr0_4_v2",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/seram7/codes/Adaptive-CoT-in-VLA/analysis_phase_transition",
    )
    parser.add_argument("--long-window", type=int, default=10)
    parser.add_argument("--expected-per-suite", type=int, default=50)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes, skipped = load_episodes(root, args.run_name)
    counts = Counter(ep.suite for ep in episodes)
    print("Loaded episode counts:", dict(counts))
    if args.require_complete:
        missing = {
            suite: args.expected_per_suite - counts.get(suite, 0)
            for suite in SUITES
            if counts.get(suite, 0) < args.expected_per_suite
        }
        if missing:
            raise SystemExit(f"Not complete yet: missing {missing}")
    if not episodes:
        raise SystemExit("No usable episodes found.")

    labels, infos, clip_len, metadata = label_all(
        episodes,
        warmup=args.long_window,
        clip_candidates=(25, 20, 15, 10, 7, 5),
    )
    critical_ratio = float(metadata["critical_ratio"])
    write_inventory(out_dir, episodes, labels, infos, clip_len, critical_ratio, skipped)
    if critical_ratio > 0.40:
        raise SystemExit(
            f"Critical ratio is still too high after clipping ({critical_ratio:.3%}); "
            "stop before metric evaluation."
        )
    plot_examples(out_dir, episodes, labels, infos)
    run_metrics(out_dir, episodes, labels, warmup=args.long_window)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
