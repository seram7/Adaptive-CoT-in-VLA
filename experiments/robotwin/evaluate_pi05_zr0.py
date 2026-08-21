#!/usr/bin/env python3
"""Evaluate Motus PI0.5 alone or route uncertain chunks to ZR-0."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from websockets.exceptions import ConnectionClosed

from experiments.robotwin.pi05_task_registry import get_pi05_task
from experiments.robotwin.robotwin_env import RoboTwinHarness
from experiments.robotwin.router_metrics import windowed_total_variation


THRESHOLD_MODE = "sampling_farmass_wtv"
STEP_THRESHOLD_MODE = "sampling_farmass_replan_max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--pi05-checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-host", default="127.0.0.1")
    parser.add_argument("--pi05-port", type=int, default=18100)
    parser.add_argument("--zr0-root", type=Path)
    parser.add_argument("--zr0-host", default="127.0.0.1")
    parser.add_argument("--zr0-port", type=int, default=18000)
    parser.add_argument(
        "--zr0-inference-mode",
        choices=("direct_action", "subtask_then_action"),
        default="direct_action",
        help="Label only: must match how the running ZR-0 server was started.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", choices=("demo_clean", "demo_randomized"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("baseline", "pilot", "fixed", "random", THRESHOLD_MODE, STEP_THRESHOLD_MODE),
        required=True,
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--cot-ratio", type=float, default=0.2)
    parser.add_argument("--tv-window", type=int, default=5)
    parser.add_argument(
        "--cooldown-queries",
        type=int,
        default=0,
        help="Minimum queries since the last zr0 call before it can trigger again (threshold mode only).",
    )
    parser.add_argument(
        "--cooldown-steps",
        type=int,
        default=0,
        help="Minimum executed simulator actions since the last zr0 call (threshold modes only).",
    )
    parser.add_argument(
        "--force-first-query",
        action="store_true",
        help="Always route the episode's first query to zr0 (random/threshold modes; fixed already does this).",
    )
    parser.add_argument("--action-steps", type=int, default=32)
    parser.add_argument(
        "--replan-stride",
        type=int,
        help="Execute at most this many actions before observing and querying again.",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--router-seed", type=int, default=2025)
    parser.add_argument("--clear-cache-every", type=int, default=1)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_state(path: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"revision": revision, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": repr(exc)}


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "sapien", "mplib", "websockets", "msgpack"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "machine_id": os.environ.get("MACHINE_ID", "MACHINE_B"),
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": packages,
        "repositories": {
            "adaptive_cot": _git_state(Path(__file__).resolve().parents[2]),
            "robotwin": _git_state(args.robotwin_root),
            "openpi": _git_state(args.openpi_root),
            "zr0": _git_state(args.zr0_root) if args.zr0_root else None,
        },
        "checkpoints": {
            "pi05": str(args.pi05_checkpoint.expanduser().resolve()),
            "zr0": os.environ.get("ZR0_CHECKPOINT"),
        },
        "pi05_server": {"host": args.pi05_host, "port": args.pi05_port},
        "zr0_server": {
            "host": args.zr0_host,
            "port": args.zr0_port,
            "inference_mode": args.zr0_inference_mode,
        },
    }


def load_manifest(path: Path, task: str, task_config: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task or payload.get("task_config") != task_config:
        raise ValueError(f"Manifest {path} does not match {task}/{task_config}")
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"Manifest {path} has no episodes")
    return episodes


def load_threshold(args: argparse.Namespace) -> float | None:
    if args.mode not in {THRESHOLD_MODE, STEP_THRESHOLD_MODE}:
        return args.threshold
    if args.threshold is not None:
        return float(args.threshold)
    if args.thresholds_json is None:
        raise ValueError(f"{THRESHOLD_MODE} requires --threshold or --thresholds-json")
    payload = json.loads(args.thresholds_json.read_text(encoding="utf-8"))
    return float(payload[args.task_config][args.task][args.mode])


def _client_class(openpi_root: Path):
    client_source = openpi_root.expanduser().resolve() / "packages" / "openpi-client" / "src"
    if not client_source.is_dir():
        raise FileNotFoundError(client_source)
    if str(client_source) not in sys.path:
        sys.path.insert(0, str(client_source))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy


class PI05Client:
    def __init__(self, openpi_root: Path, host: str, port: int, action_steps: int) -> None:
        self.client_class = _client_class(openpi_root)
        self.host = host
        self.port = port
        self.client = self.client_class(host, port)
        self.action_steps = action_steps

    def _infer_with_reconnect(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            return self.client.infer(payload), False
        except ConnectionClosed:
            # Randomized RoboTwin scene/planner initialization can exceed the
            # websocket keepalive interval before the first query. Reconnect and
            # retry exactly once; no simulator action has been executed yet.
            self.client = self.client_class(self.host, self.port)
            return self.client.infer(payload), True

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        *,
        uncertainty_seed: int,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        obs = observation["observation"]
        payload = {
            "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "images": {
                "cam_high": np.ascontiguousarray(np.transpose(obs["head_camera"]["rgb"], (2, 0, 1))),
                "cam_left_wrist": np.ascontiguousarray(
                    np.transpose(obs["left_camera"]["rgb"], (2, 0, 1))
                ),
                "cam_right_wrist": np.ascontiguousarray(
                    np.transpose(obs["right_camera"]["rgb"], (2, 0, 1))
                ),
            },
            "prompt": instruction,
            "uncertainty_seed": int(uncertainty_seed),
        }
        started = time.perf_counter()
        result, reconnected = self._infer_with_reconnect(payload)
        elapsed = time.perf_counter() - started
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"PI0.5 returned invalid actions {actions.shape}")
        if actions.shape[0] < self.action_steps:
            raise ValueError(f"PI0.5 returned {actions.shape[0]} steps, need {self.action_steps}")
        metrics = {
            "sampling_farmass_mean": result.get("sampling_farmass_mean"),
            "sampling_far_mass_mean": result.get("sampling_far_mass_mean"),
            "sampling_mean_far_distance_mean": result.get("sampling_mean_far_distance_mean"),
            "policy_timing": result.get("policy_timing", {}),
            "server_timing": result.get("server_timing", {}),
            "websocket_reconnected": reconnected,
        }
        far_mass_uncertainty = result.get("far_mass_uncertainty")
        if far_mass_uncertainty is not None:
            uncertainty = np.asarray(far_mass_uncertainty, dtype=np.float32)
            if uncertainty.ndim != 2 or uncertainty.shape[0] < self.action_steps:
                raise ValueError(
                    "PI0.5 returned invalid per-step Far-Mass uncertainty "
                    f"{uncertainty.shape}"
                )
            per_step = np.mean(uncertainty[: self.action_steps], axis=-1)
            metrics["sampling_farmass_per_step"] = per_step.tolist()
            metrics["sampling_farmass_step0"] = float(per_step[0])
        else:
            metrics["sampling_farmass_per_step"] = None
            metrics["sampling_farmass_step0"] = None
        return actions[: self.action_steps], elapsed, metrics


class ZR0Client:
    def __init__(self, root: Path, host: str, port: int, action_steps: int) -> None:
        root = root.expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from utils.websocket_client_policy import WebsocketClientPolicy

        self.client_class = WebsocketClientPolicy
        self.host = host
        self.port = port
        self.client = self.client_class(host, port)
        self.action_steps = action_steps

    def _infer_with_reconnect(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        try:
            return self.client.infer(payload), False
        except ConnectionClosed:
            self.client = self.client_class(self.host, self.port)
            return self.client.infer(payload), True

    def act(self, observation: dict[str, Any], instruction: str):
        obs = observation["observation"]
        payload = {
            "task": instruction,
            "observation.state": np.asarray(
                observation["joint_action"]["vector"], dtype=np.float32
            ),
            "n_action_steps": self.action_steps,
            "observation.images.cam_high": np.ascontiguousarray(obs["head_camera"]["rgb"]),
            "observation.images.cam_left_wrist": np.ascontiguousarray(obs["left_camera"]["rgb"]),
            "observation.images.cam_right_wrist": np.ascontiguousarray(obs["right_camera"]["rgb"]),
        }
        started = time.perf_counter()
        result, reconnected = self._infer_with_reconnect(payload)
        elapsed = time.perf_counter() - started
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 14 or actions.shape[0] == 0:
            raise ValueError(f"ZR-0 returned invalid actions {actions.shape}")
        # ZR-0's action_expert has a fixed native action_horizon (16 for this
        # checkpoint); it silently returns min(n_action_steps, action_horizon)
        # rather than padding, so unlike PI0.5 we cannot require >= action_steps.
        timing = dict(result.get("server_timing", {}))
        timing["websocket_reconnected"] = reconnected
        return actions[: self.action_steps], elapsed, timing


@dataclass
class SamplingFarMassRouter:
    mode: str
    cot_ratio: float
    threshold: float | None
    tv_window: int
    seed: int
    cooldown_queries: int = 0
    cooldown_steps: int = 0
    force_first_query: bool = False
    history: list[float] = field(default_factory=list)
    query_index: int = 0
    queries_since_zr0: int | None = None
    steps_since_zr0: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.cot_ratio <= 1.0:
            raise ValueError("cot_ratio must be in [0, 1]")
        if self.mode == THRESHOLD_MODE and self.threshold is None:
            raise ValueError(f"{THRESHOLD_MODE} requires a threshold")
        if self.mode == STEP_THRESHOLD_MODE and self.threshold is None:
            raise ValueError(f"{STEP_THRESHOLD_MODE} requires a threshold")
        if self.cooldown_queries < 0:
            raise ValueError("cooldown_queries must be >= 0")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        self.rng = np.random.default_rng(self.seed)

    def update(
        self,
        value: float | None,
        *,
        step_uncertainty: float | None = None,
    ) -> dict[str, Any]:
        if value is not None and np.isfinite(float(value)):
            self.history.append(float(value))
        wtv = windowed_total_variation(self.history, self.tv_window)
        in_cooldown = False
        if self.mode in {"baseline", "pilot"}:
            use_zr0 = False
        elif self.mode == "fixed":
            interval = max(1, round(1.0 / self.cot_ratio)) if self.cot_ratio > 0 else 10**18
            use_zr0 = self.query_index % interval == 0
        elif self.mode == "random":
            use_zr0 = bool(self.rng.random() < self.cot_ratio)
        elif self.mode in {THRESHOLD_MODE, STEP_THRESHOLD_MODE}:
            if self.mode == THRESHOLD_MODE:
                raw_trigger = len(self.history) >= 2 and wtv >= float(self.threshold)
            else:
                raw_trigger = (
                    step_uncertainty is not None
                    and np.isfinite(float(step_uncertainty))
                    and float(step_uncertainty) >= float(self.threshold)
                )
            # A slowly-decaying windowed-TV signal tends to stay above threshold
            # for several consecutive queries once it crosses, which bursts zr0
            # invocations back-to-back. cooldown_queries enforces a minimum gap
            # (in queries) since the last zr0 call, regardless of the metric.
            query_cooldown = (
                self.cooldown_queries > 0
                and self.queries_since_zr0 is not None
                and self.queries_since_zr0 < self.cooldown_queries
            )
            step_cooldown = (
                self.cooldown_steps > 0
                and self.steps_since_zr0 is not None
                and self.steps_since_zr0 < self.cooldown_steps
            )
            in_cooldown = query_cooldown or step_cooldown
            use_zr0 = raw_trigger and not in_cooldown
        else:
            raise ValueError(f"Unknown PI0.5 router mode {self.mode}")

        # WTV needs >=2 history points to fire at all, so THRESHOLD_MODE can
        # structurally never route the very first query to zr0; "random" only
        # does so with probability cot_ratio. force_first_query gives both the
        # same guaranteed first-chunk zr0 call that "fixed" already gets for
        # free (query_index % interval == 0 at index 0), so the three modes
        # are compared on equal footing instead of "fixed" alone benefiting
        # from always handling the (often outcome-deciding) opening move.
        if self.force_first_query and self.query_index == 0 and self.mode in {
            "random",
            THRESHOLD_MODE,
            STEP_THRESHOLD_MODE,
        }:
            use_zr0 = True
            in_cooldown = False

        if use_zr0:
            self.queries_since_zr0 = 0
            self.steps_since_zr0 = 0
        elif self.queries_since_zr0 is not None:
            self.queries_since_zr0 += 1

        result = {
            "query_index": self.query_index,
            "sampling_farmass_wtv": wtv,
            STEP_THRESHOLD_MODE: step_uncertainty,
            "sampling_farmass_history_count": len(self.history),
            "use_zr0": use_zr0,
            "suppressed_by_cooldown": in_cooldown,
        }
        self.query_index += 1
        return result

    def record_execution(self, executed_steps: int) -> None:
        if executed_steps < 0:
            raise ValueError("executed_steps must be non-negative")
        if self.steps_since_zr0 is not None:
            self.steps_since_zr0 += int(executed_steps)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.action_steps <= 0:
        raise ValueError("--action-steps must be positive")
    if args.replan_stride is not None and not 1 <= args.replan_stride <= args.action_steps:
        raise ValueError("--replan-stride must be in [1, action_steps]")
    task_spec = get_pi05_task(args.task)
    threshold = load_threshold(args)
    provenance = collect_provenance(args)
    episodes = load_manifest(args.manifest, args.task, args.task_config)
    stop = len(episodes) if args.episodes is None else min(
        len(episodes), args.episode_start + args.episodes
    )
    selected_episodes = list(enumerate(episodes))[args.episode_start:stop]
    if not selected_episodes:
        raise ValueError("No episodes selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    harness = RoboTwinHarness(
        args.robotwin_root,
        args.task,
        args.task_config,
        step_limit=task_spec.step_limit,
        instruction_type=args.instruction_type,
    )
    pi05 = PI05Client(args.openpi_root, args.pi05_host, args.pi05_port, args.action_steps)
    zr0 = None
    if args.mode not in {"baseline", "pilot"}:
        if args.zr0_root is None:
            raise ValueError(f"{args.mode} requires --zr0-root")
        zr0 = ZR0Client(args.zr0_root, args.zr0_host, args.zr0_port, args.action_steps)

    completed: list[dict[str, Any]] = []
    for episode_id, episode in selected_episodes:
        episode_path = args.output_dir / f"episode_{episode_id:03d}.json"
        if episode_path.exists():
            print(f"Skipping existing {episode_path}", flush=True)
            completed.append(json.loads(episode_path.read_text(encoding="utf-8")))
            continue

        seed = int(episode["seed"])
        task, instruction = harness.start_episode(episode_id, seed, episode.get("info", {}))
        router = SamplingFarMassRouter(
            args.mode,
            cot_ratio=args.cot_ratio,
            threshold=threshold,
            tv_window=args.tv_window,
            seed=args.router_seed + seed,
            cooldown_queries=args.cooldown_queries,
            cooldown_steps=args.cooldown_steps,
            force_first_query=args.force_first_query,
        )
        records: list[dict[str, Any]] = []
        episode_started = time.perf_counter()
        try:
            while task.take_action_cnt < task.step_lim and not task.eval_success:
                observation = task.get_obs()
                fast_actions, pi05_seconds, pi05_metrics = pi05.act(
                    observation,
                    instruction,
                    uncertainty_seed=(seed * 10000 + len(records)),
                )
                per_step = pi05_metrics["sampling_farmass_per_step"]
                routing_uncertainty = None
                if per_step is not None:
                    routing_horizon = args.replan_stride or args.action_steps
                    routing_uncertainty = float(np.max(per_step[:routing_horizon]))
                pi05_metrics[STEP_THRESHOLD_MODE] = routing_uncertainty
                decision = router.update(
                    pi05_metrics["sampling_farmass_mean"],
                    step_uncertainty=routing_uncertainty,
                )
                chosen_actions = fast_actions
                selected_policy = "pi05"
                zr0_seconds = None
                zr0_timing = None
                if decision["use_zr0"]:
                    assert zr0 is not None
                    chosen_actions, zr0_seconds, zr0_timing = zr0.act(observation, instruction)
                    selected_policy = f"zr0_{args.zr0_inference_mode}"
                if args.replan_stride is not None:
                    chosen_actions = chosen_actions[: args.replan_stride]
                executed, step_latencies = harness.run_action_chunk_with_timing(
                    task, chosen_actions
                )
                router.record_execution(executed)
                records.append(
                    {
                        **decision,
                        **pi05_metrics,
                        "selected_policy": selected_policy,
                        "sim_step_start": int(task.take_action_cnt - executed),
                        "executed_actions": int(executed),
                        "pi05_seconds": pi05_seconds,
                        "zr0_seconds": zr0_seconds,
                        "zr0_server_timing": zr0_timing,
                        "step_latencies_seconds": step_latencies,
                        "step_latency_mean_seconds": (
                            float(np.mean(step_latencies)) if step_latencies else None
                        ),
                    }
                )
                print(
                    f"{args.task} {args.task_config} ep={episode_id} query={len(records)} "
                    f"step={task.take_action_cnt}/{task.step_lim} policy={selected_policy} "
                    f"success={task.eval_success}",
                    flush=True,
                )

            zr0_queries = sum(r["selected_policy"] != "pi05" for r in records)
            result = {
                "task": args.task,
                "task_config": args.task_config,
                "episode_id": episode_id,
                "seed": seed,
                "instruction": instruction,
                "success": bool(task.eval_success),
                "sim_steps": int(task.take_action_cnt),
                "policy_queries": len(records),
                "zr0_queries": zr0_queries,
                "realized_cot_ratio": zr0_queries / max(1, len(records)),
                "mode": args.mode,
                "configured_cot_ratio": args.cot_ratio,
                "threshold": threshold,
                "tv_window": args.tv_window,
                "cooldown_queries": args.cooldown_queries,
                "cooldown_steps": args.cooldown_steps,
                "force_first_query": args.force_first_query,
                "action_steps": args.action_steps,
                "replan_stride": args.replan_stride,
                "failure_category": None if task.eval_success else "task_failure",
                "wall_seconds": time.perf_counter() - episode_started,
                "provenance": provenance,
                "records": records,
            }
            write_json_atomic(episode_path, result)
            completed.append(result)
        finally:
            harness.close_episode(
                task,
                clear_cache=(episode_id + 1) % max(1, args.clear_cache_every) == 0,
            )

    all_records = [record for item in completed for record in item["records"]]
    zr0_latencies = [r["zr0_seconds"] for r in all_records if r["zr0_seconds"] is not None]
    step_latencies_all = [
        latency
        for record in all_records
        for latency in record.get("step_latencies_seconds", [])
    ]
    summary = {
        "task": args.task,
        "task_config": args.task_config,
        "mode": args.mode,
        "episodes": len(completed),
        "successes": sum(bool(item["success"]) for item in completed),
        "success_rate": float(np.mean([item["success"] for item in completed])),
        "mean_cot_ratio": float(np.mean([item["realized_cot_ratio"] for item in completed])),
        "threshold": threshold,
        "tv_window": args.tv_window,
        "cooldown_queries": args.cooldown_queries,
        "cooldown_steps": args.cooldown_steps,
        "force_first_query": args.force_first_query,
        "action_steps": args.action_steps,
        "replan_stride": args.replan_stride,
        "latency_seconds": {
            "pi05_network_mean": float(np.mean([r["pi05_seconds"] for r in all_records])),
            "zr0_network_mean": float(np.mean(zr0_latencies)) if zr0_latencies else None,
            "episode_wall_mean": (
                float(np.mean([item["wall_seconds"] for item in completed if "wall_seconds" in item]))
                if any("wall_seconds" in item for item in completed)
                else None
            ),
            "sim_step_seconds": {
                "mean": float(np.mean(step_latencies_all)) if step_latencies_all else None,
                "p50": float(np.percentile(step_latencies_all, 50)) if step_latencies_all else None,
                "p95": float(np.percentile(step_latencies_all, 95)) if step_latencies_all else None,
                "count": len(step_latencies_all),
            },
        },
        "provenance": provenance,
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
