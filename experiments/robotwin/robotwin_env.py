"""Thin, deterministic wrapper around the official RoboTwin 2.0 evaluator."""

from __future__ import annotations

import importlib
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def _task_class(task_name: str):
    module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(module, task_name)
    except AttributeError as exc:
        raise RuntimeError(f"RoboTwin task class {task_name!r} is unavailable") from exc


class RoboTwinHarness:
    def __init__(
        self,
        root: str | Path,
        task_name: str,
        task_config: str,
        step_limit: int,
        instruction_type: str = "unseen",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        os.chdir(self.root)
        for path in (self.root, self.root / "description" / "utils"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        # Imports must happen after changing to the RoboTwin root because the
        # upstream project intentionally uses root-relative asset paths.
        from envs import CONFIGS_PATH
        from envs.utils.create_actor import UnStableError
        from generate_episode_instructions import generate_episode_descriptions

        self.UnStableError = UnStableError
        self.generate_episode_descriptions = generate_episode_descriptions
        self.task_name = task_name
        self.task_config = task_config
        self.step_limit = int(step_limit)
        self.instruction_type = instruction_type
        self._episode_task = None

        with (self.root / "task_config" / f"{task_config}.yml").open(
            "r", encoding="utf-8"
        ) as handle:
            args = yaml.safe_load(handle)
        args = dict(args)
        args.update(
            {
                "task_name": task_name,
                "task_config": task_config,
                "ckpt_setting": "openvla_zr0",
                "policy_name": "openvla_zr0",
                "eval_mode": True,
                "eval_video_log": False,
            }
        )

        configs_path = Path(CONFIGS_PATH)
        with (configs_path / "_embodiment_config.yml").open("r", encoding="utf-8") as handle:
            embodiments = yaml.safe_load(handle)
        with (configs_path / "_camera_config.yml").open("r", encoding="utf-8") as handle:
            cameras = yaml.safe_load(handle)

        head_camera_type = args["camera"]["head_camera_type"]
        args["head_camera_h"] = cameras[head_camera_type]["h"]
        args["head_camera_w"] = cameras[head_camera_type]["w"]

        embodiment = args["embodiment"]

        def embodiment_path(name: str) -> str:
            value = embodiments[name]["file_path"]
            if value is None:
                raise RuntimeError(f"No asset path configured for embodiment {name}")
            return value

        if len(embodiment) == 1:
            args["left_robot_file"] = embodiment_path(embodiment[0])
            args["right_robot_file"] = embodiment_path(embodiment[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment) == 3:
            args["left_robot_file"] = embodiment_path(embodiment[0])
            args["right_robot_file"] = embodiment_path(embodiment[1])
            args["embodiment_dis"] = embodiment[2]
            args["dual_arm_embodied"] = False
        else:
            raise RuntimeError(f"Unexpected embodiment config: {embodiment}")

        def load_robot_config(path: str) -> dict[str, Any]:
            with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)

        args["left_embodiment_config"] = load_robot_config(args["left_robot_file"])
        args["right_embodiment_config"] = load_robot_config(args["right_robot_file"])
        self.args = args

    def _new_task(self):
        return _task_class(self.task_name)()

    def find_feasible_seeds(
        self,
        count: int,
        *,
        start_seed: int = 100000,
        skip_seeds: Iterable[int] = (),
    ) -> list[dict[str, Any]]:
        """Run the official expert feasibility check and retain successful seeds."""

        skip = set(int(value) for value in skip_seeds)
        episodes: list[dict[str, Any]] = []
        candidate = int(start_seed)
        task = self._new_task()
        consecutive_errors = 0
        while len(episodes) < count:
            if candidate in skip:
                candidate += 1
                continue
            try:
                # Match start_episode exactly. RoboTwin seeds NumPy and Torch
                # internally, but intentionally leaves Python's random module
                # untouched even though randomized clutter uses it. Without
                # this seed, a manifest can accept a scene that is different
                # from the scene reconstructed during evaluation.
                random.seed(candidate)
                np.random.seed(candidate % (2**32))
                task.setup_demo(
                    now_ep_num=len(episodes), seed=candidate, is_test=True, **self.args
                )
                episode = task.play_once()
                feasible = bool(task.plan_success and task.check_success())
                consecutive_errors = 0
            except self.UnStableError:
                feasible = False
                episode = None
                consecutive_errors = 0
            except Exception as exc:
                print(f"Seed {candidate} failed expert check: {exc!r}", flush=True)
                traceback.print_exc()
                feasible = False
                episode = None
                consecutive_errors += 1
            finally:
                try:
                    # Frequent cache clearing is slower but avoids the known
                    # SAPIEN memory/renderer instability on A/H-series GPUs.
                    task.close_env(clear_cache=True)
                except Exception:
                    pass
            if consecutive_errors:
                # A setup failure can leave the upstream task with a partially
                # initialized Robot instance. Recreate it instead of repeatedly
                # calling Robot.reset() on missing planner attributes.
                task = self._new_task()
                if consecutive_errors >= 3:
                    raise RuntimeError(
                        "Three consecutive RoboTwin setup/expert exceptions; "
                        "aborting instead of scanning seeds indefinitely"
                    )
            if feasible and episode is not None:
                episodes.append({"seed": candidate, "info": episode.get("info", {})})
                print(
                    f"{self.task_name}: feasible seed {candidate} "
                    f"({len(episodes)}/{count})",
                    flush=True,
                )
            candidate += 1
        return episodes

    def start_episode(self, episode_id: int, seed: int, info: dict[str, Any]):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # Match the upstream evaluator by reusing the task/robot planners across
        # episodes. Curobo warmup dominates these short diagnostics otherwise.
        if self._episode_task is None:
            self._episode_task = self._new_task()
        task = self._episode_task
        try:
            task.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **self.args)
        except Exception:
            self._episode_task = None
            raise
        task.step_lim = self.step_limit
        descriptions = self.generate_episode_descriptions(self.task_name, [info], 100)
        if not descriptions or not descriptions[0].get(self.instruction_type):
            raise RuntimeError(
                f"No {self.instruction_type} instruction for {self.task_name}, seed {seed}"
            )
        choices = descriptions[0][self.instruction_type]
        instruction = choices[seed % len(choices)]
        task.set_instruction(instruction=instruction)
        return task, instruction

    @staticmethod
    def run_action_chunk(task, actions: np.ndarray) -> int:
        executed = 0
        for action in np.asarray(actions, dtype=np.float32):
            if task.take_action_cnt >= task.step_lim or task.eval_success:
                break
            task.take_action(action)
            executed += 1
        return executed

    @staticmethod
    def run_action_chunk_with_timing(
        task, actions: np.ndarray
    ) -> tuple[int, list[float]]:
        """Same as run_action_chunk, but also times each physical sim step."""
        executed = 0
        step_latencies: list[float] = []
        for action in np.asarray(actions, dtype=np.float32):
            if task.take_action_cnt >= task.step_lim or task.eval_success:
                break
            started = time.perf_counter()
            task.take_action(action)
            step_latencies.append(time.perf_counter() - started)
            executed += 1
        return executed, step_latencies

    @staticmethod
    def close_episode(task, clear_cache: bool = False) -> None:
        task.close_env(clear_cache=clear_cache)
