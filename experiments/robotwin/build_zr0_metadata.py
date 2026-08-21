#!/usr/bin/env python3
"""Build the minimal LeRobot v2.1 metadata required by ZR-0 inference.

The released ZR-0 RoboTwin policy expects the global normalization statistics
of the 50-task ALOHA-AgileX clean training slice, but its checkpoint does not
ship those files.  This script reads only joint arrays from the official
``aloha-agilex_clean_50.zip`` archives and reproduces ZR-0's own preprocessing:
state at t is paired with the absolute joint-position action at t + 1.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np


MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--expected-episodes-per-task", type=int, default=50)
    return parser.parse_args()


def episode_number(name: str) -> int:
    match = re.search(r"/data/episode(\d+)\.hdf5$", name)
    if match is None:
        raise ValueError(name)
    return int(match.group(1))


def joint_vectors(handle: h5py.File) -> np.ndarray:
    group = handle["joint_action"]
    left_arm = np.asarray(group["left_arm"], dtype=np.float32)
    left_gripper = np.asarray(group["left_gripper"], dtype=np.float32).reshape(-1, 1)
    right_arm = np.asarray(group["right_arm"], dtype=np.float32)
    right_gripper = np.asarray(group["right_gripper"], dtype=np.float32).reshape(-1, 1)
    values = np.concatenate((left_arm, left_gripper, right_arm, right_gripper), axis=1)
    if values.ndim != 2 or values.shape[1] != len(MOTORS):
        raise ValueError(f"Expected [T, 14] joint vectors, got {values.shape}")
    return values


def stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.percentile(values, 1, axis=0).tolist(),
        "q99": np.percentile(values, 99, axis=0).tolist(),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    archives = sorted(args.archives_root.glob("dataset/*/aloha-agilex_clean_50.zip"))
    if len(archives) != args.expected_tasks:
        raise RuntimeError(f"Expected {args.expected_tasks} archives, found {len(archives)}")

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    episode_index = 0

    for task_index, archive in enumerate(archives):
        task_name = archive.parent.name
        task_rows.append({"task_index": task_index, "task": task_name})
        with zipfile.ZipFile(archive) as bundle:
            members = sorted(
                (name for name in bundle.namelist() if re.search(r"/data/episode\d+\.hdf5$", name)),
                key=episode_number,
            )
            if len(members) != args.expected_episodes_per_task:
                raise RuntimeError(
                    f"{task_name}: expected {args.expected_episodes_per_task} episodes, "
                    f"found {len(members)}"
                )
            for member in members:
                with bundle.open(member) as source:
                    with h5py.File(io.BytesIO(source.read()), "r") as handle:
                        sequence = joint_vectors(handle)
                if len(sequence) < 2:
                    raise RuntimeError(f"{archive}:{member} has fewer than two frames")
                all_states.append(sequence[:-1])
                all_actions.append(sequence[1:])
                episode_rows.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [task_name],
                        "length": len(sequence) - 1,
                    }
                )
                episode_index += 1
        print(f"[{task_index + 1:02d}/{len(archives)}] {task_name}", flush=True)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    if states.shape != actions.shape or states.shape[1] != len(MOTORS):
        raise RuntimeError(f"Unexpected aggregate shapes: state={states.shape}, action={actions.shape}")

    meta = args.output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(MOTORS)],
            "names": [MOTORS],
        },
        "action": {"dtype": "float32", "shape": [len(MOTORS)], "names": [MOTORS]},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": [3, 480, 640],
            "names": ["channels", "height", "width"],
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": len(episode_rows),
        "total_frames": int(states.shape[0]),
        "total_tasks": len(task_rows),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 100000,
        "fps": 15,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "grounding_camera_keys": [f"observation.images.{camera}" for camera in CAMERAS],
    }
    write_json(meta / "info.json", info)
    write_json(meta / "stats.json", {"observation.state": stats(states), "action": stats(actions)})
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_rows), encoding="utf-8"
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tasks": len(task_rows),
                "episodes": len(episode_rows),
                "frames": int(states.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
