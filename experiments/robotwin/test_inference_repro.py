import sqlite3

import numpy as np
import pytest

from experiments.robotwin.inference_repro import InferenceCache
from experiments.robotwin.inference_repro import action_hash
from experiments.robotwin.inference_repro import observation_hash
from experiments.robotwin.inference_repro import request_identity
from experiments.robotwin.inference_repro import stable_request_seed


def _observation(pixel: int = 0):
    image = np.full((4, 5, 3), pixel, dtype=np.uint8)
    return {
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        "observation": {
            "head_camera": {"rgb": image.copy()},
            "left_camera": {"rgb": image.copy()},
            "right_camera": {"rgb": image.copy()},
        },
    }


def test_stable_seed_is_arm_independent_and_domain_separated():
    args = ("adjust_bottle", "main/demo_clean", 123, 4)
    first = stable_request_seed(*args, "pi05_execution")
    assert first == stable_request_seed(*args, "pi05_execution")
    assert first != stable_request_seed(*args, "pi05_farmass")
    assert first != stable_request_seed("adjust_bottle", "main/demo_clean", 123, 5, "pi05_execution")


def test_observation_and_action_hashes_cover_content():
    assert observation_hash(_observation(0), "task") == observation_hash(_observation(0), "task")
    assert observation_hash(_observation(0), "task") != observation_hash(_observation(1), "task")
    assert action_hash(np.zeros((2, 14), dtype=np.float32)) != action_hash(
        np.ones((2, 14), dtype=np.float32)
    )


def test_cache_reuses_actions_and_asserts_conflicting_output(tmp_path):
    identity = request_identity(
        task="adjust_bottle",
        split="main/demo_clean",
        episode_seed=123,
        query_index=0,
        stream="zr0_direct_action_execution",
        request_seed=7,
        observation_digest="abc",
    )
    actions = np.arange(28, dtype=np.float32).reshape(2, 14)
    with InferenceCache(tmp_path / "cache.sqlite3") as cache:
        cache.put_action(identity, actions)
        np.testing.assert_array_equal(cache.get_action(identity), actions)
        with pytest.raises(AssertionError, match="different actions"):
            cache.put_action(identity, actions + 1)


def test_cache_detects_corrupt_blob(tmp_path):
    identity = request_identity(
        task="adjust_bottle",
        split="main/demo_clean",
        episode_seed=123,
        query_index=0,
        stream="pi05_execution",
        request_seed=8,
        observation_digest="abc",
    )
    path = tmp_path / "cache.sqlite3"
    with InferenceCache(path) as cache:
        cache.put_action(identity, np.zeros((2, 14), dtype=np.float32))
    connection = sqlite3.connect(path)
    connection.execute("UPDATE actions SET action_bytes = ?", (sqlite3.Binary(b"broken"),))
    connection.commit()
    connection.close()
    with InferenceCache(path) as cache:
        with pytest.raises((AssertionError, ValueError)):
            cache.get_action(identity)
