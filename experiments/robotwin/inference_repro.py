"""Deterministic request identities, hashes, and persistent inference sharing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np


SEED_PROTOCOL = "robotwin-policy-request-v1"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_request_seed(
    task: str,
    split: str,
    episode_seed: int,
    query_index: int,
    stream: str,
) -> int:
    """Return a stable positive 63-bit seed without relying on Python hash()."""

    payload = {
        "protocol": SEED_PROTOCOL,
        "task": task,
        "split": split,
        "episode_seed": int(episode_seed),
        "query_index": int(query_index),
        "stream": stream,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _update_array_hash(digest: Any, name: str, value: Any, dtype: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json({"shape": list(array.shape)}).encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def observation_hash(observation: dict[str, Any], instruction: str) -> str:
    """Hash the exact state, RGB observations, and instruction sent to policies."""

    obs = observation["observation"]
    digest = hashlib.sha256()
    digest.update(instruction.encode("utf-8"))
    _update_array_hash(digest, "state", observation["joint_action"]["vector"], np.float32)
    for key in ("head_camera", "left_camera", "right_camera"):
        _update_array_hash(digest, key, obs[key]["rgb"], np.uint8)
    return digest.hexdigest()


def action_hash(actions: Any) -> str:
    digest = hashlib.sha256()
    _update_array_hash(digest, "actions", actions, np.float32)
    return digest.hexdigest()


def request_identity(
    *,
    task: str,
    split: str,
    episode_seed: int,
    query_index: int,
    stream: str,
    request_seed: int,
    observation_digest: str,
) -> dict[str, Any]:
    return {
        "protocol": SEED_PROTOCOL,
        "task": task,
        "split": split,
        "episode_seed": int(episode_seed),
        "query_index": int(query_index),
        "stream": stream,
        "request_seed": int(request_seed),
        "observation_hash": observation_digest,
    }


class InferenceCache:
    """SQLite-backed cache shared by sequential routing-arm evaluator processes."""

    def __init__(self, path: Path | None):
        self.path = path.expanduser().resolve() if path is not None else None
        self.connection: sqlite3.Connection | None = None
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS actions (
                cache_key TEXT PRIMARY KEY,
                identity_json TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                action_bytes BLOB NOT NULL,
                action_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
                cache_key TEXT PRIMARY KEY,
                identity_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _key(identity: dict[str, Any]) -> tuple[str, str]:
        encoded = _canonical_json(identity)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), encoded

    def get_action(self, identity: dict[str, Any]) -> np.ndarray | None:
        if self.connection is None:
            return None
        key, encoded = self._key(identity)
        row = self.connection.execute(
            "SELECT identity_json, shape_json, action_bytes, action_hash FROM actions WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != encoded:
            raise AssertionError(f"Inference cache identity collision for {key}")
        shape = tuple(json.loads(row[1]))
        actions = np.frombuffer(row[2], dtype=np.float32).copy().reshape(shape)
        if action_hash(actions) != row[3]:
            raise AssertionError(f"Corrupt cached action for {key}")
        return actions

    def put_action(self, identity: dict[str, Any], actions: Any) -> str:
        array = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
        digest = action_hash(array)
        if self.connection is None:
            return digest
        key, encoded = self._key(identity)
        existing = self.get_action(identity)
        if existing is not None:
            if action_hash(existing) != digest:
                raise AssertionError(
                    "Same deterministic inference request produced different actions: "
                    f"key={key} cached={action_hash(existing)} new={digest}"
                )
            return digest
        self.connection.execute(
            "INSERT INTO actions(cache_key, identity_json, shape_json, action_bytes, action_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                encoded,
                json.dumps(list(array.shape)),
                sqlite3.Binary(array.tobytes(order="C")),
                digest,
            ),
        )
        self.connection.commit()
        return digest

    def get_metrics(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        if self.connection is None:
            return None
        key, encoded = self._key(identity)
        row = self.connection.execute(
            "SELECT identity_json, metrics_json FROM metrics WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row[0] != encoded:
            raise AssertionError(f"Inference cache identity collision for {key}")
        return json.loads(row[1])

    def put_metrics(self, identity: dict[str, Any], metrics: dict[str, Any]) -> None:
        if self.connection is None:
            return
        key, encoded = self._key(identity)
        metrics_json = _canonical_json(metrics)
        existing = self.get_metrics(identity)
        if existing is not None:
            if _canonical_json(existing) != metrics_json:
                raise AssertionError(
                    f"Same deterministic Far-Mass request produced different metrics: key={key}"
                )
            return
        self.connection.execute(
            "INSERT INTO metrics(cache_key, identity_json, metrics_json) VALUES (?, ?, ?)",
            (key, encoded, metrics_json),
        )
        self.connection.commit()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "InferenceCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
