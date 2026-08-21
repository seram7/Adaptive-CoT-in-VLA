import random

import numpy as np

from experiments.robotwin.robotwin_env import RoboTwinHarness


class _FakeTask:
    def __init__(self):
        self.plan_success = True
        self.samples = None

    def setup_demo(self, **kwargs):
        del kwargs
        self.samples = (random.random(), float(np.random.random()))

    def play_once(self):
        return {"info": {}}

    def check_success(self):
        return True

    def close_env(self, clear_cache=False):
        del clear_cache


def test_feasible_seed_rng_matches_episode_reconstruction():
    seed = 12345
    task = _FakeTask()
    harness = object.__new__(RoboTwinHarness)
    harness.task_name = "fake"
    harness.args = {}
    harness.UnStableError = RuntimeError
    harness._new_task = lambda: task

    episodes = harness.find_feasible_seeds(1, start_seed=seed)

    random.seed(seed)
    np.random.seed(seed % (2**32))
    expected = (random.random(), float(np.random.random()))
    assert episodes == [{"seed": seed, "info": {}}]
    assert task.samples == expected
