import unittest

import numpy as np
import torch

from experiments.robotwin.router_metrics import (
    Router,
    compute_uncertainty,
    overlap_continuous_drift,
    overlap_proxy_farmass,
    rebin_proxy_logits,
    temporal_mode_farmass,
    threshold_for_rate,
    windowed_total_variation,
)


class RouterMetricsTest(unittest.TestCase):
    def test_uniform_entropy(self):
        result = compute_uncertainty(torch.zeros(14, 256))
        self.assertAlmostEqual(result["entropy_mean"], float(np.log(256)), places=5)

    def test_windowed_total_variation(self):
        self.assertEqual(windowed_total_variation([1.0]), 0.0)
        self.assertAlmostEqual(windowed_total_variation([1, 3, 2], window=3), 1.5)

    def test_fixed_routes_every_fifth_query(self):
        router = Router("fixed", cot_ratio=0.2)
        decisions = [router.update({"entropy_mean": 1, "farmass_mean": 1})["use_zr0"] for _ in range(10)]
        self.assertEqual(decisions, [True, False, False, False, False] * 2)

    def test_threshold_rate(self):
        threshold = threshold_for_rate(range(100), 0.2)
        self.assertAlmostEqual(threshold, 79.2)

    def test_rebin_proxy_logits_preserves_mass(self):
        probs = rebin_proxy_logits(torch.zeros(2 * 3, 256), horizon=2, slots_per_step=3)
        self.assertEqual(probs.shape, (2, 3, 100))
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-6)

    def test_temporal_mode_farmass_identical_is_zero(self):
        probs = np.zeros((2, 2, 100), dtype=np.float32)
        probs[:, :, 10] = 1.0
        result = temporal_mode_farmass(probs)
        self.assertEqual(result["step_mode_farmass"], 0.0)

    def test_overlap_proxy_farmass_identical_is_zero(self):
        probs = np.zeros((25, 2, 100), dtype=np.float32)
        probs[:, :, 20] = 1.0
        result = overlap_proxy_farmass(probs, probs, executed_steps=16)
        self.assertEqual(result["overlap_proxy_farmass"], 0.0)
        self.assertEqual(result["overlap_steps"], 9)

    def test_overlap_continuous_drift(self):
        previous = np.zeros((25, 14), dtype=np.float32)
        current = np.ones((25, 14), dtype=np.float32)
        result = overlap_continuous_drift(previous, current, executed_steps=16)
        self.assertEqual(result["overlap_continuous_drift"], 1.0)
        self.assertEqual(result["overlap_steps"], 9)

    def test_proxy_metrics_do_not_mutate_cached_chunks(self):
        rng = np.random.default_rng(7)
        previous = rng.random((25, 2, 100), dtype=np.float32)
        current = rng.random((25, 2, 100), dtype=np.float32)
        previous_copy = previous.copy()
        current_copy = current.copy()
        temporal_mode_farmass(current)
        overlap_proxy_farmass(previous, current, executed_steps=16)
        np.testing.assert_array_equal(previous, previous_copy)
        np.testing.assert_array_equal(current, current_copy)

    def test_temporal_router_modes_and_first_query(self):
        base = {"entropy_mean": 1.0, "farmass_mean": 1.0}
        step = Router("step_mode_farmass_ablation", threshold=0.5)
        self.assertTrue(step.update({**base, "step_mode_farmass": 0.5})["use_zr0"])

        for mode in ("overlap_proxy_farmass", "overlap_continuous_drift"):
            router = Router(mode, threshold=0.5)
            first = router.update({**base, mode: None})
            second = router.update({**base, mode: 0.6})
            self.assertFalse(first["use_zr0"])
            self.assertTrue(second["use_zr0"])


if __name__ == "__main__":
    unittest.main()
