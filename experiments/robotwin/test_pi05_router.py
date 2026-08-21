from experiments.robotwin.evaluate_pi05_zr0 import SamplingFarMassRouter
from experiments.robotwin.evaluate_pi05_zr0 import STEP_THRESHOLD_MODE
from experiments.robotwin.evaluate_pi05_zr0 import THRESHOLD_MODE


def test_sampling_far_mass_router_uses_windowed_total_variation():
    router = SamplingFarMassRouter(
        THRESHOLD_MODE,
        cot_ratio=0.2,
        threshold=0.4,
        tv_window=3,
        seed=1,
    )
    first = router.update(0.0)
    second = router.update(1.0)
    third = router.update(1.2)

    assert not first["use_zr0"]
    assert second["use_zr0"]
    assert third["sampling_farmass_wtv"] == 0.6
    assert third["use_zr0"]


def test_sampling_far_mass_router_cooldown_suppresses_consecutive_triggers():
    router = SamplingFarMassRouter(
        THRESHOLD_MODE,
        cot_ratio=0.2,
        threshold=0.4,
        tv_window=3,
        seed=1,
        cooldown_queries=2,
    )
    router.update(0.0)
    triggered = router.update(1.0)
    suppressed = router.update(1.2)
    still_cooling = router.update(1.2)
    retriggered = router.update(2.4)

    assert triggered["use_zr0"]
    assert not triggered["suppressed_by_cooldown"]
    # WTV alone would fire again here (0.6 >= 0.4), but cooldown blocks it.
    assert suppressed["sampling_farmass_wtv"] >= 0.4
    assert not suppressed["use_zr0"]
    assert suppressed["suppressed_by_cooldown"]
    assert not still_cooling["use_zr0"]
    assert retriggered["use_zr0"]


def test_force_first_query_overrides_threshold_mode_and_starts_cooldown():
    router = SamplingFarMassRouter(
        THRESHOLD_MODE,
        cot_ratio=0.2,
        threshold=0.4,
        tv_window=3,
        seed=1,
        cooldown_queries=2,
        force_first_query=True,
    )
    # WTV can never fire at query 0 (history < 2), yet it must still use zr0.
    first = router.update(0.0)
    assert first["use_zr0"]
    assert not first["suppressed_by_cooldown"]

    # The forced call starts the cooldown clock like any other zr0 use.
    second = router.update(5.0)
    assert not second["use_zr0"]
    assert second["suppressed_by_cooldown"]


def test_force_first_query_overrides_random_mode():
    router = SamplingFarMassRouter(
        "random",
        cot_ratio=0.0,  # would never fire zr0 on its own
        threshold=None,
        tv_window=5,
        seed=1,
        force_first_query=True,
    )
    first = router.update(0.1)
    second = router.update(0.1)
    assert first["use_zr0"]
    assert not second["use_zr0"]


def test_sampling_far_mass_router_ignores_missing_metric():
    router = SamplingFarMassRouter(
        "baseline",
        cot_ratio=0.2,
        threshold=None,
        tv_window=5,
        seed=1,
    )
    result = router.update(None)
    assert result["sampling_farmass_history_count"] == 0
    assert result["sampling_farmass_wtv"] == 0.0
    assert not result["use_zr0"]


def test_step_router_replans_with_executed_step_cooldown():
    router = SamplingFarMassRouter(
        STEP_THRESHOLD_MODE,
        cot_ratio=0.2,
        threshold=0.4,
        tv_window=2,
        seed=1,
        cooldown_steps=16,
        force_first_query=True,
    )

    first = router.update(None, step_uncertainty=0.0)
    assert first["use_zr0"]
    router.record_execution(8)

    cooling = router.update(None, step_uncertainty=1.0)
    assert not cooling["use_zr0"]
    assert cooling["suppressed_by_cooldown"]
    router.record_execution(8)

    eligible = router.update(None, step_uncertainty=1.0)
    assert eligible["use_zr0"]
