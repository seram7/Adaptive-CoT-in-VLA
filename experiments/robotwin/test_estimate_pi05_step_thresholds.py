from experiments.robotwin.estimate_pi05_step_thresholds import choose_threshold
from experiments.robotwin.estimate_pi05_step_thresholds import realized_ratio


def test_step_threshold_calibration_replays_step_cooldown_and_force_first():
    sequences = [
        [(0.1, 8), (0.9, 8), (0.8, 8), (0.1, 8), (0.7, 8)],
        [(0.1, 8), (0.8, 8), (0.1, 8), (0.7, 8), (0.1, 8)],
    ]
    threshold, ratio, eligible = choose_threshold(
        sequences,
        target_ratio=0.3,
        cooldown_steps=16,
        force_first_query=True,
    )

    replayed = realized_ratio(
        sequences,
        threshold,
        target_ratio=0.3,
        cooldown_steps=16,
        force_first_query=True,
    )
    assert eligible == 10
    assert ratio == replayed
    assert abs(ratio - 0.3) <= 0.1
