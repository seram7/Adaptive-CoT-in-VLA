from experiments.robotwin.audit_shared_prefix import audit_pair


def _episode(policies):
    records = []
    for index, policy in enumerate(policies):
        records.append(
            {
                "query_index": index,
                "observation_hash": f"obs-{index}",
                "pi05_action_seed": index + 10,
                "pi05_action_hash": f"pi-{index}",
                "selected_policy": policy,
                "selected_action_hash": f"selected-{index}-{policy}",
                "executed_action_hash": f"executed-{index}-{policy}",
                "executed_actions": 16,
            }
        )
    return {
        "task": "adjust_bottle",
        "task_config": "demo_clean",
        "seed": 123,
        "success": False,
        "sim_steps": len(records) * 16,
        "records": records,
    }


def test_pair_accepts_identical_prefix_and_reports_first_routing_divergence():
    left = _episode(["zr0_direct_action", "pi05", "zr0_direct_action"])
    right = _episode(["zr0_direct_action", "pi05", "pi05"])
    assert audit_pair(left, right) == 2


def test_pair_accepts_fully_identical_trajectory():
    left = _episode(["zr0_direct_action", "pi05"])
    right = _episode(["zr0_direct_action", "pi05"])
    assert audit_pair(left, right) is None
