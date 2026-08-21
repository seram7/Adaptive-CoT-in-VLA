from experiments.robotwin.pi05_task_registry import PI05_ALL_TASKS
from experiments.robotwin.pi05_task_registry import PI05_EVEN_TASKS
from experiments.robotwin.pi05_task_registry import PI05_ODD_TASKS


def test_campaign_is_exactly_the_25_even_zero_based_task_indices():
    assert len(PI05_EVEN_TASKS) == 25
    assert [task.task_idx for task in PI05_EVEN_TASKS] == list(range(0, 50, 2))
    assert len({task.name for task in PI05_EVEN_TASKS}) == 25


def test_odd_campaign_is_the_complementary_25_task_indices():
    assert len(PI05_ODD_TASKS) == 25
    assert [task.task_idx for task in PI05_ODD_TASKS] == list(range(1, 50, 2))
    assert len({task.name for task in PI05_ODD_TASKS}) == 25
    assert not ({task.name for task in PI05_EVEN_TASKS} & {task.name for task in PI05_ODD_TASKS})


def test_full_registry_is_canonical_alphabetical_order():
    assert len(PI05_ALL_TASKS) == 50
    assert [task.task_idx for task in PI05_ALL_TASKS] == list(range(50))
    assert [task.name for task in PI05_ALL_TASKS] == sorted(task.name for task in PI05_ALL_TASKS)
