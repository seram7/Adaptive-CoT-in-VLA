"""RoboTwin task registry for PI0.5 lightweight Flow-Far-Mass runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PI05TaskSpec:
    name: str
    step_limit: int
    task_idx: int


# RoboTwin's canonical alphabetical 0-based task_idx ordering. Keep the full
# list here so even/odd campaigns cannot silently drift apart when moved to a
# different machine.
PI05_ALL_TASKS: tuple[PI05TaskSpec, ...] = (
    PI05TaskSpec("adjust_bottle", 400, 0),
    PI05TaskSpec("beat_block_hammer", 400, 1),
    PI05TaskSpec("blocks_ranking_rgb", 1200, 2),
    PI05TaskSpec("blocks_ranking_size", 1200, 3),
    PI05TaskSpec("click_alarmclock", 400, 4),
    PI05TaskSpec("click_bell", 400, 5),
    PI05TaskSpec("dump_bin_bigbin", 600, 6),
    PI05TaskSpec("grab_roller", 400, 7),
    PI05TaskSpec("handover_block", 800, 8),
    PI05TaskSpec("handover_mic", 600, 9),
    PI05TaskSpec("hanging_mug", 900, 10),
    PI05TaskSpec("lift_pot", 400, 11),
    PI05TaskSpec("move_can_pot", 400, 12),
    PI05TaskSpec("move_pillbottle_pad", 400, 13),
    PI05TaskSpec("move_playingcard_away", 400, 14),
    PI05TaskSpec("move_stapler_pad", 400, 15),
    PI05TaskSpec("open_laptop", 700, 16),
    PI05TaskSpec("open_microwave", 1500, 17),
    PI05TaskSpec("pick_diverse_bottles", 400, 18),
    PI05TaskSpec("pick_dual_bottles", 400, 19),
    PI05TaskSpec("place_a2b_left", 400, 20),
    PI05TaskSpec("place_a2b_right", 400, 21),
    PI05TaskSpec("place_bread_basket", 700, 22),
    PI05TaskSpec("place_bread_skillet", 500, 23),
    PI05TaskSpec("place_burger_fries", 500, 24),
    PI05TaskSpec("place_can_basket", 700, 25),
    PI05TaskSpec("place_cans_plasticbox", 800, 26),
    PI05TaskSpec("place_container_plate", 400, 27),
    PI05TaskSpec("place_dual_shoes", 600, 28),
    PI05TaskSpec("place_empty_cup", 500, 29),
    PI05TaskSpec("place_fan", 400, 30),
    PI05TaskSpec("place_mouse_pad", 400, 31),
    PI05TaskSpec("place_object_basket", 700, 32),
    PI05TaskSpec("place_object_scale", 400, 33),
    PI05TaskSpec("place_object_stand", 400, 34),
    PI05TaskSpec("place_phone_stand", 400, 35),
    PI05TaskSpec("place_shoe", 500, 36),
    PI05TaskSpec("press_stapler", 400, 37),
    PI05TaskSpec("put_bottles_dustbin", 1700, 38),
    PI05TaskSpec("put_object_cabinet", 700, 39),
    PI05TaskSpec("rotate_qrcode", 400, 40),
    PI05TaskSpec("scan_object", 500, 41),
    PI05TaskSpec("shake_bottle", 700, 42),
    PI05TaskSpec("shake_bottle_horizontally", 700, 43),
    PI05TaskSpec("stack_blocks_three", 1200, 44),
    PI05TaskSpec("stack_blocks_two", 800, 45),
    PI05TaskSpec("stack_bowls_three", 1200, 46),
    PI05TaskSpec("stack_bowls_two", 900, 47),
    PI05TaskSpec("stamp_seal", 400, 48),
    PI05TaskSpec("turn_switch", 400, 49),
)

PI05_EVEN_TASKS = PI05_ALL_TASKS[0::2]
PI05_ODD_TASKS = PI05_ALL_TASKS[1::2]

_FOCUSED_RERUN_NAMES = {"beat_block_hammer", "pick_dual_bottles"}
PI05_FOCUSED_RERUN_TASKS = tuple(
    task for task in PI05_ODD_TASKS if task.name in _FOCUSED_RERUN_NAMES
)

PI05_TASKS = PI05_ALL_TASKS
PI05_TASK_BY_NAME = {task.name: task for task in PI05_TASKS}
PI05_EVEN_TASK_NAMES = tuple(task.name for task in PI05_EVEN_TASKS)
PI05_ODD_TASK_NAMES = tuple(task.name for task in PI05_ODD_TASKS)


def get_pi05_task(name: str) -> PI05TaskSpec:
    try:
        return PI05_TASK_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(task.name for task in PI05_TASKS)
        raise KeyError(f"Unknown PI0.5 evaluation task {name!r}; choose one of: {supported}") from exc
