import numpy as np


class FakePickPlaceBackend:
    cube_pos = np.array([0.2, 0.0, 0.034])
    target_pos = np.array([0.2, 0.0, 0.021])
    ee_pos = np.array([0.2, 0.0, 0.08])
    table_top = 0.02
    cube_half = 0.014


def test_pick_place_task_owns_metrics_and_reward():
    from physai.tasks import PickPlaceTask

    task = PickPlaceTask(success_xy_tol=0.04)
    backend = FakePickPlaceBackend()
    info = task.evaluate(backend)

    assert info["at_target"]
    assert info["dist_ee_cube"] > 0
    assert task.reward(backend, info) > 0


class FakeSortingBackend(FakePickPlaceBackend):
    target_color = "blue"
    cube_positions = {
        "red": np.array([0.25, 0.10, 0.034]),
        "blue": np.array([0.2, 0.0, 0.034]),
        "yellow": np.array([0.18, -0.08, 0.034]),
    }


def test_sorting_task_targets_the_named_color_and_ignores_the_rest():
    from physai.tasks import create_task

    task = create_task("sorting", success_xy_tol=0.04)
    backend = FakeSortingBackend()
    info = task.evaluate(backend)

    assert info["target_color"] == "blue"
    np.testing.assert_allclose(info["cube_pos"], backend.cube_positions["blue"])
    assert info["at_target"]
    assert task.reward(backend, info) > 0