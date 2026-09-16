import numpy as np
import pytest

from physai.robots import RobotSpec
from tests.support.fakes import FakeRobotPort


def test_builtin_scene_registry_returns_typed_configs():
    from physai.sim import (
        PickPlaceMinimalSceneConfig,
        SortingMinimalSceneConfig,
        WorldSceneConfig,
        create_scene,
    )
    from physai.sim.scenes import available_scenes

    assert {"pick_place_minimal", "sorting_minimal"} <= set(available_scenes())
    assert isinstance(create_scene("pick_place_minimal"), PickPlaceMinimalSceneConfig)
    assert isinstance(create_scene("sorting_minimal"), SortingMinimalSceneConfig)
    assert not hasattr(WorldSceneConfig(), "static_pad_body")


def test_runtime_rejects_task_scene_mismatch(monkeypatch):
    from physai.runtime import composition

    fake = FakeRobotPort(
        RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=("joint",),
            action_joint_names=("joint",),
            action_modes=("joint_position",),
            capabilities=("arm_kinematics", "gripper"),
        ),
        validate_actions=False,
    )
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    with pytest.raises(ValueError, match="scene .* incompatible"):
        composition.create_runtime(
            "so101",
            scene_name="sorting_minimal",
            task_name="pick_place",
        )
    assert fake.closed


def test_runtime_records_explicit_scene(monkeypatch):
    from physai.runtime import composition

    fake = FakeRobotPort(
        RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=("joint",),
            action_joint_names=("joint",),
            action_modes=("joint_position",),
            capabilities=("arm_kinematics", "gripper"),
        ),
        validate_actions=False,
    )
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)
    runtime = composition.create_runtime(
        "so101",
        scene_name="sorting_minimal",
        task_name="sorting",
    )
    try:
        assert runtime.scene_name == "sorting_minimal"
        assert runtime.task.name == "sorting"
    finally:
        runtime.close()
