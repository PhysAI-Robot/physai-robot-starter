import numpy as np
import pytest

from physai.contracts import JointState, Observation
from physai.robots import RobotSpec


class FakeRobot:
    def __init__(self) -> None:
        self.robot_spec = RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=("joint",),
            action_joint_names=("joint",),
            action_modes=("joint_position",),
            capabilities=("arm_kinematics", "gripper"),
        )
        self.closed = False
        self.observation = Observation(
            joint_state=JointState(
                name=("joint",),
                position=np.zeros(1),
                velocity=np.zeros(1),
                effort=np.zeros(1),
            )
        )

    def reset(self, seed=None):
        del seed
        return self.observation

    def observe(self):
        return self.observation

    def step(self, action):
        del action
        return self.observation, 0.0, False, False, {}

    def close(self):
        self.closed = True


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

    fake = FakeRobot()
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

    fake = FakeRobot()
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
