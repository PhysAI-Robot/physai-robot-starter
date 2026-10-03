import pytest

from physai.robots import RobotSpec
from tests.core.support.fakes import FakeRobotPort


def test_builtin_scene_registry_returns_typed_configs():
    from physai.sim.mujoco import (
        SingleCubeFixedPlaceSceneConfig,
        SortingMinimalSceneConfig,
        WorldSceneConfig,
        create_scene,
    )
    from physai.sim.mujoco.scenes import available_scenes

    assert {"single_cube_fixed_place", "sorting_minimal"} <= set(available_scenes())
    assert isinstance(
        create_scene("single_cube_fixed_place"), SingleCubeFixedPlaceSceneConfig
    )
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
            task_name="single_cube_fixed_place",
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


def test_scenes_name_their_layout_and_reject_an_unknown_one():
    from types import SimpleNamespace

    import pytest

    from physai.robots.so101.layout import create_layout
    from physai.sim.mujoco import (
        SingleCubeFixedPlaceSceneConfig,
        SortingMinimalSceneConfig,
    )

    assert SingleCubeFixedPlaceSceneConfig.layout_kind == "single_cube"
    assert SortingMinimalSceneConfig.layout_kind == "sorting"
    # a class-level hint, not a field, so dataset metadata is unchanged
    assert "layout_kind" not in SingleCubeFixedPlaceSceneConfig().to_metadata()
    with pytest.raises(ValueError, match="supports: single_cube, sorting"):
        create_layout(None, SimpleNamespace(layout_kind="stacking"))


def test_the_robot_supplies_the_grasp_pad_fit_a_generic_scene_lacks():
    import pytest

    from physai.robots.registry import scene_defaults
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    with pytest.raises(ValueError, match="description"):
        SingleCubeFixedPlaceSceneConfig().build_spec()
    defaults = scene_defaults("so101")
    description = defaults["description"]
    assert description.derivation["pad_align_gripper_q"] == 0.16
    assert {pad.name for pad in description.contact_pads} == {
        "pad_static",
        "pad_moving",
    }
