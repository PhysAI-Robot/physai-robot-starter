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


def test_scenes_name_their_object_layout():
    from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig

    assert PickPlaceMinimalSceneConfig.layout_kind == "single_cube"
    assert SortingMinimalSceneConfig.layout_kind == "sorting"
    # a class-level hint, not a field, so dataset metadata is unchanged
    assert "layout_kind" not in PickPlaceMinimalSceneConfig().to_metadata()


def test_a_scene_with_an_unknown_layout_is_refused():
    from types import SimpleNamespace

    import pytest

    from physai.robots.so101.layout import create_layout

    with pytest.raises(ValueError, match="supports: single_cube, sorting"):
        create_layout(None, SimpleNamespace(layout_kind="stacking"))


def test_a_scene_without_robot_defaults_names_what_is_missing():
    import pytest

    from physai.sim import PickPlaceMinimalSceneConfig

    with pytest.raises(ValueError, match="pad_size.*wrist_cam_pos"):
        PickPlaceMinimalSceneConfig().build_spec()


def test_the_so101_supplies_the_grasp_pad_fit_and_wrist_camera_pose():
    from physai.robots.registry import scene_defaults

    defaults = scene_defaults("so101")

    assert defaults["pad_align_gripper_q"] == 0.16
    assert len(defaults["wrist_cam_xyaxes"]) == 6
