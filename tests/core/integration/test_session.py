from pathlib import Path

import pytest
import yaml
from conftest import requires_assets, requires_turtlebot_assets

pytestmark = [pytest.mark.integration]

SO101_MODEL = "assets/so101/so101_new_calib_camera.xml"
TURTLEBOT_MODEL = "assets/turtlebot4/turtlebot4.xml"


def _manifest(tmp_path: Path, **data):
    from physai.config import load_manifest

    data.setdefault("schema_version", 1)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_manifest(path)


def _arm(**extra) -> dict:
    return {
        "id": "arm_1",
        "robot": "so101",
        "task": "pick_place",
        "config": {"render": False, "max_steps": 10},
        **extra,
    }


@requires_assets
def test_a_single_robot_manifest_builds_a_task_wrapped_runtime(tmp_path):
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            robots=[_arm()],
            simulation={"seed": 5},
            success_hold_steps=3,
            scene={
                "overrides": {
                    "robot_xml": SO101_MODEL,  # replaces the robot's own default
                    "camera_width": 80,
                    "camera_height": 60,
                }
            },
        ),
        render=True,
    )
    try:
        runtime = session.runtime
        robot = runtime.robot
        assert session.world is None
        assert session.robot_name == "so101"
        assert robot.success_hold_steps == 3
        assert (robot.cfg.seed, robot.cfg.max_steps) == (5, 10)
        assert runtime.policy is None
        assert robot.cfg.scene.robot_xml.name == "so101_new_calib_camera.xml"
        assert robot.render_enabled is True
        assert robot.camera_size == (80, 60)
        assert robot.robot_spec is robot.robot_spec  # built once, not per access
        assert runtime.reset(seed=5).joint_state.position.size == 6
    finally:
        session.close()

    quiet = create_session(_manifest(tmp_path, robots=[_arm()]), render=False)
    try:
        assert quiet.runtime.robot.render_enabled is False
        assert quiet.runtime.robot.camera_size is None
    finally:
        quiet.close()


@requires_assets
def test_a_host_driven_session_disables_inline_camera_rendering(tmp_path):
    from physai.runtime import create_session

    session = create_session(
        _manifest(tmp_path, robots=[_arm()]), render=True, host_driven=True
    )
    try:
        cfg = session.runtime.robot.cfg
        assert cfg.render is True
        assert cfg.camera_stride == 0
        assert session.host_renders_cameras is True
        # the host scores no task, so none is composed around the robot
        assert session.runtime.task is None
        assert session.runtime.policy is None
    finally:
        session.close()


@requires_turtlebot_assets
def test_a_base_robot_gets_a_zero_twist_hold_policy(tmp_path):
    from physai.policy import ConstantTwistPolicy
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            robots=[{"id": "base", "robot": "turtlebot4", "policy": "constant"}],
        ),
        host_driven=True,  # TurtleBot4 has no camera_stride; must not fail
    )
    try:
        assert isinstance(session.runtime.policy, ConstantTwistPolicy)
        # it renders inline and cannot stop, so the host must not render for it
        assert session.host_renders_cameras is False
    finally:
        session.close()


@requires_assets
@requires_turtlebot_assets
def test_a_world_block_builds_one_shared_world(tmp_path):
    from physai.config import load_manifest
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            world={"control_hz": 20},
            robots=[
                {"id": "arm_1", "robot": "so101", "model": SO101_MODEL},
                {"id": "base_1", "robot": "turtlebot4", "model": TURTLEBOT_MODEL},
            ],
        )
    )
    try:
        assert session.runtime is None
        assert [item.instance_id for item in session.instances] == ["arm_1", "base_1"]
        assert session.world.control_hz == 20
    finally:
        session.close()

    example = create_session(
        load_manifest("configs/manifests/example_heterogeneous.yaml")
    )
    try:
        assert len(example.instances) == 2
    finally:
        example.close()


def test_a_session_refuses_what_it_cannot_build(tmp_path):
    from physai.runtime import create_session

    cases = [
        (
            _manifest(tmp_path, robots=[_arm(config={"render": False, "seed": 1})]),
            "set it under 'simulation'",
        ),
        (
            _manifest(
                tmp_path,
                world={},
                robots=[
                    {
                        "id": "arm_1",
                        "robot": "so101",
                        "model": SO101_MODEL,
                        "task": "pick_place",
                    }
                ],
            ),
            "do not run tasks or policies",
        ),
        (
            _manifest(
                tmp_path, backend="ros2_sim", robots=[{"id": "a", "robot": "so101"}]
            ),
            "not supported by create_session",
        ),
    ]

    for manifest, message in cases:
        with pytest.raises(ValueError, match=message):
            create_session(manifest)
