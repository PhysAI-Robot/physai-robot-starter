from pathlib import Path

import pytest
import yaml
from tests.conftest import requires_assets, requires_turtlebot_assets

from physai.contracts import CAMERA_SIZE

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
        "task": "single_cube_fixed_place",
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
        assert robot.camera_size == CAMERA_SIZE
        assert robot.robot_spec is robot.robot_spec  # built once, not per access
        assert runtime.reset(seed=5).joint_state.position.size == 6
    finally:
        session.close()

    chosen = create_session(
        _manifest(
            tmp_path,
            robots=[_arm()],
            simulation={"seed": 5, "camera_resolution": "640x480"},
            scene={"overrides": {"robot_xml": SO101_MODEL}},
        ),
        render=True,
    )
    try:
        assert chosen.runtime.robot.camera_size == (640, 480)
    finally:
        chosen.close()

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


def test_a_manifest_routes_its_top_level_simulator_field(tmp_path, monkeypatch):
    """The manifest's top-level `simulator` field reaches
    `create_robot(simulator=...)` (see `runtime.session._create_single_session`
    and `_robot_fields`); a robot `config` that repeats it is rejected (see
    `test_a_robot_config_must_not_repeat_the_manifest_simulator` below).
    """
    from physai.robots import RobotSpec
    from physai.runtime import composition, create_session
    from tests.core.support.fakes import FakeRobotPort

    captured: dict = {}

    def fake_create_robot(robot_name, **kwargs):
        captured["robot_name"] = robot_name
        captured["kwargs"] = kwargs
        return FakeRobotPort(
            RobotSpec(
                name=robot_name,
                kind="fixed_base_manipulator",
                joint_names=("joint",),
                action_joint_names=("joint",),
                action_modes=("joint_position",),
                capabilities=(),
            ),
            validate_actions=False,
        )

    monkeypatch.setattr(composition, "create_robot", fake_create_robot)

    # No task: a task would auto-select a scene (see composition.py), and
    # IsaacEnvConfig has no `scene` field yet (SO101IsaacEnv is robot-only
    # for now, see its module docstring) — a real manifest combining a task
    # with simulator: isaac is not supported yet.
    create_session(
        _manifest(
            tmp_path,
            simulator="isaac",
            robots=[
                {
                    "id": "arm_1",
                    "robot": "so101",
                    "config": {"render": False},
                }
            ],
        )
    )
    assert captured["kwargs"]["simulator"] == "isaac"
    assert captured["kwargs"]["render"] is False


def test_a_robot_config_must_not_repeat_the_manifest_simulator(tmp_path):
    from physai.runtime import create_session

    with pytest.raises(ValueError, match="manifest's top level"):
        create_session(
            _manifest(
                tmp_path,
                robots=[
                    {
                        "id": "arm_1",
                        "robot": "so101",
                        "config": {"simulator": "isaac", "render": False},
                    }
                ],
            )
        )


def test_a_manifest_rejects_a_simulator_unsupported_by_a_robot(tmp_path):
    with pytest.raises(ValueError, match="does not support simulator 'isaac'"):
        _manifest(
            tmp_path,
            simulator="isaac",
            robots=[{"id": "base_1", "robot": "turtlebot4", "config": {}}],
        )


def test_a_manifest_rejects_a_world_with_a_non_mujoco_simulator(tmp_path):
    with pytest.raises(ValueError, match="does not support a"):
        _manifest(
            tmp_path,
            simulator="isaac",
            world={},
            robots=[
                {"id": "arm_1", "robot": "so101", "model": "assets/so101/so101.xml"}
            ],
        )


def test_a_manifest_rejects_ros2_sim_with_a_non_mujoco_simulator(tmp_path):
    with pytest.raises(ValueError, match="does not support backend='ros2_sim'"):
        _manifest(
            tmp_path,
            simulator="isaac",
            backend="ros2_sim",
            robots=[{"id": "arm_1", "robot": "so101", "config": {}}],
        )


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
                        "task": "single_cube_fixed_place",
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
