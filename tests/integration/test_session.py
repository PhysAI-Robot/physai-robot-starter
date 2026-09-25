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
def test_single_robot_manifest_builds_a_task_wrapped_runtime(tmp_path):
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            robots=[_arm()],
            simulation={"seed": 5},
            success_hold_steps=3,
        )
    )
    try:
        runtime = session.runtime
        assert session.world is None
        assert session.robot_name == "so101"
        assert runtime.robot.success_hold_steps == 3
        assert runtime.robot.cfg.seed == 5
        assert runtime.robot.cfg.max_steps == 10
        assert runtime.policy is None
        observation = runtime.reset(seed=5)
        assert observation.joint_state.position.size == 6
    finally:
        session.close()


@requires_assets
def test_a_robot_config_may_not_repeat_the_simulation_seed(tmp_path):
    from physai.runtime import create_session

    manifest = _manifest(tmp_path, robots=[_arm(config={"render": False, "seed": 1})])

    with pytest.raises(ValueError, match="set it under 'simulation'"):
        create_session(manifest)


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
    finally:
        session.close()


@requires_turtlebot_assets
def test_a_base_robot_gets_a_zero_twist_hold_policy(tmp_path):
    from physai.policy import ConstantTwistPolicy
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            robots=[
                {"id": "base", "robot": "turtlebot4", "policy": "constant"},
            ],
        ),
        host_driven=True,  # TurtleBot4 has no camera_stride; must not fail
    )
    try:
        assert isinstance(session.runtime.policy, ConstantTwistPolicy)
    finally:
        session.close()


@requires_assets
@requires_turtlebot_assets
def test_a_world_block_builds_one_shared_world(tmp_path):
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
        assert [item.instance_id for item in session.instances] == [
            "arm_1",
            "base_1",
        ]
        assert session.world.control_hz == 20
    finally:
        session.close()


def test_a_shared_world_refuses_tasks_and_policies_it_cannot_run(tmp_path):
    from physai.runtime import create_session

    manifest = _manifest(
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
    )

    with pytest.raises(ValueError, match="do not run tasks or policies"):
        create_session(manifest)


def test_only_the_direct_backend_is_built_here(tmp_path):
    from physai.runtime import create_session

    manifest = _manifest(
        tmp_path, backend="ros2_sim", robots=[{"id": "a", "robot": "so101"}]
    )

    with pytest.raises(ValueError, match="not supported by create_session"):
        create_session(manifest)


@requires_assets
@requires_turtlebot_assets
def test_the_heterogeneous_example_manifest_builds():
    from physai.config import load_manifest
    from physai.runtime import create_session

    session = create_session(
        load_manifest("configs/manifests/example_heterogeneous.yaml")
    )
    try:
        assert len(session.instances) == 2
    finally:
        session.close()


@requires_assets
def test_a_scene_override_replaces_a_robot_scene_default(tmp_path):
    from physai.runtime import create_session

    session = create_session(
        _manifest(
            tmp_path,
            scene={"overrides": {"robot_xml": SO101_MODEL, "camera_width": 64}},
            robots=[_arm()],
        )
    )
    try:
        scene = session.runtime.robot.cfg.scene
        assert scene.camera_width == 64
        assert scene.robot_xml.name == "so101_new_calib_camera.xml"
    finally:
        session.close()
