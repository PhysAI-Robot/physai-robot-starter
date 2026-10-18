"""ViperX 300 S (ALOHA follower arm): registration, joint contract, scene, cameras, IK."""

import numpy as np
import pytest

from tests.conftest import requires_viperx300s_assets

pytestmark = [pytest.mark.acceptance]


def test_viperx300s_is_registered_as_a_mujoco_only_fixed_base_manipulator():
    from physai.robots.registry import (
        available_robots,
        available_simulators,
        create_robot,
        default_task,
        robot_kind,
        scene_defaults,
    )

    assert "viperx300s" in available_robots()
    assert robot_kind("viperx300s") == "fixed_base_manipulator"
    assert default_task("viperx300s") == "single_cube_place"
    assert available_simulators("viperx300s") == ("mujoco", "isaac")
    defaults = scene_defaults("viperx300s")
    assert defaults["robot_xml"].name == "vx300s.xml"
    assert defaults["description"].ee_site == "pinch"
    with pytest.raises(ValueError, match="does not support simulator"):
        create_robot("viperx300s", simulator="not-a-simulator")


def test_the_isaac_config_defaults_to_this_robot_not_the_so101():
    from physai.robots import create_env_config

    config = create_env_config("viperx300s", simulator="isaac")
    assert config.assets_root.name == "viperx300s"
    assert config.cameras == ("front", "wrist")
    assert config.gravity_compensation is True
    assert config.gripper_force_limit is None
    assert tuple(config.cube_x_range) == (0.28, 0.38)


def test_action_contract_has_six_arm_joints_and_a_finger():
    from physai.contracts import Action
    from physai.robots.viperx300s import (
        ALL_JOINT_NAMES,
        ARM_JOINT_NAMES,
        viperx300s_action_encoder,
        viperx300s_training_contract,
    )

    assert len(ARM_JOINT_NAMES) == 6
    assert ALL_JOINT_NAMES[-1] == "gripper"
    values, names = viperx300s_action_encoder(
        Action(joint_position=np.zeros(6)), gripper_joint=0.04
    )
    assert names == ALL_JOINT_NAMES
    assert values.shape == (7,) and values[-1] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="6 arm targets"):
        viperx300s_action_encoder(
            Action(joint_position=np.zeros(5)), gripper_joint=0.04
        )
    contract = viperx300s_training_contract(
        camera_config={"front": {"width": 64, "height": 48}, "wrist": {}}
    )
    assert contract.action_spec.fields[0].shape == (7,)
    assert [c.name for c in contract.observation_spec.cameras] == ["front", "wrist"]


@pytest.fixture
def viperx300s():
    from physai.robots.viperx300s import EnvConfig, ViperX300SEnv

    robot = ViperX300SEnv(EnvConfig(seed=0, render=False, max_steps=200))
    yield robot
    robot.close()


@requires_viperx300s_assets
def test_reset_reports_seven_joints_and_open_fingers(viperx300s):
    from physai.robots.viperx300s import ALL_JOINT_NAMES

    observation = viperx300s.reset(seed=0)
    state = observation.joint_state
    assert state.name == ALL_JOINT_NAMES
    assert state.position.shape == (7,)
    assert viperx300s.joint_to_gripper(state.position[-1]) == pytest.approx(
        1.0, abs=0.02
    )
    assert observation.ee_pose is not None


@requires_viperx300s_assets
def test_home_pose_touches_nothing_but_the_floor_and_the_cube_on_the_table(viperx300s):
    import mujoco

    viperx300s.reset(seed=0)
    model, data = viperx300s.model, viperx300s.data

    def geom(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    floor, table, cube = geom("physai_floor"), geom("table_top"), geom("cube_geom")
    for index in range(data.ncon):
        pair = {data.contact[index].geom1, data.contact[index].geom2}
        if floor in pair or pair == {table, cube}:
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair]
        raise AssertionError(f"unexpected contact at the home pose: {names}")


@requires_viperx300s_assets
def test_cube_rests_on_the_table_and_the_gripper_closes_and_opens(viperx300s):
    from physai.contracts import Action, GripperCommand
    from physai.robots.viperx300s import HOME_QPOS

    observation = viperx300s.reset(seed=0)

    def hold(gripper: float, steps: int):
        nonlocal observation
        for _ in range(steps):
            observation, *_ = viperx300s.step(
                Action(
                    joint_position=HOME_QPOS.copy(),
                    gripper=GripperCommand(position=gripper),
                )
            )
        return viperx300s.joint_to_gripper(observation.joint_state.position[-1])

    assert viperx300s.cube_pos[2] == pytest.approx(viperx300s.rest_z, abs=0.003)
    assert hold(0.0, 40) < 0.1
    assert hold(1.0, 40) > 0.9


@requires_viperx300s_assets
def test_every_cube_position_in_the_default_range_has_a_top_down_grasp_pose(viperx300s):
    from physai.robots.viperx300s import HOME_QPOS

    rng = np.random.default_rng(0)
    cfg = viperx300s.cfg
    q = HOME_QPOS.copy()
    for _ in range(25):
        x = rng.uniform(*cfg.cube_x_range)
        y = rng.uniform(*cfg.cube_y_range)
        target = np.array([x, y, viperx300s.rest_z])
        result = viperx300s.kin.ik_pinch(target, q_init=q)
        assert result.converged, f"no top-down pose at {target}"
        assert result.position_error < 2e-3
        assert result.axis("x")[2] < -0.99, "gripper is not pointing down"
        # fingers close along the world y axis, so a square cube is gripped on faces
        assert abs(result.axis("y")[1]) > 0.99
        q = result.qpos


@requires_viperx300s_assets
def test_task_runtime_accepts_the_robot(viperx300s):
    from physai.tasks import TaskRuntime, create_task

    runtime = TaskRuntime(viperx300s, create_task("single_cube_place"))
    runtime.reset(seed=0)
    assert runtime.robot_spec.name == "viperx300s"


@requires_viperx300s_assets
@pytest.mark.slow
def test_front_and_wrist_cameras_carry_signal_and_the_wrist_sees_the_fingers():
    import mujoco

    from physai.robots.viperx300s import EnvConfig, ViperX300SEnv

    robot = ViperX300SEnv(EnvConfig(seed=0, render=True, max_steps=200))
    try:
        observation = robot.reset(seed=0)
        assert set(observation.images) == {"front", "wrist"}
        for name, frame in observation.images.items():
            assert frame.data.std() > 15.0, f"{name} camera is nearly uniform"
        # the wrist camera looks along the gripper, toward the fingertips
        cam = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        view = -robot.data.cam_xmat[cam].reshape(3, 3)[:, 2]
        to_pinch = robot.pinch_center() - robot.data.cam_xpos[cam]
        to_pinch /= np.linalg.norm(to_pinch)
        assert float(view @ to_pinch) > 0.7
    finally:
        robot.close()
