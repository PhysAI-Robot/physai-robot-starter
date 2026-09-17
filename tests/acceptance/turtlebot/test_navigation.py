import numpy as np
import pytest

pytestmark = [pytest.mark.acceptance, pytest.mark.slow]


def test_turtlebot4_reset_is_deterministic_for_a_given_seed():
    from physai.robots import create_robot

    env = create_robot("turtlebot4", render=False)
    try:
        first = env.reset(seed=7)
        second = env.reset(seed=7)
        np.testing.assert_allclose(
            first.joint_state.position, second.joint_state.position
        )
        np.testing.assert_allclose(
            first.ee_pose.pose.position.as_array(),
            second.ee_pose.pose.position.as_array(),
        )
        assert first.step == second.step == 0
        assert first.sim_time == second.sim_time == 0.0
    finally:
        env.close()


def test_turtlebot4_lidar_detects_configured_obstacle():
    from physai.robots.turtlebot import TurtleBot4Config, TurtleBot4Env

    env = TurtleBot4Env(
        TurtleBot4Config(
            render=False,
            obstacles=((0.0, -0.8, 0.3, 0.1, 0.4),),
        )
    )
    try:
        env.reset(seed=0)
        ranges = env.lidar_ranges()
        forward_index = (
            round(
                (0.0 - env.cfg.lidar_angle_min) / (2.0 * np.pi) * env.cfg.lidar_samples
            )
            % env.cfg.lidar_samples
        )
        assert ranges[forward_index] < 0.8
    finally:
        env.close()


def test_turtlebot4_collision_counter_detects_obstacle_contact():
    from physai.contracts import Action, Twist, Vector3
    from physai.robots.turtlebot import TurtleBot4Config, TurtleBot4Env

    env = TurtleBot4Env(
        TurtleBot4Config(
            render=False,
            obstacles=((0.0, -0.5, 0.4, 0.1, 0.4),),
        )
    )
    try:
        env.reset(seed=0)
        action = Action(ee_twist=Twist(linear=Vector3(x=0.4)))
        for _ in range(400):
            env.step(action)
        assert env.collision_count > 0
        assert env.non_ground_contact_count() > 0
    finally:
        env.close()


def test_turtlebot4_stays_on_the_ground_while_driving():
    from physai.contracts import Action, Twist, Vector3
    from physai.robots import create_robot

    env = create_robot("turtlebot4", render=False)
    try:
        env.reset(seed=0)
        start_height = env.data.xpos[env._base_body_id][2]
        action = Action(ee_twist=Twist(linear=Vector3(x=0.4), angular=Vector3(z=0.4)))
        for _ in range(200):
            env.step(action)
        position = env.data.xpos[env._base_body_id]
        assert abs(position[2] - start_height) < 0.1
        assert float(position[0] ** 2 + position[1] ** 2) ** 0.5 > 0.5
    finally:
        env.close()


def test_turtlebot4_rpp_reaches_deterministic_goal():
    from physai.robots.turtlebot import NavigationGoal, navigate_to_goal

    result = navigate_to_goal(NavigationGoal(x=1.0, y=-1.0), seed=0)
    assert result.reached
    assert result.steps < 100
    assert result.position_error <= 0.10
    assert result.heading_error <= 0.15
    assert result.collision_count == 0
    assert result.failure_reason is None


def test_turtlebot4_rpp_goal_result_is_reproducible():
    from physai.robots.turtlebot import NavigationGoal, navigate_to_goal

    goal = NavigationGoal(x=1.0, y=-1.0)
    first = navigate_to_goal(goal, seed=7)
    second = navigate_to_goal(goal, seed=7)
    assert first == second


def test_turtlebot4_published_image_is_not_blank():
    from physai.contracts import Action, Twist, Vector3
    from physai.robots import create_robot

    env = create_robot("turtlebot4", render=True)
    try:
        env.reset(seed=0)
        assert env.robot_spec.supports("images")
        for _ in range(60):
            env.step(
                Action(ee_twist=Twist(linear=Vector3(x=0.4), angular=Vector3(z=0.4)))
            )
        frame = env.observe().images["free"].data.astype(float)
        assert frame.std() > 1.0
    finally:
        env.close()
