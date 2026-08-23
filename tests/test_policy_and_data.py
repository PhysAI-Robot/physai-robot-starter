import numpy as np

from conftest import requires_assets

pytestmark = requires_assets


@requires_assets
def test_expert_state_machine_reaches_the_end(env):
    from physai.policy import Phase, ScriptedPickPlace

    obs = env.reset(seed=0)
    policy = ScriptedPickPlace(env.kin, env)
    policy.reset(obs)
    seen = {policy.phase}
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(policy.act(obs))
        seen.add(policy.phase)
        if policy.done:
            break
    assert Phase.CLOSE in seen and Phase.SQUEEZE in seen and Phase.LIFT in seen


@requires_assets
def test_expert_actually_closes_on_the_cube(env):
    """The grasp must be a real contact, not the jaws shutting on empty air."""
    import mujoco

    from physai.policy import Phase, ScriptedPickPlace

    obs = env.reset(seed=0)
    policy = ScriptedPickPlace(env.kin, env)
    policy.reset(obs)
    pad_contact = False
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(policy.act(obs))
        if policy.phase in (Phase.SQUEEZE, Phase.LIFT):
            for c in range(env.data.ncon):
                names = {
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in (env.data.contact[c].geom1, env.data.contact[c].geom2)
                }
                if "cube_geom" in names and names & {"pad_static", "pad_moving"}:
                    pad_contact = True
        if policy.done:
            break
    assert pad_contact, "expert never made pad-to-cube contact"


@requires_assets
def test_recorder_writes_lerobot_shaped_arrays(env, tmp_path):
    from physai.data import EpisodeRecorder, load_episode
    from physai.policy import ScriptedPickPlace

    rec = EpisodeRecorder(tmp_path, fps=env.cfg.control_hz, store_images=False)
    obs = env.reset(seed=1)
    policy = ScriptedPickPlace(env.kin, env)
    policy.reset(obs)
    rec.start_episode()
    for _ in range(25):
        action = policy.act(obs)
        prev = obs
        obs, reward, term, trunc, _ = env.step(action)
        rec.record(prev, action, reward=reward, done=term or trunc,
                   phase=policy.phase.name,
                   gripper_joint=env.gripper_to_joint(action.gripper.clipped()))
    path = rec.end_episode(success=False)
    rec.write_meta()

    data = load_episode(path)
    assert data["observation.state"].shape == (25, 6)
    assert data["action"].shape == (25, 6)
    # Actions must be absolute joint targets in radians, same units as state —
    # a delta action space here would silently break transfer to hardware.
    assert np.abs(data["action"]).max() < 4.0


def test_recorder_accepts_twist_actions(tmp_path):
    from physai.contracts import Action, JointState, Observation, Twist, Vector3
    from physai.data import EpisodeRecorder, load_episode

    observation = Observation(
        joint_state=JointState(name=("left_wheel", "right_wheel"),
                               position=np.zeros(2)),
    )
    action = Action(ee_twist=Twist(linear=Vector3(x=0.2)))
    recorder = EpisodeRecorder(tmp_path, robot_type="turtlebot4",
                               store_images=False)
    recorder.start_episode()
    recorder.record(observation, action)
    path = recorder.end_episode(success=False)
    recorder.write_meta()

    data = load_episode(path)
    assert data["observation.state"].shape == (1, 2)
    assert data["action"].shape == (1, 6)
    assert '"robot_type": "turtlebot4"' in (tmp_path / "meta.json").read_text()


@requires_assets
def test_replay_policy_reproduces_recorded_actions(env):
    from physai.policy import ReplayPolicy

    obs = env.reset(seed=2)
    recorded = np.tile(
        np.concatenate([obs.joint_state.position[:5], [env.gripper_to_joint(0.5)]]),
        (10, 1),
    )
    policy = ReplayPolicy(env, recorded)
    policy.reset(obs)
    action = policy.act(obs)
    np.testing.assert_allclose(action.joint_position, recorded[0][:5])
    assert action.gripper.clipped() == 0.5


@requires_assets
def test_plan_runner_executes_a_scripted_plan(env):
    from physai.planner import ScriptedPlanner
    from physai.policy.plan_runner import PlanRunner

    obs = env.reset(seed=0)
    plan = ScriptedPlanner(env.cube_pos, env.target_pos).plan("test", obs)
    runner = PlanRunner(env.kin, plan, dt=env.control_dt)
    runner.reset(obs)
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(runner.act(obs))
        runner.note_progress(env.kin.pinch_center(env.data),
                             env.joint_to_gripper(obs.joint_state.position[5]))
        if runner.done:
            break
    assert runner.index > 0, "plan runner never completed a sub-goal"


def test_ros2_contract_is_self_consistent():
    from physai.bridge import ALL_ENDPOINTS, EXTERNAL_INPUTS, Direction

    published = {e.topic for e in ALL_ENDPOINTS if e.direction is Direction.PUBLISH}
    subscribed = {e.topic for e in ALL_ENDPOINTS if e.direction is Direction.SUBSCRIBE}
    orphans = subscribed - published - EXTERNAL_INPUTS
    assert not orphans, f"subscribed but nothing publishes: {sorted(orphans)}"

    types = {}
    for e in ALL_ENDPOINTS:
        types.setdefault(e.topic, set()).add(e.msg_type)
    clashes = {t: v for t, v in types.items() if len(v) > 1}
    assert not clashes, f"topic type mismatch: {clashes}"
