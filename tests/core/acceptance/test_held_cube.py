"""A cube held by the gripper stays held, on every robot and every physics engine.

Every engine models friction its own way, and MuJoCo's is soft: under a constant load a held
object creeps out of the fingers even when the friction force is a fraction of its limit. An
engine counts as supported for a robot only when this passes, so research can trust a grasp
on it. The grasp is core's own `ScriptedPlanner` run by `PlanRunner`; the measure is the
cube's position against the pinch point, which every engine reports.

The cube first seats itself between the jaws (up to about 2.5 mm in 10 s), then must not move: drift
is measured over the 60 s after that.
"""

import importlib.util

import numpy as np
import pytest
from tests.conftest import requires_assets, requires_viperx300s_assets

HOLD_SECONDS = 60.0
SETTLE_SECONDS = 10.0  # the cube turns and seats itself between the jaws; not creep
SEEDS = (0, 1, 2, 3)
MAX_CREEP = 0.002  # metres the cube may move against the pinch point over HOLD_SECONDS
LIFTED = 0.045  # metres: the cube counts as lifted above this height
MIN_HEIGHT = 0.04  # a cube that fell is back down at about 0.035

MANIFESTS = {
    "so101": "configs/manifests/so101_single_cube_fixed_place.yaml",
    "viperx300s": "configs/manifests/viperx300s_single_cube_fixed_place.yaml",
}
REQUIRES = {"so101": requires_assets, "viperx300s": requires_viperx300s_assets}


def _engine_available(engine: str) -> bool:
    # find_spec, never an import: importing newton loads warp, and Isaac Sim must start
    # before warp does (docs/DECISIONS.md, I).
    if engine == "mujoco":
        return True
    if engine == "newton":
        return importlib.util.find_spec("newton") is not None
    isaacsim = importlib.util.find_spec("isaacsim")
    return isaacsim is not None and isaacsim.origin is not None


def _cases():
    # Isaac first: it must start before anything loads warp (Newton).
    for engine in ("isaac", "newton", "mujoco"):
        for robot in ("so101", "viperx300s"):
            marks = [REQUIRES[robot], pytest.mark.slow, pytest.mark.acceptance]
            if engine != "mujoco":
                marks.append(getattr(pytest.mark, engine))
            if not _engine_available(engine):
                marks.append(
                    pytest.mark.skip(
                        reason=(
                            "uv sync --extra isaac installs isaacsim"
                            if engine == "isaac"
                            else "uv sync installs newton"
                        )
                    )
                )
            yield pytest.param(robot, engine, id=f"{robot}-{engine}", marks=marks)


def _grasp_and_lift(env, seed):
    """Run core's scripted pick until the cube is off the table, then settle."""
    from physai.planner import ScriptedPlanner
    from physai.policy.plan_runner import PlanRunner

    obs = env.reset(seed=seed)
    profile = env.grasp_profile()
    plan = ScriptedPlanner(env.cube_pos, env.target_pos, profile).plan("", obs)
    runner = PlanRunner(
        env.kin,
        plan,
        dt=1.0 / env.cfg.control_hz,
        approach_dir=profile.approach_dir,
        ik_kwargs=profile.ik_kwargs,
    )
    runner.reset(obs)
    joints = len(env.kin.joint_names)
    grasped = False
    for _ in range(env.cfg.max_steps):
        runner.note_progress(
            env.pinch_center(), env.joint_to_gripper(obs.joint_state.position[joints])
        )
        obs, *_rest = env.step(runner.act(obs))
        grasped = grasped or any(
            sg.skill == "grasp" for sg in plan.subgoals[: runner.index]
        )
        if grasped and env.cube_pos[2] > LIFTED:
            return obs, profile, joints
    raise AssertionError("the scripted pick never lifted the cube")


@pytest.mark.parametrize("robot,engine", list(_cases()))
def test_a_held_cube_stays_in_the_fingers(robot, engine):
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.contracts import Action, GripperCommand
    from physai.runtime import create_session

    manifest = with_overrides(
        load_manifest(MANIFESTS[robot]),
        simulator=engine,
        policy="idle",
        seed=0,
        max_steps=800,
    )
    session = create_session(manifest, render=False)
    try:
        env = session.runtime.robot
        creeps, fell = [], []
        for seed in SEEDS:
            obs, profile, joints = _grasp_and_lift(env, seed)
            hold = Action(
                joint_position=np.array(
                    obs.joint_state.position[:joints], dtype=np.float64
                ),
                gripper=GripperCommand(profile.grip_hold),
            )
            for _ in range(int(SETTLE_SECONDS * env.cfg.control_hz)):
                env.step(
                    hold
                )  # the lift's motion dies out and the contacts take the load
            start = env.cube_pos - env.pinch_center()
            for _ in range(int(HOLD_SECONDS * env.cfg.control_hz)):
                env.step(hold)
            moved = (env.cube_pos - env.pinch_center()) - start
            creeps.append(1000 * float(np.linalg.norm(moved)))
            fell.append(env.cube_pos[2] <= MIN_HEIGHT)
        report = ", ".join(f"seed {k}: {c:.2f} mm" for k, c in zip(SEEDS, creeps))
        print(f"{robot} on {engine}, cube moved in {HOLD_SECONDS:.0f} s: {report}")
        assert not any(fell), f"the cube fell ({report})"
        assert max(creeps) < 1000 * MAX_CREEP, report
    finally:
        session.close()
