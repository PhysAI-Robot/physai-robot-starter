from pathlib import Path

import mujoco
import numpy as np
import pytest

from physai.robots import shared_attach
from physai.sim import RobotInstanceConfig, SharedWorld
from physai.web.telemetry import build_scene_manifest

ROOT = Path(__file__).resolve().parents[3]
SO101_MODEL = ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"
TURTLEBOT_MODEL = ROOT / "assets" / "turtlebot4" / "turtlebot4.xml"


pytestmark = pytest.mark.skipif(
    not SO101_MODEL.exists() or not TURTLEBOT_MODEL.exists(),
    reason="robot assets are not available",
)


def make_world() -> SharedWorld:
    return SharedWorld(
        (
            RobotInstanceConfig(
                "arm_1", SO101_MODEL, "so101", position=(0.3, 0.0, 0.0)
            ),
            RobotInstanceConfig(
                "base_1", TURTLEBOT_MODEL, "turtlebot4", position=(-0.3, 0.0, 0.1)
            ),
        ),
        control_hz=10.0,
        shared_attach=shared_attach,
    )


def test_instances_share_one_model_and_clock_and_their_controls_are_isolated():
    world = make_world()

    assert set(world.bindings) == {"arm_1", "base_1"}
    assert world.bindings["arm_1"].prefix == "arm_1__"
    assert world.bindings["base_1"].prefix == "base_1__"
    assert world.model.nbody > 1
    assert world.step_count == 0

    world.step()

    assert world.step_count == 1
    assert world.data.time == pytest.approx(world.model.opt.timestep * world.n_substeps)

    arm_actuator = world.bindings["arm_1"].actuator_ids["shoulder_pan"]
    base_actuator = world.bindings["base_1"].actuator_ids["forward"]
    before = world.data.ctrl.copy()

    world.set_controls("base_1", {"forward": 0.2, "turn": -0.1})

    assert world.data.ctrl[arm_actuator] == before[arm_actuator]
    assert world.data.ctrl[base_actuator] == pytest.approx(0.2)

    with pytest.raises(KeyError, match="unknown robot instance"):
        world.set_controls("missing", {"forward": 0.1})
    with pytest.raises(KeyError, match="no actuator"):
        world.set_controls("base_1", {"shoulder_pan": 0.1})
    with pytest.raises(ValueError, match="not finite"):
        world.set_controls("base_1", {"forward": np.nan})


def test_a_shared_world_keeps_its_meshes_visible_and_the_arm_cameras():
    world = SharedWorld(
        (RobotInstanceConfig("base_1", TURTLEBOT_MODEL, "turtlebot4"),),
        shared_attach=shared_attach,
    )
    manifest = build_scene_manifest(
        world.model,
        instance_prefixes={"base_1": "base_1__"},
    )

    meshes = [
        item
        for item in manifest["geometries"]
        if item["instance_id"] == "base_1" and item["type"] == "mesh"
    ]
    assert meshes
    assert all(item["visual"] for item in meshes)

    arm_world = SharedWorld(
        (RobotInstanceConfig("arm_1", SO101_MODEL, "so101"),),
        shared_attach=shared_attach,
    )
    camera_names = {
        mujoco.mj_id2name(arm_world.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(arm_world.model.ncam)
    }
    assert camera_names == {"arm_1__front", "arm_1__wrist"}
