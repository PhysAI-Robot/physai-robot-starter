import mujoco
import numpy as np

from physai.contracts import Action
from physai.web.runtime import action_from_payload
from physai.web.telemetry import build_scene_manifest, build_state_snapshot

XML = """
<mujoco>
  <worldbody>
        <body name="arm" pos="0 0 0.2" quat="0.7071068 0 0 0.7071068">
            <joint name="arm_hinge" type="hinge" axis="0 0 1"/>
      <geom name="arm_box" type="box" size="0.1 0.02 0.02" rgba="0.2 0.4 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_scene_manifest_describes_static_geometry():
    model = mujoco.MjModel.from_xml_string(XML)

    manifest = build_scene_manifest(model, robot="test")

    assert manifest["type"] == "scene"
    assert manifest["robot"] == "test"
    geometry = manifest["geometries"][0]
    assert geometry["id"] == 0
    assert geometry["name"] == "arm_box"
    assert geometry["type"] == "box"
    assert geometry["body"] == "arm"
    np.testing.assert_allclose(geometry["size"], [0.1, 0.02, 0.02])
    np.testing.assert_allclose(geometry["rgba"], [0.2, 0.4, 0.8, 1.0])
    assert geometry["asset"] is None
    assert geometry["visual"] is False


def test_state_snapshot_uses_three_js_quaternion_order():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    snapshot = build_state_snapshot(model, data, step=4, robot="test")

    assert snapshot["step"] == 4
    assert snapshot["sim_time"] == 0.0
    assert snapshot["joints"] == [{"id": 0, "name": "arm_hinge", "qpos": [0.0]}]
    np.testing.assert_allclose(
        snapshot["bodies"][0]["quaternion"],
        [0.0, 0.0, 0.7071068, 0.7071068],
        atol=1e-6,
    )
    assert snapshot["geometries"][0]["position"] == [0.0, 0.0, 0.2]


def test_browser_commands_map_to_shared_actions():
    joint_action = action_from_payload(
        {
            "mode": "joint_position",
            "position": [0.1, 0.2],
            "names": ["joint_a", "joint_b"],
            "gripper": 0.25,
        }
    )
    assert isinstance(joint_action, Action)
    assert joint_action.mode == "joint_position"
    assert joint_action.joint_names == ("joint_a", "joint_b")
    assert joint_action.gripper.position == 0.25

    twist_action = action_from_payload(
        {"mode": "twist", "linear": {"x": 0.2}, "angular": {"z": -0.3}}
    )
    assert twist_action.mode == "twist"
    assert twist_action.ee_twist.linear.x == 0.2
    assert twist_action.ee_twist.angular.z == -0.3


def test_websocket_route_resolves_fastapi_websocket_annotation():
    from physai.robots import RobotSpec
    from physai.web.app import create_app
    from physai.web.runtime import SimulationHost
    from tests.support.fakes import FakeRobotPort

    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    app = create_app(host=SimulationHost(FakeRobotPort(spec), robot_name="test"))
    websocket_route = next(route for route in app.routes if route.path == "/ws")

    assert websocket_route.endpoint.__annotations__["websocket"].__name__ == "WebSocket"
