"""TurtleBot4's shared-world (multi-robot) instance adapter.

Registered with `robots.registry` as turtlebot4's `shared_instance` factory
so `web/world_runtime.py` never branches on a robot name to get here.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Action, Header, JointState, Observation
from ..base import RobotSpec


class TurtleBot4SharedInstance:
    """TurtleBot4's view over one binding in a shared world."""

    def __init__(self, world, config) -> None:
        self.world = world
        self.config = config
        self.binding = world.binding(config.instance_id)
        self.prefix = self.binding.prefix
        self.robot_spec = RobotSpec(
            name="turtlebot4",
            kind="mobile_base",
            joint_names=("left_wheel", "right_wheel"),
            action_joint_names=(),
            action_modes=("twist",),
            observation_modalities=("state",),
            capabilities=("base_velocity", "odometry"),
            joint_state_frame=f"{config.instance_id}/base_link",
            action_frame="base",
            camera_frames={},
            units={
                "position": "m",
                "linear_velocity": "m/s",
                "angular_velocity": "rad/s",
            },
        )
        self._last_action = Action()

    def reset(self) -> None:
        joint_id = self.binding.joint_ids["floating_base_joint"]
        address = int(self.world.model.jnt_qposadr[joint_id])
        self.world.data.qpos[address : address + 7] = (
            *self.config.position,
            *self.config.quaternion,
        )
        self._last_action = Action()

    def _joint_state(self) -> JointState:
        names = self.robot_spec.joint_names
        positions = []
        velocities = []
        efforts = []
        for name in names:
            joint_id = self.binding.joint_ids[name]
            qpos_address = int(self.world.model.jnt_qposadr[joint_id])
            dof_address = int(self.world.model.jnt_dofadr[joint_id])
            positions.append(float(self.world.data.qpos[qpos_address]))
            velocities.append(float(self.world.data.qvel[dof_address]))
            actuator_id = self.binding.actuator_ids.get(name)
            efforts.append(
                float(self.world.data.actuator_force[actuator_id])
                if actuator_id is not None
                else 0.0
            )
        return JointState(
            name=names,
            position=np.asarray(positions),
            velocity=np.asarray(velocities),
            effort=np.asarray(efforts),
            header=Header(
                stamp=float(self.world.data.time),
                frame_id=self.robot_spec.joint_state_frame or "",
            ),
        )

    def observe(self) -> Observation:
        return Observation(
            joint_state=self._joint_state(),
            step=self.world.step_count,
            sim_time=float(self.world.data.time),
            ee_pose=None,
        )

    def prepare_action(self, action: Action) -> Action:
        self.robot_spec.validate_action(action)
        return action

    def submit(self, action: Action) -> None:
        action = self.prepare_action(action)
        twist = action.ee_twist
        self.world.set_controls(
            self.config.instance_id,
            {"forward": float(twist.linear.x), "turn": float(twist.angular.z)},
        )
        self._last_action = action

    def hold(self) -> None:
        self.world.set_controls(self.config.instance_id, {"forward": 0.0, "turn": 0.0})


__all__ = ["TurtleBot4SharedInstance"]
