"""SO-101's shared-world (multi-robot) instance adapter.

Owns everything SO-101-specific about running inside a `SharedWorld`:
the extra cameras a shared attachment needs (the standalone model already
has them; the shared-world attachment path does not), the per-instance
kinematics/joint mapping, and command translation. Registered with
`robots.registry` as so101's `shared_attach`/`shared_instance` factories
so `web/world_runtime.py` never branches on a robot name to get here.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Action, GripperCommand, Header, JointState, Observation
from ...control.resolver import TwistToJointResolver
from ..base import RobotSpec
from .contracts import ALL_JOINT_NAMES, ARM_JOINT_NAMES
from .kinematics import ArmKinematics


def so101_shared_attach(child_spec) -> None:
    """Add the front/wrist cameras a shared-world attachment needs.

    The standalone SO-101 model (`SO101Env`) defines these itself; a shared
    world attaches the bare arm model and needs them added at attach time.
    """
    child_spec.worldbody.add_camera(
        name="front",
        pos=[0.62, 0.0, 0.38],
        xyaxes=[0.0, 1.0, 0.0, -0.45, 0.0, 0.9],
        fovy=48,
    )
    camera_body = next(
        (body for body in child_spec.bodies if body.name == "wrist_camera"),
        None,
    )
    if camera_body is not None:
        camera_body.add_camera(
            name="wrist",
            pos=[0.0, 0.0, 0.025],
            xyaxes=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            fovy=62,
        )


class SO101SharedInstance:
    """SO-101's view over one binding in a shared world."""

    def __init__(self, world, config) -> None:
        self.world = world
        self.config = config
        self.binding = world.binding(config.instance_id)
        self.prefix = self.binding.prefix
        qualified = tuple(self.prefix + name for name in ARM_JOINT_NAMES)
        self.kin = ArmKinematics(
            world.model,
            ee_site=self.prefix + "gripperframe",
            joint_names=qualified,
        )
        self._gripper_joint = self.binding.joint_ids["gripper"]
        self._gripper_limits = world.model.jnt_range[self._gripper_joint]
        self._resolver = TwistToJointResolver(self.kin, world.data, dt=0.04)
        self.robot_spec = RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=ALL_JOINT_NAMES,
            action_joint_names=ARM_JOINT_NAMES,
            action_modes=("joint_position",),
            observation_modalities=("state", "ee_pose"),
            capabilities=("joint_position", "arm_kinematics", "gripper"),
            joint_limits={
                name: tuple(float(value) for value in world.model.jnt_range[joint_id])
                for name, joint_id in zip(
                    ARM_JOINT_NAMES,
                    [self.binding.joint_ids[name] for name in ARM_JOINT_NAMES],
                )
            },
            max_joint_delta={name: 0.5 for name in ARM_JOINT_NAMES},
            joint_state_frame=f"{config.instance_id}/base",
            camera_frames={
                "front": self.prefix + "front",
                "wrist": self.prefix + "wrist",
            },
        )
        self._last_action = Action(
            joint_position=np.zeros(len(ARM_JOINT_NAMES)),
            gripper=GripperCommand(),
        )

    def reset(self) -> None:
        qpos = np.array([0.0, -1.05, 1.25, 0.75, 0.0])
        for name, value in zip(ARM_JOINT_NAMES, qpos):
            self.world.data.qpos[self.binding.qpos_addresses[name]] = value
        self.world.data.qpos[self.binding.qpos_addresses["gripper"]] = float(
            self._gripper_limits[1]
        )
        self._last_action = Action(joint_position=qpos, gripper=GripperCommand())

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
            ee_pose=self.kin.fk(self.world.data),
        )

    def prepare_action(self, action: Action) -> Action:
        if action.mode == "twist":
            action = self._resolver(
                action.ee_twist, self._joint_state(), action.gripper
            )
        self.robot_spec.validate_action(action)
        return action

    def submit(self, action: Action) -> None:
        action = self.prepare_action(action)
        for name, value in zip(ARM_JOINT_NAMES, action.joint_position):
            self.world.set_controls(self.config.instance_id, {name: float(value)})
        gripper = action.gripper or GripperCommand()
        lo, hi = self._gripper_limits
        self.world.set_controls(
            self.config.instance_id,
            {"gripper": float(lo + gripper.clipped() * (hi - lo))},
        )
        self._last_action = action

    def hold(self) -> None:
        self.submit(self._last_action)


__all__ = ["SO101SharedInstance", "so101_shared_attach"]
