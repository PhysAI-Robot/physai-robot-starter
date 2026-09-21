"""Web runtime for multiple robot instances in one MuJoCo world."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from ..contracts import Action, GripperCommand, Header, JointState, Observation
from ..control.resolver import TwistToJointResolver
from ..robots.base import RobotSpec
from ..robots.so101.contracts import ALL_JOINT_NAMES, ARM_JOINT_NAMES
from ..robots.so101.kinematics import ArmKinematics
from ..sim.world import RobotInstanceConfig, SharedWorld
from .telemetry import build_scene_manifest, build_state_snapshot


@dataclass
class SharedRobotInstance:
    """Embodiment-local view over one binding in a shared world."""

    world: SharedWorld
    config: RobotInstanceConfig

    def __post_init__(self) -> None:
        self.binding = self.world.binding(self.config.instance_id)
        self.prefix = self.binding.prefix
        if self.config.robot_name == "so101":
            qualified = tuple(self.prefix + name for name in ARM_JOINT_NAMES)
            self.kin = ArmKinematics(
                self.world.model,
                ee_site=self.prefix + "gripperframe",
                joint_names=qualified,
            )
            self._gripper_joint = self.binding.joint_ids["gripper"]
            self._gripper_limits = self.world.model.jnt_range[self._gripper_joint]
            self._resolver = TwistToJointResolver(
                self.kin,
                self.world.data,
                dt=0.04,
            )
            self.robot_spec = RobotSpec(
                name="so101",
                kind="fixed_base_manipulator",
                joint_names=ALL_JOINT_NAMES,
                action_joint_names=ARM_JOINT_NAMES,
                action_modes=("joint_position",),
                observation_modalities=("state", "ee_pose"),
                capabilities=("joint_position", "arm_kinematics", "gripper"),
                joint_limits={
                    name: tuple(
                        float(value) for value in self.world.model.jnt_range[joint_id]
                    )
                    for name, joint_id in zip(
                        ARM_JOINT_NAMES,
                        [self.binding.joint_ids[name] for name in ARM_JOINT_NAMES],
                    )
                },
                max_joint_delta={name: 0.5 for name in ARM_JOINT_NAMES},
                joint_state_frame=f"{self.config.instance_id}/base",
                camera_frames={
                    "front": self.prefix + "front",
                    "wrist": self.prefix + "wrist",
                },
            )
            self._last_action = Action(
                joint_position=np.zeros(len(ARM_JOINT_NAMES)),
                gripper=GripperCommand(),
            )
        elif self.config.robot_name == "turtlebot4":
            self.robot_spec = RobotSpec(
                name="turtlebot4",
                kind="mobile_base",
                joint_names=("left_wheel", "right_wheel"),
                action_joint_names=(),
                action_modes=("twist",),
                observation_modalities=("state",),
                capabilities=("base_velocity", "odometry"),
                joint_state_frame=f"{self.config.instance_id}/base_link",
                action_frame="base",
                camera_frames={},
                units={
                    "position": "m",
                    "linear_velocity": "m/s",
                    "angular_velocity": "rad/s",
                },
            )
            self._last_action = Action()
        else:
            raise ValueError(
                f"shared-world web runtime does not support robot {self.config.robot_name!r}"
            )

    def reset(self) -> None:
        if self.config.robot_name == "so101":
            qpos = np.array([0.0, -1.05, 1.25, 0.75, 0.0])
            for name, value in zip(ARM_JOINT_NAMES, qpos):
                self.world.data.qpos[self.binding.qpos_addresses[name]] = value
            self.world.data.qpos[self.binding.qpos_addresses["gripper"]] = float(
                self._gripper_limits[1]
            )
            self._last_action = Action(
                joint_position=qpos,
                gripper=GripperCommand(),
            )
        else:
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
            ee_pose=(
                self.kin.fk(self.world.data)
                if self.config.robot_name == "so101"
                else None
            ),
        )

    def prepare_action(self, action: Action) -> Action:
        if self.config.robot_name == "so101" and action.mode == "twist":
            action = self._resolver(
                action.ee_twist, self._joint_state(), action.gripper
            )
        self.robot_spec.validate_action(action)
        return action

    def submit(self, action: Action) -> None:
        action = self.prepare_action(action)
        if self.config.robot_name == "so101":
            for name, value in zip(ARM_JOINT_NAMES, action.joint_position):
                self.world.set_controls(self.config.instance_id, {name: float(value)})
            gripper = action.gripper or GripperCommand()
            lo, hi = self._gripper_limits
            self.world.set_controls(
                self.config.instance_id,
                {"gripper": float(lo + gripper.clipped() * (hi - lo))},
            )
        else:
            twist = action.ee_twist
            self.world.set_controls(
                self.config.instance_id,
                {"forward": float(twist.linear.x), "turn": float(twist.angular.z)},
            )
        self._last_action = action

    def hold(self) -> None:
        if self.config.robot_name == "so101":
            self.submit(self._last_action)
        else:
            self.world.set_controls(
                self.config.instance_id, {"forward": 0.0, "turn": 0.0}
            )


class SharedWorldHost:
    """One web host for a composite world and its per-instance controllers."""

    _CAMERA_PERIOD = 0.2

    def __init__(self, world: SharedWorld, instances: tuple[RobotInstanceConfig, ...]):
        self.world = world
        self.instances = {
            config.instance_id: SharedRobotInstance(world, config)
            for config in instances
        }
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._state: dict[str, Any] | None = None
        self._control_owner: dict[str, str] = {}
        self._control_deadline: dict[str, float] = {}
        self._pending: dict[str, Action] = {}
        self._camera_images: dict[str, np.ndarray] = {}
        self._camera_data = mujoco.MjData(self.model)
        self._camera_thread: threading.Thread | None = None
        self._camera_request = threading.Event()
        self._camera_ready = threading.Event()
        self._renderer: mujoco.Renderer | None = None

    @property
    def model(self):
        return self.world.model

    @property
    def data(self):
        return self.world.data

    @property
    def physics_lock(self):
        return self._physics_lock

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def control_hz(self) -> float:
        return float(self.world.control_hz)

    def scene(self) -> dict[str, Any]:
        return build_scene_manifest(
            self.model,
            robot="shared",
            instance_prefixes={
                instance_id: instance.prefix
                for instance_id, instance in self.instances.items()
            },
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_camera_thread()
        self._thread = threading.Thread(
            target=self._run, name="physai-shared-world", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            self.world.close()
            return
        self._thread.join(timeout=2.0)
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
        self.world.close()

    def camera_jpeg(self, instance_id: str, name: str) -> bytes:
        camera_key = f"{instance_id}:{name}"
        with self._lock:
            try:
                image = self._camera_images[camera_key].copy()
            except KeyError as exc:
                raise ValueError(
                    f"unknown camera {name!r} for robot instance {instance_id!r}"
                ) from exc
        buffer = BytesIO()
        iio.imwrite(buffer, image, extension=".jpg", quality=82)
        return buffer.getvalue()

    def reset(self) -> None:
        self._start_camera_thread()
        with self._physics_lock:
            self.world.reset()
            for instance in self.instances.values():
                instance.reset()
            import mujoco

            mujoco.mj_forward(self.model, self.data)
            self._pending.clear()
            self._publish()
        self._request_camera_capture(wait=True)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def release_control(self, source: str) -> None:
        with self._lock:
            for instance_id, owner in list(self._control_owner.items()):
                if owner == source:
                    self._control_owner.pop(instance_id, None)
                    self._control_deadline.pop(instance_id, None)
                    self._pending.pop(instance_id, None)

    def submit(
        self, instance_id: str, action: Action, *, source: str = "local"
    ) -> None:
        if instance_id not in self.instances:
            raise ValueError(f"unknown robot instance {instance_id!r}")
        with self._physics_lock:
            action = self.instances[instance_id].prepare_action(action)
        now = time.monotonic()
        with self._lock:
            owner = self._control_owner.get(instance_id)
            if (
                owner not in (None, source)
                and now < self._control_deadline[instance_id]
            ):
                raise PermissionError(
                    f"robot instance {instance_id!r} is controlled by another client"
                )
            self._control_owner[instance_id] = source
            self._control_deadline[instance_id] = now + 0.35
            self._pending[instance_id] = action

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    def _run(self) -> None:
        self.reset()
        period = 1.0 / self.control_hz
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._paused:
                with self._physics_lock:
                    for instance_id, instance in self.instances.items():
                        action = self._pending.pop(instance_id, None)
                        if action is not None:
                            instance.submit(action)
                        else:
                            instance.hold()
                    self.world.step()
                    self._publish()
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _publish(self) -> None:
        with self._lock:
            self._state = build_state_snapshot(
                self.model,
                self.data,
                step=self.world.step_count,
                robot="shared",
                instance_prefixes={
                    instance_id: instance.prefix
                    for instance_id, instance in self.instances.items()
                },
            )

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None:
            return
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="physai-shared-camera",
            daemon=True,
        )
        self._camera_thread.start()

    def _request_camera_capture(self, *, wait: bool = False) -> None:
        self._camera_ready.clear()
        self._camera_request.set()
        if wait:
            self._camera_ready.wait(timeout=2.0)

    def _camera_loop(self) -> None:
        renderer = mujoco.Renderer(self.model, height=240, width=320)
        next_capture = 0.0
        try:
            while not self._stop.is_set():
                timeout = max(0.0, next_capture - time.monotonic())
                requested = self._camera_request.wait(timeout=timeout)
                self._camera_request.clear()
                if self._stop.is_set():
                    break
                if not requested and time.monotonic() < next_capture:
                    continue
                with self._physics_lock:
                    mujoco.mj_copyData(self._camera_data, self.model, self.data)
                for instance_id, instance in self.instances.items():
                    for name, camera in instance.robot_spec.camera_frames.items():
                        renderer.update_scene(self._camera_data, camera=camera)
                        image = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        with self._lock:
                            self._camera_images[f"{instance_id}:{name}"] = image
                next_capture = time.monotonic() + self._CAMERA_PERIOD
                self._camera_ready.set()
        finally:
            renderer.close()
