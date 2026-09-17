"""Thread-isolated MuJoCo session used by the FastAPI gateway."""

from __future__ import annotations

import queue
import threading
import time
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import numpy as np

from ..contracts import Action, GripperCommand, Twist, Vector3
from ..control.resolver import TwistToJointResolver
from ..robots.base import RobotPort
from .telemetry import build_scene_manifest, build_state_snapshot


class SimulationSession:
    """Own one robot and keep browser I/O away from the physics thread."""

    def __init__(
        self, robot: RobotPort, *, robot_name: str, policy: Any = None
    ) -> None:
        self.robot = robot
        self.robot_name = robot_name
        self.policy = policy
        self._commands: queue.Queue[Action] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._observation = None
        self._gripper = GripperCommand()
        self._camera_frames: dict[str, bytes] = {}
        self._twist_resolver = None
        if hasattr(robot, "kin") and hasattr(robot, "data"):
            self._twist_resolver = TwistToJointResolver(
                robot.kin,
                robot.data,
                dt=float(getattr(getattr(robot, "cfg", None), "control_dt", 0.04)),
            )

    @property
    def model(self):
        return self.robot.model

    @property
    def data(self):
        return self.robot.data

    def scene(self) -> dict[str, Any]:
        return build_scene_manifest(self.model, robot=self.robot_name)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="physai-sim", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.robot.close()

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    def reset(self) -> None:
        with self._physics_lock:
            self._observation = self.robot.reset()
            self._gripper = GripperCommand()
            if self.policy is not None:
                self.policy.reset(self._observation)
            self._publish()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def submit(self, action: Action) -> None:
        if action.gripper is not None:
            self._gripper = action.gripper
        elif hasattr(self.robot, "joint_to_gripper") and self._observation is not None:
            gripper_joint = float(self._observation.joint_state.position[-1])
            self._gripper = GripperCommand(
                position=float(self.robot.joint_to_gripper(gripper_joint))
            )
        if action.mode == "twist" and "twist" not in self.robot.robot_spec.action_modes:
            if self._twist_resolver is None or self._observation is None:
                raise ValueError("twist jog is not available for this robot")
            action = self._twist_resolver(
                action.ee_twist,
                self._observation.joint_state,
                action.gripper or self._gripper,
            )
        self.robot.robot_spec.validate_action(action)
        try:
            self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(action)
        except queue.Full:
            pass

    def _run(self) -> None:
        self.reset()
        hold_action = self._hold_action()
        control_hz = float(
            getattr(getattr(self.robot, "cfg", None), "control_hz", 25.0)
        )
        period = 1.0 / control_hz
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._paused:
                action = self._latest_command()
                if (
                    action is None
                    and self.policy is not None
                    and self._observation is not None
                ):
                    action = self.policy.act(self._observation)
                if action is None:
                    action = hold_action
                if action is not None:
                    with self._physics_lock:
                        result = self.robot.step(action)
                        self._observation = result[0]
                        if self.policy is None:
                            hold_action = self._hold_action()
                        self._publish()
                        if self.policy is not None and (
                            self.policy.done or result[2] or result[3]
                        ):
                            self._observation = self.robot.reset()
                            self.policy.reset(self._observation)
                            self._publish()
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _hold_action(self) -> Action:
        if "twist" in self.robot.robot_spec.action_modes:
            return Action(ee_twist=Twist(), gripper=self._gripper)
        joint_count = len(self.robot.robot_spec.action_joint_names)
        return Action(
            joint_position=np.asarray(
                self._observation.joint_state.position[:joint_count]
            ),
            gripper=self._gripper,
        )

    def _latest_command(self) -> Action | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _publish(self) -> None:
        if self._observation is not None:
            for name, image in self._observation.images.items():
                buffer = BytesIO()
                iio.imwrite(buffer, image.data, extension=".jpg", quality=82)
                self._camera_frames[name] = buffer.getvalue()
        with self._lock:
            self._state = build_state_snapshot(
                self.model,
                self.data,
                step=self.robot.step_count,
                robot=self.robot_name,
            )

    def camera_jpeg(self, name: str) -> bytes:
        with self._physics_lock:
            try:
                return self._camera_frames[name]
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc


def action_from_payload(payload: dict[str, Any]) -> Action:
    """Convert the browser command schema to the shared ``Action`` contract."""
    mode = payload.get("mode")
    if mode == "joint_position":
        return Action(
            joint_position=payload["position"],
            joint_names=tuple(payload["names"]) if payload.get("names") else None,
            gripper=GripperCommand(float(payload.get("gripper", 1.0))),
        )
    if mode == "twist":
        linear = payload.get("linear", {})
        angular = payload.get("angular", {})
        gripper = payload.get("gripper")
        return Action(
            ee_twist=Twist(
                linear=Vector3(
                    float(linear.get("x", 0.0)),
                    float(linear.get("y", 0.0)),
                    float(linear.get("z", 0.0)),
                ),
                angular=Vector3(
                    float(angular.get("x", 0.0)),
                    float(angular.get("y", 0.0)),
                    float(angular.get("z", 0.0)),
                ),
            ),
            gripper=(GripperCommand(float(gripper)) if gripper is not None else None),
        )
    raise ValueError("command mode must be 'joint_position' or 'twist'")
