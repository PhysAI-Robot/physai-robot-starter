"""Thread-isolated MuJoCo session used by the FastAPI gateway."""

from __future__ import annotations

import queue
import threading
import time
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from ..contracts import Action, GripperCommand, Twist, Vector3
from ..control.resolver import TwistToJointResolver
from ..robots.base import RobotPort
from .telemetry import build_scene_manifest, build_state_snapshot


class SimulationHost:
    """Own the authoritative robot state and keep clients off the physics thread."""

    _CAMERA_PERIOD = 0.2

    def __init__(
        self,
        robot: RobotPort,
        *,
        robot_name: str,
        policy: Any = None,
        reset_seed: int | None = None,
        async_cameras: bool = False,
    ) -> None:
        self.robot = robot
        self.robot_name = robot_name
        self.policy = policy
        self.reset_seed = reset_seed
        self._async_cameras = async_cameras
        self._commands: queue.Queue[Action] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._observation = None
        self._gripper = GripperCommand()
        self._camera_images: dict[str, np.ndarray] = {}
        self._camera_data: Any = None
        self._camera_thread: threading.Thread | None = None
        self._camera_request = threading.Event()
        self._camera_ready = threading.Event()
        self._control_owner: str | None = None
        self._control_deadline = 0.0
        self._control_timeout = 0.35
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

    @property
    def physics_lock(self) -> threading.Lock:
        """Serialize renderer clients with MuJoCo access in the physics loop."""
        return self._physics_lock

    def scene(self) -> dict[str, Any]:
        return build_scene_manifest(self.model, robot=self.robot_name)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_camera_thread()
        self._thread = threading.Thread(
            target=self._run, name="physai-sim", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            self.robot.close()
            return
        self._thread.join(timeout=2.0)
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    def _reset_episode(self):
        return self.robot.reset(seed=self.reset_seed)

    def reset(self) -> None:
        self._start_camera_thread()
        with self._physics_lock:
            self._discard_command()
            self._observation = self._reset_episode()
            self._gripper = GripperCommand()
            if self.policy is not None:
                self.policy.reset(self._observation)
            self._publish()
        self._request_camera_capture(wait=True)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    @property
    def paused(self) -> bool:
        return self._paused

    def release_control(self, source: str) -> None:
        with self._lock:
            if self._control_owner == source:
                self._control_owner = None
                self._control_deadline = 0.0
        self._discard_command()

    def _discard_command(self) -> None:
        try:
            self._commands.get_nowait()
        except queue.Empty:
            pass

    def submit(self, action: Action, *, source: str = "local") -> None:
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
        now = time.monotonic()
        with self._lock:
            if (
                self._control_owner not in (None, source)
                and now < self._control_deadline
            ):
                raise PermissionError("simulation control is held by another client")
            self._control_owner = source
            self._control_deadline = now + self._control_timeout
        try:
            self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(action)
        except queue.Full:
            pass

    def _run(self) -> None:
        try:
            self._loop()
        finally:
            # Camera rendering happens on this thread, so the renderer and its
            # OpenGL context belong to it. Freeing them from the thread that
            # called stop() is an access violation on Windows.
            self.robot.close()

    def _loop(self) -> None:
        self.reset()
        hold_action = self._hold_action()
        control_hz = float(
            getattr(getattr(self.robot, "cfg", None), "control_hz", 30.0)
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
                            self._observation = self._reset_episode()
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
        expired = False
        with self._lock:
            if (
                self._control_owner is not None
                and time.monotonic() >= self._control_deadline
            ):
                self._control_owner = None
                self._control_deadline = 0.0
                expired = True
        if expired:
            self._discard_command()
            return None
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _publish(self) -> None:
        with self._lock:
            if self._observation is not None:
                for name, image in self._observation.images.items():
                    self._camera_images[name] = np.asarray(
                        image.data, dtype=np.uint8
                    ).copy()
            self._state = build_state_snapshot(
                self.model,
                self.data,
                step=self.robot.step_count,
                robot=self.robot_name,
            )

    def _start_camera_thread(self) -> None:
        if not self._async_cameras or self._camera_thread is not None:
            return
        camera_names = tuple(self.robot.robot_spec.camera_frames)
        if not camera_names:
            return
        mj_data_type = getattr(mujoco, "MjData")  # noqa: B009
        self._camera_data = mj_data_type(self.model)
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_names,),
            name="physai-camera",
            daemon=True,
        )
        self._camera_thread.start()

    def _request_camera_capture(self, *, wait: bool = False) -> None:
        if self._camera_thread is None:
            return
        self._camera_ready.clear()
        self._camera_request.set()
        if wait:
            self._camera_ready.wait(timeout=2.0)

    def _camera_loop(self, camera_names: tuple[str, ...]) -> None:
        width = int(getattr(self.robot, "_camera_width", 640))
        height = int(getattr(self.robot, "_camera_height", 480))
        renderer = mujoco.Renderer(self.model, height=height, width=width)
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
                    copy_data = getattr(mujoco, "mj_copyData")  # noqa: B009
                    copy_data(self._camera_data, self.model, self.data)
                for name in camera_names:
                    renderer.update_scene(self._camera_data, camera=name)
                    image = np.asarray(renderer.render(), dtype=np.uint8).copy()
                    with self._lock:
                        self._camera_images[name] = image
                next_capture = time.monotonic() + self._CAMERA_PERIOD
                self._camera_ready.set()
        finally:
            renderer.close()

    def camera_jpeg(self, name: str) -> bytes:
        with self._lock:
            try:
                image = self._camera_images[name].copy()
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc
        buffer = BytesIO()
        iio.imwrite(buffer, image, extension=".jpg", quality=82)
        return buffer.getvalue()

    def camera_image(self, name: str) -> np.ndarray:
        with self._lock:
            try:
                return self._camera_images[name].copy()
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc

    def render_camera(self, name: str) -> np.ndarray:
        """Return a cached frame, falling back to a direct render if needed."""
        try:
            return self.camera_image(name)
        except ValueError:
            pass
        renderer = getattr(self.robot, "render_camera", None)
        if renderer is None:
            raise ValueError(f"robot {self.robot_name!r} has no camera renderer")
        return np.asarray(renderer(name))


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
