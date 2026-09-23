"""The one host class for the web/viewer layer.

A single-robot session and a multi-robot session are both `Host`
instances; a single robot is simply a session with one instance. Build
one with `Host.for_robot(...)` (direct-MuJoCo, one `RobotPort`, optional
policy) or `Host.for_world(...)` (a `SharedWorld` with N namespaced
instances, command/hold only, no policy). Every public method is
instance-keyed; a caller with one robot may omit `instance_id` and get
the sole instance. Reset and pause stay world-atomic, per the shared-world
contract, for both cases.
"""

from __future__ import annotations

import queue
import threading
import time
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from ..contracts import Action, GripperCommand, Header, ImageFrame, Twist
from ..control.resolver import TwistToJointResolver
from ..robots.base import RobotPort
from ..robots.registry import create_shared_instance
from ..sim.world import RobotInstanceConfig, SharedWorld
from .telemetry import build_scene_manifest, build_state_snapshot


class Host:
    """Own the authoritative simulation state; keep clients off the physics thread."""

    _CAMERA_PERIOD = 0.2

    def __init__(
        self,
        *,
        robot: RobotPort | None = None,
        robot_name: str | None = None,
        policy: Any = None,
        reset_seed: int | None = None,
        async_cameras: bool = False,
        world: SharedWorld | None = None,
        instances: tuple[RobotInstanceConfig, ...] | None = None,
    ) -> None:
        self._shared = world is not None
        if self._shared:
            self.world = world
            self.instances = {
                config.instance_id: create_shared_instance(
                    config.robot_name, world, config
                )
                for config in (instances or ())
            }
        else:
            self.robot = robot
            self.robot_name = robot_name
            self.policy = policy
            self.reset_seed = reset_seed
            self._async_cameras = async_cameras
            self._gripper = GripperCommand()
            self._twist_resolver = None
            if hasattr(robot, "kin") and hasattr(robot, "data"):
                self._twist_resolver = TwistToJointResolver(
                    robot.kin,
                    robot.data,
                    dt=float(getattr(getattr(robot, "cfg", None), "control_dt", 0.04)),
                )
            self.instances = {robot_name: robot}

        self._commands: dict[str, queue.Queue[Action]] = {
            instance_id: queue.Queue(maxsize=1) for instance_id in self.instances
        }
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._observation = None
        self._camera_images: dict[str, np.ndarray] = {}
        self._camera_data: Any = None
        self._camera_thread: threading.Thread | None = None
        self._camera_request = threading.Event()
        self._camera_ready = threading.Event()
        self._control_owner: dict[str, str] = {}
        self._control_deadline: dict[str, float] = {}
        self._control_timeout = 0.35

    @classmethod
    def for_robot(
        cls,
        robot: RobotPort,
        *,
        robot_name: str,
        policy: Any = None,
        reset_seed: int | None = None,
        async_cameras: bool = False,
    ) -> Host:
        """A single-robot session: direct MuJoCo, one `RobotPort`, optional policy."""
        return cls(
            robot=robot,
            robot_name=robot_name,
            policy=policy,
            reset_seed=reset_seed,
            async_cameras=async_cameras,
        )

    @classmethod
    def for_world(
        cls, world: SharedWorld, instances: tuple[RobotInstanceConfig, ...]
    ) -> Host:
        """A multi-robot session over a `SharedWorld`: command/hold only, no policy."""
        return cls(world=world, instances=instances)

    # -- identity -------------------------------------------------------
    @property
    def default_instance_id(self) -> str:
        return next(iter(self.instances))

    def _resolve(self, instance_id: str | None) -> str:
        if instance_id is None:
            return self.default_instance_id
        if instance_id not in self.instances:
            raise ValueError(f"unknown robot instance {instance_id!r}")
        return instance_id

    def _robot_spec(self, instance_id: str):
        instance = self.instances[instance_id]
        return instance.robot_spec if self._shared else instance.robot_spec

    @property
    def model(self):
        return self.world.model if self._shared else self.robot.model

    @property
    def data(self):
        return self.world.data if self._shared else self.robot.data

    @property
    def physics_lock(self) -> threading.Lock:
        """Serialize renderer clients with MuJoCo access in the physics loop."""
        return self._physics_lock

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def control_hz(self) -> float:
        if self._shared:
            return float(self.world.control_hz)
        return float(getattr(getattr(self.robot, "cfg", None), "control_hz", 30.0))

    # -- discovery --------------------------------------------------------
    def scene(self) -> dict[str, Any]:
        if self._shared:
            return build_scene_manifest(
                self.model,
                robot="shared",
                instance_prefixes={
                    instance_id: instance.prefix
                    for instance_id, instance in self.instances.items()
                },
            )
        return build_scene_manifest(self.model, robot=self.robot_name)

    def list_robots(self) -> list[dict[str, Any]]:
        result = []
        for instance_id in self.instances:
            spec = self._robot_spec(instance_id)
            cameras = list(spec.camera_frames)
            if not self._shared:
                # Duck-typed extension point: a policy may optionally publish
                # named debug/annotated frames (e.g. detection overlays)
                # through the same camera cache; see `_publish_debug_frames`.
                cameras += list(getattr(self.policy, "debug_camera_names", ()))
            prefix = getattr(self.instances[instance_id], "prefix", "")
            joint_names = [prefix + name for name in spec.joint_names]
            result.append(
                {
                    "name": instance_id,
                    "kind": spec.kind,
                    "robot": spec.name,
                    "action_modes": list(spec.action_modes),
                    "capabilities": list(spec.capabilities),
                    "cameras": cameras,
                    "joint_names": joint_names,
                    "joint_limits": {
                        prefix + name: self._joint_limit(spec, name, prefix + name)
                        for name in spec.joint_names
                    },
                }
            )
        return result

    def _joint_limit(self, spec, name: str, compiled_name: str) -> list[float]:
        limit = spec.joint_limits.get(name)
        if limit is not None:
            return [float(limit[0]), float(limit[1])]
        try:
            model = self.model
        except AttributeError:
            model = None
        joint_id = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, compiled_name)
            if model is not None
            else -1
        )
        if joint_id >= 0 and bool(model.jnt_limited[joint_id]):
            return [float(value) for value in model.jnt_range[joint_id]]
        return [-3.14159265, 3.14159265]

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_camera_thread()
        self._thread = threading.Thread(
            target=self._run, name="physai-host", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            self._close()
            return
        self._thread.join(timeout=2.0)
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
        if self._shared:
            self.world.close()

    def _close(self) -> None:
        if self._shared:
            self.world.close()
        else:
            self.robot.close()

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    # -- reset/pause/lease (world-atomic) ---------------------------------
    def _reset_episode(self):
        return self.robot.reset(seed=self.reset_seed)

    def reset(self) -> None:
        self._start_camera_thread()
        with self._physics_lock:
            self._discard_commands()
            if self._shared:
                self.world.reset()
                for instance in self.instances.values():
                    instance.reset()
                mujoco.mj_forward(self.model, self.data)
            else:
                self._observation = self._reset_episode()
                self._gripper = GripperCommand()
                if self.policy is not None:
                    self.policy.reset(self._observation)
            self._publish()
        self._request_camera_capture(wait=True)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def release_control(self, source: str) -> None:
        with self._lock:
            owned = [
                instance_id
                for instance_id, owner in self._control_owner.items()
                if owner == source
            ]
            for instance_id in owned:
                self._control_owner.pop(instance_id, None)
                self._control_deadline.pop(instance_id, None)
        for instance_id in owned:
            try:
                self._commands[instance_id].get_nowait()
            except queue.Empty:
                pass

    def _discard_commands(self) -> None:
        for command_queue in self._commands.values():
            try:
                command_queue.get_nowait()
            except queue.Empty:
                pass

    # -- commands -----------------------------------------------------------
    def submit(
        self, action: Action, *, instance_id: str | None = None, source: str = "local"
    ) -> None:
        instance_id = self._resolve(instance_id)
        if self._shared:
            # SharedRobotInstance.prepare_action() reads the world's live MuJoCo
            # state (kinematics/resolver), so it needs the same lock the physics
            # thread holds while stepping.
            with self._physics_lock:
                action = self._prepare_action(instance_id, action)
        else:
            action = self._prepare_action(instance_id, action)
        now = time.monotonic()
        with self._lock:
            owner = self._control_owner.get(instance_id)
            deadline = self._control_deadline.get(instance_id, 0.0)
            if owner not in (None, source) and now < deadline:
                raise PermissionError(
                    f"robot instance {instance_id!r} is controlled by another client"
                )
            self._control_owner[instance_id] = source
            self._control_deadline[instance_id] = now + self._control_timeout
        command_queue = self._commands[instance_id]
        try:
            command_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            command_queue.put_nowait(action)
        except queue.Full:
            pass

    def _prepare_action(self, instance_id: str, action: Action) -> Action:
        if self._shared:
            return self.instances[instance_id].prepare_action(action)
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
        return action

    def _latest_command(self, instance_id: str) -> Action | None:
        expired = False
        with self._lock:
            deadline = self._control_deadline.get(instance_id)
            if deadline is not None and time.monotonic() >= deadline:
                self._control_owner.pop(instance_id, None)
                self._control_deadline.pop(instance_id, None)
                expired = True
        if expired:
            try:
                self._commands[instance_id].get_nowait()
            except queue.Empty:
                pass
            return None
        try:
            return self._commands[instance_id].get_nowait()
        except queue.Empty:
            return None

    # -- physics loop -------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            # Camera rendering happens on this thread, so the renderer and its
            # OpenGL context belong to it. Freeing them from the thread that
            # called stop() is an access violation on Windows.
            self._close()

    def _loop(self) -> None:
        self.reset()
        period = 1.0 / self.control_hz
        hold_action = None if self._shared else self._hold_action()
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._paused:
                with self._physics_lock:
                    if self._shared:
                        self._tick_shared()
                    else:
                        hold_action = self._tick_single(hold_action)
                    self._publish()
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _tick_shared(self) -> None:
        for instance_id, instance in self.instances.items():
            action = self._latest_command(instance_id)
            if action is not None:
                instance.submit(action)
            else:
                instance.hold()
        self.world.step()

    def _tick_single(self, hold_action: Action) -> Action:
        action = self._latest_command(self.robot_name)
        if action is None and self.policy is not None and self._observation is not None:
            self._sync_observation_images()
            action = self.policy.act(self._observation)
            self._publish_debug_frames()
        if action is None:
            action = hold_action
        result = self.robot.step(action)
        self._observation = result[0]
        if self.policy is None:
            hold_action = self._hold_action()
        if self.policy is not None and (self.policy.done or result[2] or result[3]):
            self._observation = self._reset_episode()
            self.policy.reset(self._observation)
        return hold_action

    def _sync_observation_images(self) -> None:
        """Merge the async camera worker's latest frames into `_observation`.

        The env's own `observe()` never renders images itself in interactive
        (viewer/serve) mode — that used to happen inline on the physics
        thread every `camera_stride` ticks, stalling it for the length of a
        render. The async camera thread (`_camera_loop`) is the only
        renderer now, decoupled onto its own thread; this is what lets a
        vision-dependent policy (e.g. visual_servo) still see a reasonably
        fresh image without that render blocking physics stepping.
        """
        if self._observation is None:
            return
        prefix = f"{self.robot_name}:"
        with self._lock:
            cached = {
                name[len(prefix) :]: image
                for name, image in self._camera_images.items()
                if name.startswith(prefix)
            }
        if not cached:
            return
        data_obj = getattr(self.robot, "data", None)
        stamp = float(data_obj.time) if data_obj is not None else 0.0
        for name, data in cached.items():
            self._observation.images[name] = ImageFrame(
                data=data,
                camera_name=name,
                header=Header(stamp=stamp, frame_id=f"camera_{name}"),
            )

    def _publish_debug_frames(self) -> None:
        """Merge a policy's optional annotated frames into the camera cache.

        `Policy` (`policy/base.py`) is a frozen port, so this is a duck-typed
        convention rather than an abstract method: a concrete policy may
        define `debug_frames() -> dict[str, np.ndarray]` to publish extra
        named views (e.g. a detection overlay) through the same cache that
        backs every other camera, with no change to core contracts.
        """
        debug_frames = getattr(self.policy, "debug_frames", None)
        if debug_frames is None:
            return
        frames = debug_frames()
        if not frames:
            return
        with self._lock:
            for local_name, image in frames.items():
                self._camera_images[f"{self.robot_name}:{local_name}"] = np.asarray(
                    image, dtype=np.uint8
                )

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

    def _publish(self) -> None:
        with self._lock:
            if self._shared:
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
            else:
                if self._observation is not None:
                    for name, image in self._observation.images.items():
                        self._camera_images[f"{self.robot_name}:{name}"] = np.asarray(
                            image.data, dtype=np.uint8
                        ).copy()
                self._state = build_state_snapshot(
                    self.model,
                    self.data,
                    step=self.robot.step_count,
                    robot=self.robot_name,
                )

    # -- cameras --------------------------------------------------------------
    def _camera_specs(self) -> list[tuple[str, str, str]]:
        """(instance_id, local_name, mujoco_camera_name) for every camera.

        `RobotSpec.camera_frames` maps local name -> value, but the value
        means different things in each mode: a shared-world instance's value
        *is* the compiled (prefixed) MuJoCo camera name, while a standalone
        robot's value is its ROS2/TF frame id and the MuJoCo camera name is
        the local name itself (e.g. so101's `{"front": "camera_front"}` — the
        MuJoCo camera is named "front", not "camera_front").
        """
        specs = []
        for instance_id in self.instances:
            spec = self._robot_spec(instance_id)
            for local_name, value in spec.camera_frames.items():
                mujoco_name = value if self._shared else local_name
                specs.append((instance_id, local_name, mujoco_name))
        return specs

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None:
            return
        if not self._shared and not self._async_cameras:
            return
        camera_specs = self._camera_specs()
        if not camera_specs:
            return
        self._camera_data = mujoco.MjData(self.model)
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(camera_specs,),
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

    def _camera_loop(self, camera_specs: list[tuple[str, str, str]]) -> None:
        width = (
            int(getattr(self.robot, "_camera_width", 640)) if not self._shared else 320
        )
        height = (
            int(getattr(self.robot, "_camera_height", 480)) if not self._shared else 240
        )
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
                    mujoco.mj_copyData(self._camera_data, self.model, self.data)
                for instance_id, local_name, mujoco_name in camera_specs:
                    renderer.update_scene(self._camera_data, camera=mujoco_name)
                    image = np.asarray(renderer.render(), dtype=np.uint8).copy()
                    with self._lock:
                        self._camera_images[f"{instance_id}:{local_name}"] = image
                next_capture = time.monotonic() + self._CAMERA_PERIOD
                self._camera_ready.set()
        finally:
            renderer.close()

    def camera_jpeg(self, name: str, *, instance_id: str | None = None) -> bytes:
        instance_id = instance_id or self.default_instance_id
        with self._lock:
            try:
                image = self._camera_images[f"{instance_id}:{name}"].copy()
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc
        buffer = BytesIO()
        iio.imwrite(buffer, image, extension=".jpg", quality=82)
        return buffer.getvalue()

    def camera_image(self, name: str, *, instance_id: str | None = None) -> np.ndarray:
        instance_id = instance_id or self.default_instance_id
        with self._lock:
            try:
                return self._camera_images[f"{instance_id}:{name}"].copy()
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc

    def render_camera(self, name: str, *, instance_id: str | None = None) -> np.ndarray:
        """Return a cached frame, falling back to a direct render if needed."""
        try:
            return self.camera_image(name, instance_id=instance_id)
        except ValueError:
            pass
        if self._shared:
            raise ValueError(f"robot instance {instance_id!r} has no camera renderer")
        renderer = getattr(self.robot, "render_camera", None)
        if renderer is None:
            raise ValueError(f"robot {self.robot_name!r} has no camera renderer")
        return np.asarray(renderer(name))


__all__ = ["Host"]
