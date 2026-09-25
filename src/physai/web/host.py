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

import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from ..contracts import Action, GripperCommand, Header, ImageFrame, Twist
from ..control.resolver import TwistToJointResolver
from ..robots.base import RobotPort
from ..robots.registry import create_shared_instance
from ..sim.world import RobotInstanceConfig, SharedWorld
from .cameras import CameraFeed
from .lease import ControlLease
from .playback import Playback
from .recording import SessionRecorder
from .telemetry import build_scene_manifest, build_state_snapshot


class Host:
    """Own the authoritative simulation state; keep clients off the physics thread."""

    # Longest wall-clock gap one playback tick may advance by (see
    # `_advance_playback`).
    _MAX_PLAYBACK_STEP = 0.25

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
        record_dir: Path | str | None = None,
    ) -> None:
        self._shared = world is not None
        self._recorder: SessionRecorder | None = None
        self._playback: Playback | None = None
        self._playback_status: dict[str, Any] = {"active": False}
        self._playback_clock: float | None = None
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
            if hasattr(robot, "resolve_twist_jog"):
                self._twist_resolver = robot.resolve_twist_jog
            elif hasattr(robot, "kin") and hasattr(robot, "data"):
                self._twist_resolver = TwistToJointResolver(
                    robot.kin,
                    robot.data,
                    dt=float(getattr(getattr(robot, "cfg", None), "control_dt", 0.04)),
                )
            self.instances = {robot_name: robot}
            if record_dir is not None:
                data = getattr(robot, "data", None)
                self._recorder = SessionRecorder(
                    record_dir,
                    robot=robot,
                    fps=self.control_hz,
                    environment_state_dim=None if data is None else int(data.qpos.size),
                )

        self._lease = ControlLease(self.instances)
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._state: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._observation = None
        self._hold_action_value: Action | None = None
        self._cameras = CameraFeed()

    @classmethod
    def for_robot(
        cls,
        robot: RobotPort,
        *,
        robot_name: str,
        policy: Any = None,
        reset_seed: int | None = None,
        async_cameras: bool = False,
        record_dir: Path | str | None = None,
    ) -> Host:
        """A single-robot session: direct MuJoCo, one `RobotPort`, optional policy.

        `record_dir` enables browser-driven episode recording into that
        dataset directory (see `start_recording`).
        """
        return cls(
            robot=robot,
            robot_name=robot_name,
            policy=policy,
            reset_seed=reset_seed,
            async_cameras=async_cameras,
            record_dir=record_dir,
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
        return instance.robot_spec

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
        self._cameras.join()
        if self._shared:
            self.world.close()

    def _close(self) -> None:
        if self._shared:
            self.world.close()
        else:
            self.robot.close()

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            if self._state is None:
                return None
            state = dict(self._state)
        # Merged at read time, not published by the tick loop, so a paused
        # world (no ticks) still reports pause/recording changes immediately.
        state["paused"] = self._paused
        state["recording"] = self.recording_status()
        state["playback"] = self._playback_status
        return state

    # -- reset/pause/lease (world-atomic) ---------------------------------
    def _reset_episode(self):
        return self.robot.reset(seed=self.reset_seed)

    def reset(self) -> None:
        self._start_camera_thread()
        with self._physics_lock:
            self._require_not_playing_back("reset the world")
            self._discard_commands()
            self._abort_recording("world reset")
            if self._shared:
                self.world.reset()
                for instance in self.instances.values():
                    instance.reset()
                mujoco.mj_forward(self.model, self.data)
            else:
                self._observation = self._reset_episode()
                self._gripper = GripperCommand()
                # The tick loop's hold target otherwise keeps whatever the
                # robot was last told to do before this reset (e.g. mid-jog),
                # which can be far from the freshly-reset pose -- resubmitting
                # it on the next idle tick then fails the safety gate's
                # max-step check against the new observation.
                self._hold_action_value = self._hold_action()
                if self.policy is not None:
                    self.policy.reset(self._observation)
            self._publish()
        self._request_camera_capture(wait=True)

    def set_paused(self, paused: bool) -> None:
        if not paused:
            self._require_not_playing_back("resume the simulation")
        self._paused = paused

    # -- recording ---------------------------------------------------------
    def recording_status(self) -> dict[str, Any]:
        if self._recorder is None:
            return {"enabled": False}
        return self._recorder.status()

    def start_recording(self) -> None:
        """Begin an episode; frames are captured on every unpaused tick."""
        recorder = self._require_recorder()
        self._require_not_playing_back("record")
        recorder.start()

    def stop_recording(self, success: bool | None) -> None:
        """End the episode as a success/fail take, or discard it with `None`."""
        self._require_recorder().stop(success)

    def _require_recorder(self) -> SessionRecorder:
        if self._shared:
            raise ValueError("recording is not available for shared-world hosts")
        if self._recorder is None:
            raise ValueError("recording is disabled; start the host with --record-dir")
        return self._recorder

    def _abort_recording(self, reason: str) -> None:
        if self._recorder is not None:
            self._recorder.abort(reason)

    def _record_tick(self, action: Action) -> None:
        """Capture the observation the action was chosen from, with its action."""
        if (
            self._recorder is None
            or not self._recorder.active
            or self._observation is None
        ):
            return
        self._sync_observation_images()
        data = getattr(self.robot, "data", None)
        self._recorder.record_tick(
            self._observation, action, None if data is None else data.qpos
        )

    # -- playback ------------------------------------------------------------
    def list_episodes(self) -> list[dict[str, Any]]:
        """Saved episodes of the record directory, with success tags."""
        if self._recorder is None:
            return []
        return self._recorder.episodes()

    def load_episode(self, file: str) -> None:
        """Pause the world and show frame 0 of a saved episode.

        Playback restores recorded simulator state on the paused world; it
        never re-simulates. Loading another episode while one is open keeps
        the snapshot of the original live world.
        """
        recorder = self._require_recorder()
        if recorder.active:
            raise ValueError("stop recording before loading an episode")
        states, entry = recorder.load_states(file)
        if states.ndim != 2 or states.shape[1] != self.data.qpos.size:
            raise ValueError(
                f"episode {file!r} was recorded for a different model "
                f"({states.shape[-1]} vs {self.data.qpos.size} qpos values)"
            )
        with self._physics_lock:
            previous = self._playback
            if previous is None:
                live_data = mujoco.MjData(self.model)
                mujoco.mj_copyData(live_data, self.model, self.data)
                live = (live_data, self._observation, self._hold_action_value)
            else:
                live = (
                    previous.live_data,
                    previous.live_observation,
                    previous.live_hold_action,
                )
            self._paused = True
            self._discard_commands()
            self._playback = Playback(
                file=file,
                states=states,
                fps=float(entry.get("fps") or self.control_hz),
                success=entry.get("success"),
                live_data=live[0],
                live_observation=live[1],
                live_hold_action=live[2],
            )
            self._show_playback_frame()

    def seek(self, frame: int, *, relative: bool = False) -> None:
        """Jump to a frame, or step by `frame` frames when `relative`.

        Stepping stops playback (frame-by-frame inspection); scrubbing to an
        absolute frame leaves it running.
        """
        with self._physics_lock:
            playback = self._require_playback()
            if relative:
                playback.playing = False
                playback.seek(playback.frame + int(frame))
            else:
                playback.seek(int(frame))
            self._show_playback_frame()

    def set_playback(self, playing: bool, speed: float | None = None) -> None:
        with self._physics_lock:
            playback = self._require_playback()
            if speed is not None:
                playback.set_speed(float(speed))
            if playing and playback.position >= playback.length - 1:
                playback.seek(0)  # replay from the start after reaching the end
            playback.playing = bool(playing)
            self._playback_clock = None  # restart timing from the next tick
            self._show_playback_frame()

    def exit_playback(self) -> None:
        """Restore the live world that playback interrupted; stay paused."""
        with self._physics_lock:
            playback = self._require_playback()
            mujoco.mj_copyData(self.data, self.model, playback.live_data)
            mujoco.mj_forward(self.model, self.data)
            self._observation = playback.live_observation
            self._hold_action_value = playback.live_hold_action
            self._playback = None
            self._playback_status = {"active": False}
            self._discard_commands()
            self._publish()

    def _require_playback(self) -> Playback:
        if self._playback is None:
            raise ValueError("no episode is loaded for playback")
        return self._playback

    def _require_not_playing_back(self, action: str) -> None:
        if self._playback is not None:
            raise ValueError(f"exit playback before you {action}")

    def _show_playback_frame(self) -> None:
        """Write the current frame's recorded state into the paused world."""
        playback = self._playback
        if playback.frame != playback.shown:
            self.data.qpos[:] = playback.states[playback.frame]
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            playback.shown = playback.frame
            self._playback_status = playback.status()
            self._publish()
        else:
            self._playback_status = playback.status()

    def _advance_playback(self, now: float) -> None:
        """One paused-loop tick of an ongoing playback (no separate engine).

        Advances by measured wall time, so 1x is real time even when the loop
        runs slower than `control_hz`; a stall is capped so it cannot skip
        far ahead.
        """
        with self._physics_lock:
            playback = self._playback
            if playback is None or not playback.playing:
                self._playback_clock = None
                return
            elapsed = (
                0.0 if self._playback_clock is None else now - self._playback_clock
            )
            self._playback_clock = now
            elapsed = min(max(elapsed, 0.0), self._MAX_PLAYBACK_STEP)
            playback.advance(playback.speed * playback.fps * elapsed)
            self._show_playback_frame()

    def release_control(self, source: str) -> None:
        self._lease.release(source)

    def _discard_commands(self) -> None:
        self._lease.discard_all()

    # -- commands -----------------------------------------------------------
    def submit(
        self, action: Action, *, instance_id: str | None = None, source: str = "local"
    ) -> None:
        self._require_not_playing_back("send commands")
        instance_id = self._resolve(instance_id)
        if self._shared:
            # SharedRobotInstance.prepare_action() reads the world's live MuJoCo
            # state (kinematics/resolver), so it needs the same lock the physics
            # thread holds while stepping.
            with self._physics_lock:
                action = self._prepare_action(instance_id, action)
        else:
            action = self._prepare_action(instance_id, action)
        self._lease.submit(instance_id, action, source)

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
        return self._lease.latest(instance_id)

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
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._paused:
                with self._physics_lock:
                    if self._shared:
                        self._tick_shared()
                    else:
                        self._tick_single()
                    self._publish()
            elif self._playback is not None:
                self._advance_playback(started)
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _tick_shared(self) -> None:
        for instance_id, instance in self.instances.items():
            action = self._latest_command(instance_id)
            if action is not None:
                instance.submit(action)
            else:
                instance.hold()
        self.world.step()

    def _tick_single(self) -> None:
        action = self._latest_command(self.robot_name)
        if action is None and self.policy is not None and self._observation is not None:
            self._sync_observation_images()
            action = self.policy.act(self._observation)
            self._publish_debug_frames()
        if action is None:
            action = self._hold_action_value
        elif self.policy is None and action.mode == "joint_position":
            # Anchor future idle ticks to the target this command actually
            # resolved to, not a live re-read of the observation after
            # stepping: re-deriving the hold target from the observation
            # every tick re-affirms that tick's small gravity/servo sag as
            # the new floor, so an idle arm never actually settles -- it
            # ratchets into a slow, continuous fall instead.
            self._hold_action_value = action
        self._record_tick(action)
        result = self.robot.step(action)
        self._observation = result[0]
        if self.policy is not None and (self.policy.done or result[2] or result[3]):
            self._abort_recording("episode ended by the policy")
            self._observation = self._reset_episode()
            # Same reason as Host.reset(): resync the hold target to the new
            # episode's pose so a later idle tick doesn't resubmit wherever
            # the robot was when the previous episode ended.
            self._hold_action_value = self._hold_action()
            self.policy.reset(self._observation)

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
        cached = self._cameras.with_prefix(prefix)
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
        for local_name, image in frames.items():
            self._cameras.put(
                f"{self.robot_name}:{local_name}", np.asarray(image, dtype=np.uint8)
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
                if self._observation is not None and self._playback is None:
                    for name, image in self._observation.images.items():
                        self._cameras.put(
                            f"{self.robot_name}:{name}",
                            np.asarray(image.data, dtype=np.uint8).copy(),
                        )
                self._state = build_state_snapshot(
                    self.model,
                    self.data,
                    step=self.robot.step_count,
                    robot=self.robot_name,
                )
                self._state["ee_pose"] = self._ee_pose_payload()
                if self._playback is not None:
                    # Restored qpos cannot reproduce the actuator state that
                    # produced the recorded squeeze, so a force would be fiction.
                    for contact in self._state["gripper_contacts"]:
                        contact["force_n"] = None

    def _ee_pose_payload(self) -> dict[str, Any] | None:
        """The gripper tip pose for the HUD, or None if the robot has none.

        A robot whose kinematics offers `tool_pose(data)` (an optional
        extension outside the frozen `KinematicsPort`) supplies the pose it
        wants shown, e.g. the pinch centre where objects are held; otherwise
        the observation's `ee_pose` is passed through unchanged. Called from
        `_publish`, so `data` is consistent under `physics_lock`.
        """
        tool_pose = getattr(getattr(self.robot, "kin", None), "tool_pose", None)
        data = getattr(self.robot, "data", None)
        if tool_pose is not None and data is not None:
            return {**tool_pose(data).to_dict(), "reference": "tool"}
        if (
            self._playback is not None
            or self._observation is None
            or self._observation.ee_pose is None
        ):
            return None
        return {**self._observation.ee_pose.to_dict(), "reference": "ee_pose"}

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
        if self._cameras.running:
            return
        if not self._shared and not self._async_cameras:
            return
        camera_specs = self._camera_specs()
        if not camera_specs:
            return
        if self._shared:
            size = (320, 240)
        else:
            size = getattr(self.robot, "camera_size", None) or (640, 480)
        self._cameras.start(
            self.model,
            self.data,
            self._physics_lock,
            self._stop,
            [
                (f"{instance_id}:{local_name}", mujoco_name)
                for instance_id, local_name, mujoco_name in camera_specs
            ],
            size,
        )

    def _request_camera_capture(self, *, wait: bool = False) -> None:
        self._cameras.request(wait=wait)

    def camera_jpeg(self, name: str, *, instance_id: str | None = None) -> bytes:
        buffer = BytesIO()
        iio.imwrite(
            buffer,
            self.camera_image(name, instance_id=instance_id),
            extension=".jpg",
            quality=82,
        )
        return buffer.getvalue()

    def camera_image(self, name: str, *, instance_id: str | None = None) -> np.ndarray:
        instance_id = instance_id or self.default_instance_id
        image = self._cameras.get(f"{instance_id}:{name}")
        if image is None:
            raise ValueError(f"unknown camera {name!r}")
        return image

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
