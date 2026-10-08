"""`--record` for headless runs, and the dataset `collect_demos.py` writes.

Flat mode (`--record`) writes one `<name>_seed<seed>.npz` + `.json` per episode
under `<out>/recordings/`, next to each other like videos. Dataset mode (a
`dataset_dir`, used by `collect_demos.py`) feeds one `EpisodeRecorder` in the
layout training, replay and the web viewer read.
"""

from __future__ import annotations

from pathlib import Path

from _video import next_video_stem

from physai.contracts import parse_camera_resolution
from physai.data import EpisodeRecorder
from physai.data.extras import collect_extras
from physai.runtime import EpisodeObserver


class RunRecorder(EpisodeObserver):
    def __init__(
        self,
        runtime,
        *,
        fps: float,
        name: str,
        task: str,
        dataset_dir: Path | None = None,
        out_dir: Path = Path("outputs"),
        metadata: dict | None = None,
    ) -> None:
        """`metadata` is extra `EpisodeRecorder` arguments (scene, cameras, ...)."""
        self._runtime, self._robot = runtime, runtime.robot
        self._fps, self._name, self._task = fps, name, task
        self._dataset_dir = dataset_dir
        self._metadata = metadata or {}
        self._flat_dir = out_dir / "recordings"
        self._rec: EpisodeRecorder | None = None
        self._pending: tuple | None = None
        if dataset_dir is not None:
            self._rec = self._new_recorder(dataset_dir)

    def _new_recorder(self, root: Path) -> EpisodeRecorder:
        robot = self._robot
        data = getattr(robot, "data", None)
        return EpisodeRecorder(
            root,
            task=self._task,
            fps=self._fps,
            robot_spec=robot.robot_spec,
            training_contract=getattr(robot, "training_contract", None),
            **{"simulator_config": {"control_hz": self._fps}, **self._metadata},
            environment_state_dim=None if data is None else int(data.qpos.size),
        )

    def on_reset(self, observation) -> None:
        if self._dataset_dir is None:
            self._rec = self._new_recorder(self._flat_dir)
        self._rec.start_episode()

    def before_step(self, observation, action, info: dict) -> None:
        """The state, grip force and policy metrics this step's observation and
        action belong to (the world before the step runs)."""
        policy = self._runtime.policy
        data = getattr(self._robot, "data", None)
        self._pending = (
            None if data is None else data.qpos.copy(),
            collect_extras(self._robot, policy, observation, info),
            getattr(getattr(policy, "metrics", None), "phase", None)
            or getattr(getattr(policy, "phase", None), "name", ""),
        )

    def after_step(self, observation, action, next_observation, reward, done, info):
        state, extras, phase = self._pending
        gripper_to_joint = getattr(self._robot, "gripper_to_joint", None)
        grip = None
        if gripper_to_joint is not None and action.gripper is not None:
            grip = float(gripper_to_joint(action.gripper.clipped()))
        self._rec.record(
            observation,
            action,
            reward=reward,
            done=done,
            phase=phase,
            gripper_joint=grip,
            environment_state=state,
            extras=extras,
        )

    def set_task(self, task: str) -> None:
        """The instruction of the episode just run (it can depend on the seed)."""
        self._rec.task = task

    def discard(self) -> None:
        """Drop the episode just run without writing it (e.g. a failed demo)."""
        self._rec.discard_episode()

    def end(self, success: bool, seed: int, **extra) -> Path | None:
        extra = {"seed": seed, **extra}
        if self._dataset_dir is not None:
            return self._rec.end_episode(success, extra=extra)
        stem = next_video_stem(self._flat_dir, self._name, seed, (".npz",))
        path = self._rec.end_episode(
            success, extra=extra, path=stem.with_suffix(".npz")
        )
        if path is not None:
            self._rec.write_meta(stem.with_suffix(".json"))
        return path

    def close(self) -> None:
        if self._dataset_dir is not None:
            self._rec.write_meta()


def dataset_metadata(runtime, manifest, *, store_images: bool = True) -> dict:
    """The `EpisodeRecorder` arguments that describe the session a dataset was recorded in."""
    env = runtime.robot
    width, height = parse_camera_resolution(manifest.simulation.camera_resolution)
    return {
        "task_name": manifest.task_for(manifest.robots[0]),
        "store_images": store_images,
        "simulator_config": {
            "control_hz": env.cfg.control_hz,
            "max_steps": env.cfg.max_steps,
            "randomize_cube": env.cfg.randomize_cube,
            "randomize_target": env.cfg.randomize_target,
        },
        "camera_config": {
            name: {"width": width, "height": height, "encoding": "rgb8"}
            for name in env.cfg.cameras
        },
        "scene_name": manifest.scene.name,
        "scene_config": env.cfg.scene.to_metadata(),
    }
