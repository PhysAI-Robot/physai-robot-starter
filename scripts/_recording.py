"""`--record` / `--record-dir` for headless runs (`run_sim.py`, `eval_policy.py`).

Dataset mode (`--record-dir`) feeds one `EpisodeRecorder` in the layout
training, replay and the web viewer read. Flat mode (`--record`) writes one
`<name>_seed<seed>.npz` + `.json` per episode, next to each other like videos.
"""

from __future__ import annotations

from pathlib import Path

from _common_args import DEFAULT_RECORD_DIR
from _video import next_video_stem

from physai.data import EpisodeRecorder
from physai.data.extras import collect_extras


class RunRecorder:
    def __init__(
        self,
        robot,
        *,
        fps: float,
        name: str,
        task: str,
        record_dir: Path | None = None,
    ) -> None:
        self._robot, self._fps, self._name, self._task = robot, fps, name, task
        self._record_dir = record_dir
        self._rec: EpisodeRecorder | None = None
        self._pending: tuple | None = None
        if record_dir is not None:
            self._rec = self._new_recorder(record_dir)
            self._load_existing()

    def _new_recorder(self, root: Path) -> EpisodeRecorder:
        robot = self._robot
        data = getattr(robot, "data", None)
        return EpisodeRecorder(
            root,
            task=self._task,
            fps=self._fps,
            robot_spec=robot.robot_spec,
            training_contract=getattr(robot, "training_contract", None),
            simulator_config={"control_hz": self._fps},
            environment_state_dim=None if data is None else int(data.qpos.size),
        )

    def _load_existing(self) -> None:
        import json

        meta = self._record_dir / "meta.json"
        if meta.exists():
            self._rec.episodes = json.loads(meta.read_text(encoding="utf-8"))[
                "episodes"
            ]

    def start(self) -> None:
        if self._record_dir is None:
            self._rec = self._new_recorder(DEFAULT_RECORD_DIR)
        self._rec.start_episode()

    def capture(self, policy, observation, info: dict | None) -> None:
        """Call before `env.step`: the state, grip force and policy metrics the
        step's observation and action belong to."""
        data = getattr(self._robot, "data", None)
        self._pending = (
            None if data is None else data.qpos.copy(),
            collect_extras(self._robot, policy, observation, info),
            getattr(getattr(policy, "metrics", None), "phase", None)
            or getattr(getattr(policy, "phase", None), "name", ""),
        )

    def record(self, observation, action, reward: float, done: bool) -> None:
        """Call after `env.step` with the observation `capture` saw."""
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

    def end(self, success: bool, seed: int, **extra) -> Path | None:
        extra = {"seed": seed, **extra}
        if self._record_dir is not None:
            return self._rec.end_episode(success, extra=extra)
        stem = next_video_stem(DEFAULT_RECORD_DIR, self._name, seed, (".npz",))
        path = self._rec.end_episode(
            success, extra=extra, path=stem.with_suffix(".npz")
        )
        if path is not None:
            self._rec.write_meta(stem.with_suffix(".json"))
        return path

    def close(self) -> None:
        if self._record_dir is not None:
            self._rec.write_meta()
