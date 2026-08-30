"""Explicit runtime assembly for P0 workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..control.safety import SafetyController
from ..contracts import Action, Observation
from ..planner.base import Planner
from ..policy.base import Policy
from ..policy.registry import create_policy
from ..robots.base import RobotPort
from ..robots.registry import create_robot
from ..sim.scenes import create_scene, get_scene_definition
from ..tasks import TaskRuntime, create_task
from ..tasks.base import Task


@dataclass
class RuntimeComposition:
    """Compose independent runtime ports without hiding their dependencies."""

    robot: RobotPort
    task: Task | None = None
    policy: Policy | None = None
    planner: Planner | None = None
    safety: SafetyController | None = None
    scene_name: str | None = None
    _observation: Observation | None = None

    @property
    def robot_spec(self):
        return self.robot.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        self._observation = self.robot.reset(seed=seed)
        if self.policy is not None:
            self.policy.reset(self._observation)
        return self._observation

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        if self.safety is not None:
            action = self.safety.validate(self._observation, action)
        result = self.robot.step(action)
        self._observation = result[0]
        return result

    def close(self) -> None:
        self.robot.close()


def create_runtime(
    robot_name: str,
    *,
    task_name: str | None = None,
    robot_config: Any = None,
    robot_kwargs: dict[str, Any] | None = None,
    scene_name: str | None = None,
    scene_kwargs: dict[str, Any] | None = None,
    task_kwargs: dict[str, Any] | None = None,
    task_success_hold_steps: int = 10,
    adapter: str = "direct_mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    policy_name: str | None = None,
    policy: Policy | None = None,
    planner: Planner | None = None,
    safety: SafetyController | None = None,
    **policy_kwargs: Any,
) -> RuntimeComposition:
    """Build and validate a robot-task-policy composition.

    ``robot_config`` and ``robot_kwargs`` are kept separate so callers can
    pass either an existing typed config or factory fields, but not silently
    merge both. Task semantics are composed around the robot port after the
    robot is built.
    """
    if robot_config is not None and robot_kwargs:
        raise TypeError("pass either robot_config or robot_kwargs, not both")

    fields = dict(robot_kwargs or {})
    scene_config = None
    if scene_name is not None:
        if robot_name != "so101":
            raise ValueError(
                f"scene selection is not supported by robot {robot_name!r} yet"
            )
        if robot_config is not None or "scene" in fields:
            raise TypeError(
                "pass either scene_name or an explicit robot scene config, not both"
            )
        scene_config = create_scene(scene_name, **(scene_kwargs or {}))
        fields["scene"] = scene_config

    default_scenes = {
        "pick_place": "pick_place_minimal",
        "sorting": "sorting_minimal",
    }
    selected_scene_name = scene_name
    if selected_scene_name is None and task_name in default_scenes and robot_name == "so101":
        selected_scene_name = default_scenes[task_name]
        if robot_config is None and "scene" not in fields:
            scene_config = create_scene(selected_scene_name, **(scene_kwargs or {}))
            fields["scene"] = scene_config

    if task_name is not None and robot_name == "so101" and robot_config is None:
        fields.setdefault("task", task_name)
    robot = create_robot(
        robot_name,
        adapter=adapter,
        transport=transport,
        hardware=hardware,
        **({"config": robot_config} if robot_config is not None else fields),
    )

    try:
        task = create_task(task_name, **(task_kwargs or {})) if task_name is not None else None

        if task is not None:
            robot.robot_spec.validate_task(task)

        if selected_scene_name is not None:
            definition = get_scene_definition(selected_scene_name)
            if not definition.supports(robot.robot_spec.kind, task.name if task else None):
                task_label = task.name if task else "no task"
                raise ValueError(
                    f"scene {selected_scene_name!r} is incompatible with "
                    f"robot kind {robot.robot_spec.kind!r} and task {task_label!r}"
                )

        if policy is not None and policy_name is not None:
            raise TypeError("pass either policy or policy_name, not both")
        runtime_robot = robot
        if task is not None:
            runtime_robot = TaskRuntime(
                robot,
                task,
                success_hold_steps=task_success_hold_steps,
            )
        if policy is None and policy_name is not None:
            policy = create_policy(policy_name, env=runtime_robot, **policy_kwargs)

        safety = safety or SafetyController(robot.robot_spec)
        return RuntimeComposition(
            robot=runtime_robot,
            task=task,
            policy=policy,
            planner=planner,
            safety=safety,
            scene_name=selected_scene_name,
        )
    except Exception:
        robot.close()
        raise
