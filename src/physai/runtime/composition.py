"""Explicit runtime assembly for direct and ROS2-backed workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import Action, Observation
from ..control.safety import SafetyController
from ..policy.base import Policy
from ..policy.registry import create_policy
from ..robots.base import RobotPort
from ..robots.registry import create_robot, robot_kind, scene_defaults
from ..sim.scenes import create_scene, default_scene_for, get_scene_definition
from ..tasks import TaskRuntime, create_task
from ..tasks.base import Task


@dataclass
class RuntimeComposition:
    """Compose independent runtime ports without hiding their dependencies."""

    robot: RobotPort
    task: Task | None = None
    policy: Policy | None = None
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
    task_success_hold_steps: int | None = None,
    adapter: str = "direct_mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    policy_name: str | None = None,
    policy: Policy | None = None,
    safety: SafetyController | None = None,
    **policy_kwargs: Any,
) -> RuntimeComposition:
    """Build and validate a robot-task-policy composition.

    The registered robot kind is resolved before construction so scene
    defaults and scene compatibility can be selected without branching on a
    concrete robot name. The constructed robot is then checked again through
    its ``RobotSpec`` before task, policy, and safety components are composed.

    ``robot_config`` and ``robot_kwargs`` are kept separate so callers can
    pass either an existing typed config or factory fields, but not silently
    merge both. Task semantics are composed around the robot port after the
    robot is built.
    """
    if robot_config is not None and robot_kwargs:
        raise TypeError("pass either robot_config or robot_kwargs, not both")

    fields = dict(robot_kwargs or {})
    scene_config = None
    embodiment_kind = robot_kind(robot_name)
    if scene_name is not None:
        if embodiment_kind is None:
            raise ValueError(
                f"scene selection requires a registered kind for robot {robot_name!r}"
            )
        if robot_config is not None or "scene" in fields:
            raise TypeError(
                "pass either scene_name or an explicit robot scene config, not both"
            )
        scene_config = create_scene(
            scene_name,
            **scene_defaults(robot_name),
            **(scene_kwargs or {}),
        )
        fields["scene"] = scene_config

    selected_scene_name = scene_name
    if selected_scene_name is None and task_name is not None and embodiment_kind:
        selected_scene_name = default_scene_for(embodiment_kind, task_name)
        if (
            robot_config is None
            and "scene" not in fields
            and selected_scene_name is not None
        ):
            scene_config = create_scene(
                selected_scene_name,
                **scene_defaults(robot_name),
                **(scene_kwargs or {}),
            )
            fields["scene"] = scene_config
    robot = create_robot(
        robot_name,
        adapter=adapter,
        transport=transport,
        hardware=hardware,
        **({"config": robot_config} if robot_config is not None else fields),
    )

    try:
        task = (
            create_task(task_name, **(task_kwargs or {}))
            if task_name is not None
            else None
        )

        if task is not None:
            robot.robot_spec.validate_task(task)

        if selected_scene_name is not None:
            definition = get_scene_definition(selected_scene_name)
            if not definition.supports(
                robot.robot_spec.kind, task.name if task else None
            ):
                task_label = task.name if task else "no task"
                raise ValueError(
                    f"scene {selected_scene_name!r} is incompatible with "
                    f"robot kind {robot.robot_spec.kind!r} and task {task_label!r}"
                )

        if policy is not None and policy_name is not None:
            raise TypeError("pass either policy or policy_name, not both")
        runtime_robot = robot
        if task is not None:
            # None keeps TaskRuntime's own default.
            hold = (
                {}
                if task_success_hold_steps is None
                else {"success_hold_steps": task_success_hold_steps}
            )
            runtime_robot = TaskRuntime(robot, task, **hold)
        if policy is None and policy_name is not None:
            policy = create_policy(policy_name, env=runtime_robot, **policy_kwargs)

        # The adapter owns the gate so every path shares it; an explicit
        # controller replaces the default the adapter built for itself.
        if safety is not None:
            robot.safety = safety
        else:
            safety = getattr(robot, "safety", None)
        return RuntimeComposition(
            robot=runtime_robot,
            task=task,
            policy=policy,
            safety=safety,
            scene_name=selected_scene_name,
        )
    except Exception:
        robot.close()
        raise
