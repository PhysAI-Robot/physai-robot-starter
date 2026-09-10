"""Deterministic TurtleBot4 navigation primitives.

The controller uses the same base-frame convention as the MuJoCo model:
positive linear velocity drives along world ``-Y`` when yaw is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

import numpy as np

from ...contracts import Action, Twist, Vector3
from .env import TurtleBot4Config, TurtleBot4Env


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class NavigationGoal:
    """A world-frame TurtleBot goal pose in metres and radians."""

    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class NavigationResult:
    """Outcome and metrics from one deterministic navigation episode."""

    reached: bool
    steps: int
    position_error: float
    heading_error: float
    collision_count: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class Nav2AcceptanceResult:
    """Structured result for the live ROS2/Nav2 acceptance path."""

    action_status: int | None
    goal_accepted: bool
    timed_out: bool
    position_error: float | None
    collision_count: int | None
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.action_status == 4
            and self.goal_accepted
            and not self.timed_out
            and self.position_error is not None
            and self.collision_count == 0
            and self.failure_reason is None
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "action_status": self.action_status,
            "goal_accepted": self.goal_accepted,
            "timed_out": self.timed_out,
            "position_error": self.position_error,
            "collision_count": self.collision_count,
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }


@dataclass(frozen=True)
class RPPConfig:
    """Small regulated-pure-pursuit parameter set shared with Nav2 defaults."""

    max_linear_speed: float = 0.35
    max_angular_speed: float = 1.0
    heading_gain: float = 2.5
    position_tolerance: float = 0.10
    yaw_tolerance: float = 0.15


class RegulatedPurePursuit:
    """Compute a bounded Twist toward a world-frame goal pose."""

    def __init__(self, config: RPPConfig | None = None) -> None:
        self.config = config or RPPConfig()

    def command(self, pose: np.ndarray, goal: NavigationGoal) -> Twist:
        x, y, yaw = (float(value) for value in pose)
        dx = goal.x - x
        dy = goal.y - y
        distance = hypot(dx, dy)
        if distance <= self.config.position_tolerance:
            heading_error = _wrap_angle(goal.yaw - yaw)
            angular = np.clip(
                self.config.heading_gain * heading_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
            return Twist(
                linear=Vector3(),
                angular=Vector3(z=float(angular)),
                frame_id="base",
            )
        desired_heading = atan2(dx, -dy)
        heading_error = _wrap_angle(desired_heading - yaw)
        linear = min(self.config.max_linear_speed, distance)
        linear *= max(0.0, cos(heading_error))
        angular = np.clip(
            self.config.heading_gain * heading_error,
            -self.config.max_angular_speed,
            self.config.max_angular_speed,
        )
        return Twist(
            linear=Vector3(x=float(linear)),
            angular=Vector3(z=float(angular)),
            frame_id="base",
        )

    def reached(self, pose: np.ndarray, goal: NavigationGoal) -> bool:
        position_error = hypot(float(pose[0]) - goal.x, float(pose[1]) - goal.y)
        heading_error = abs(_wrap_angle(float(pose[2]) - goal.yaw))
        return (
            position_error <= self.config.position_tolerance
            and heading_error <= self.config.yaw_tolerance
        )


def navigate_to_goal(
    goal: NavigationGoal,
    *,
    config: TurtleBot4Config | None = None,
    controller: RegulatedPurePursuit | None = None,
    seed: int = 0,
    max_steps: int = 300,
) -> NavigationResult:
    """Run the deterministic Point A to Point B baseline in MuJoCo."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    pursuit = controller or RegulatedPurePursuit()
    env = TurtleBot4Env(config or TurtleBot4Config(render=False, max_steps=max_steps))
    try:
        observation = env.reset(seed=seed)
        collision_count = 0
        reached = False
        for step in range(1, max_steps + 1):
            pose = env._pose_array()
            if pursuit.reached(pose, goal):
                reached = True
                break
            observation, _, _, _, _ = env.step(
                Action(ee_twist=pursuit.command(pose, goal))
            )
            collision_count += env.non_ground_contact_count()
        pose = env._pose_array()
        position_error = hypot(float(pose[0]) - goal.x, float(pose[1]) - goal.y)
        heading_error = abs(_wrap_angle(float(pose[2]) - goal.yaw))
        reached = reached or pursuit.reached(pose, goal)
        return NavigationResult(
            reached=reached,
            steps=step,
            position_error=position_error,
            heading_error=heading_error,
            collision_count=collision_count,
            failure_reason=None if reached else "goal_timeout",
        )
    finally:
        env.close()


def navigate_to_coordinates(
    *,
    goal_x: float,
    goal_y: float,
    goal_yaw: float = 0.0,
    **kwargs: object,
) -> NavigationResult:
    """Adapt generic navigation coordinates to the TurtleBot goal contract."""
    return navigate_to_goal(
        NavigationGoal(goal_x, goal_y, goal_yaw),
        **kwargs,
    )


__all__ = [
    "NavigationGoal",
    "NavigationResult",
    "Nav2AcceptanceResult",
    "RPPConfig",
    "RegulatedPurePursuit",
    "navigate_to_goal",
    "navigate_to_coordinates",
]