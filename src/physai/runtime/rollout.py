"""One episode of a composed runtime: the loop `run_sim` and `eval_policy` share.

Owns how an episode runs (reset, act, step, when it ends, what a refused
action means). Callers own what to do with it: observers watch the steps
(video, recording), and each script turns the `EpisodeResult` into its own
output (a printed line, an `EvaluationReport` row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import Action, Observation
from ..control import SafetyViolation
from .composition import RuntimeComposition


class RenderGlitch(RuntimeError):
    """Isaac Sim stopped drawing the robot: a run on it would fail for a reason
    unrelated to the policy, so it is reported instead of scored."""


class EpisodeObserver:
    """Watches an episode; override what you need."""

    def on_reset(self, observation: Observation) -> None:
        """The first observation of the episode."""

    def before_step(self, observation: Observation, action: Action, info: dict) -> None:
        """`action` was chosen from `observation`; `info` is the previous step's."""

    def after_step(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
        reward: float,
        done: bool,
        info: dict,
    ) -> None:
        """The step happened; `observation` is the one `action` was chosen from."""


@dataclass
class EpisodeResult:
    seed: int
    steps: int
    reward: float
    info: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    #: The refused action's message when the safety gate stopped the episode.
    violation: str | None = None

    @property
    def success(self) -> bool:
        return self.violation is None and bool(self.info.get("success"))


def run_episode(
    runtime: RuntimeComposition,
    seed: int,
    observers: tuple[EpisodeObserver, ...] = (),
    max_steps: int | None = None,
) -> EpisodeResult:
    """Reset `runtime` on `seed` and run its policy until the episode ends.

    It ends on termination, truncation, `policy.done`, or a `SafetyViolation`
    (recorded in the result, not raised). `max_steps` defaults to the robot's
    own limit.
    """
    policy = runtime.policy
    if policy is None:
        raise ValueError("run_episode needs a runtime with a policy")
    observation = runtime.reset(seed=seed)
    rendered = getattr(runtime.robot, "robot_is_rendered", None)
    if rendered is not None and not rendered():
        raise RenderGlitch(f"seed {seed}: Isaac Sim is not drawing the robot")
    for observer in observers:
        observer.on_reset(observation)

    limit = max_steps if max_steps is not None else runtime.robot.cfg.max_steps
    result = EpisodeResult(seed=seed, steps=0, reward=0.0)
    for _ in range(limit):
        try:
            action = policy.act(observation)
            for observer in observers:
                observer.before_step(observation, action, result.info)
            following, reward, terminated, truncated, info = runtime.step(action)
        except SafetyViolation as exc:
            result.violation = str(exc)
            break
        done = terminated or truncated
        for observer in observers:
            observer.after_step(observation, action, following, reward, done, info)
        observation = following
        result.steps += 1
        result.reward += reward
        result.info, result.truncated = info, truncated
        if done or policy.done:
            break
    return result
