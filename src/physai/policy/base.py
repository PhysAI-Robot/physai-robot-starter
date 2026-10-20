"""The policy interface every controller in this repo implements.

A policy maps (Observation, optional sub-goal) -> Action at the control rate.
Scripted experts, VLA checkpoints, and RL agents are all the same shape, so
`scripts/eval_policy.py` can run any of them against the same task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from ..contracts import Action, Observation, Twist, PoseStamped


class Policy(ABC):
    """Base class. Stateful policies (action chunking, RNNs) reset in `reset`."""

    name: str = "policy"
    # Seeds a trained policy was fitted on, when it knows them; `eval_policy.py`
    # warns when an evaluation reuses them (that measures recall, not generalisation).
    training_seeds: frozenset[int] | None = None

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        """Called once per episode, before the first `act`."""

    @abstractmethod
    def act(self, observation: Observation) -> Action: ...

    @property
    def done(self) -> bool:
        """True when a scripted/finite policy considers its job finished."""
        return False

    def episode_metrics(self) -> dict[str, Any]:
        """Extra per-episode fields for an evaluation row; read after the episode."""
        return {}


class ConstantPolicy(Policy):
    """Holds the starting pose. Useful as a sanity baseline and for testing."""

    name = "constant"

    def __init__(self, joint_count: int = 5) -> None:
        # Arm joints held, before the gripper. 5 is the SO-101's; a robot with a
        # different arm passes its own (see `_make_constant`).
        self._n = joint_count
        self._q = None

    def reset(self, observation: Observation, goal=None, instruction=None) -> None:
        self._q = observation.joint_state.position[: self._n].copy()

    def act(self, observation: Observation) -> Action:
        from ..contracts import GripperCommand

        q = (
            self._q
            if self._q is not None
            else observation.joint_state.position[: self._n]
        )
        return Action(joint_position=q, gripper=GripperCommand(position=1.0))


class ConstantTwistPolicy(Policy):
    """Hold a mobile base still for a generic simulation smoke test."""

    name = "constant_twist"

    def act(self, observation: Observation) -> Action:
        return Action(ee_twist=Twist())
