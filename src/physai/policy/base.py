"""The policy interface every controller in this repo implements.

A policy maps (Observation, optional sub-goal) -> Action at the control rate.
Scripted experts, VLA checkpoints, and RL agents are all the same shape, so
`scripts/eval_policy.py` can run any of them against the same task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import Action, Observation, PoseStamped, Twist


class Policy(ABC):
    """Base class. Stateful policies (action chunking, RNNs) reset in `reset`."""

    name: str = "policy"

    def reset(self, observation: Observation, goal: PoseStamped | None = None,
              instruction: str | None = None) -> None:
        """Called once per episode, before the first `act`."""

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        ...

    @property
    def done(self) -> bool:
        """True when a scripted/finite policy considers its job finished."""
        return False


class ConstantPolicy(Policy):
    """Holds the starting pose. Useful as a sanity baseline and for testing."""

    name = "constant"

    def __init__(self) -> None:
        self._q = None

    def reset(self, observation: Observation, goal=None, instruction=None) -> None:
        self._q = observation.joint_state.position[:5].copy()

    def act(self, observation: Observation) -> Action:
        from ..contracts import GripperCommand

        q = self._q if self._q is not None else observation.joint_state.position[:5]
        return Action(joint_position=q, gripper=GripperCommand(position=1.0))


class ConstantTwistPolicy(Policy):
    """Hold a mobile base still for a generic simulation smoke test."""

    name = "constant_twist"

    def act(self, observation: Observation) -> Action:
        return Action(ee_twist=Twist())
