"""Gymnasium adapter for transport-neutral PhysAI robot/task runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..contracts import Action, ActionSpec, Observation, ObservationSpec
from ..control.safety import SafetyController
from ..robots.base import RobotPort

ObservationEncoder = Callable[[Observation], Mapping[str, Any]]
ActionDecoder = Callable[[Mapping[str, Any]], Action]


def _space_for_spec(spec: Any) -> spaces.Box:
    dtype = np.dtype(spec.dtype)
    if spec.minimum is None:
        if np.issubdtype(dtype, np.integer):
            minimum = np.iinfo(dtype).min
        else:
            minimum = -np.inf
    else:
        minimum = spec.minimum
    if spec.maximum is None:
        if np.issubdtype(dtype, np.integer):
            maximum = np.iinfo(dtype).max
        else:
            maximum = np.inf
    else:
        maximum = spec.maximum
    return spaces.Box(
        low=np.full(spec.shape, minimum, dtype=dtype),
        high=np.full(spec.shape, maximum, dtype=dtype),
        dtype=dtype,
    )


class GymnasiumAdapter(gym.Env):
    """Expose a PhysAI robot/task runtime through the Gymnasium API.

    The encoder and decoder make the canonical field layout explicit for each
    embodiment while task rewards and termination remain owned by the backend.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        backend: RobotPort,
        *,
        observation_spec: ObservationSpec,
        action_spec: ActionSpec,
        observation_encoder: ObservationEncoder,
        action_decoder: ActionDecoder,
        render_mode: str | None = None,
        safety: SafetyController | None = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render mode: {render_mode!r}")
        self.backend = backend
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.render_mode = render_mode
        self.safety = safety or SafetyController(backend.robot_spec)
        self._observation: Observation | None = None
        self.action_space = spaces.Dict({
            spec.name: _space_for_spec(spec)
            for spec in action_spec.fields
        })
        self.observation_space = spaces.Dict({
            spec.name: _space_for_spec(spec)
            for spec in (*observation_spec.fields, *observation_spec.cameras)
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._observation = self.backend.reset(seed=seed)
        self.backend.robot_spec.validate_observation(self._observation)
        values = dict(self.observation_encoder(self._observation))
        self.observation_spec.validate(values)
        info = {"seed": seed}
        randomization = getattr(self.backend, "randomization_metadata", None)
        if randomization is not None:
            info["randomization"] = randomization.as_dict()
        return values, info

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        self.action_spec.validate(action)
        command = self.action_decoder(action)
        command = self.safety.validate(self._observation, command)
        observation, reward, terminated, truncated, info = self.backend.step(command)
        self.backend.robot_spec.validate_observation(observation)
        self._observation = observation
        values = dict(self.observation_encoder(observation))
        self.observation_spec.validate(values)
        return values, float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._observation is None:
            raise RuntimeError("call reset() before render()")
        cameras = self._observation.images
        if not cameras:
            raise RuntimeError("rgb_array rendering requires an observation camera")
        return next(iter(cameras.values())).data.copy()

    def close(self) -> None:
        self.backend.close()
        self._observation = None


__all__ = ["GymnasiumAdapter"]
