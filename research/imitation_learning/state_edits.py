"""Replace the gripper reading a policy gets, to test whether it relies on it.

MuJoCo's gripper moves toward its command at a fixed 0.0142 rad per control step
(measured on the demonstrations), so after a command its reading ramps for about 4 s.
Isaac Sim's drive reaches the command within a few steps. Both read about 0.18 once
the cube blocks the jaws. `eval_policy.py --policy-arg state_edit=NAME` makes the
policy see the other simulator's behaviour, with no change to what the robot does:

    gripper_track  the reading follows the last command at once (Isaac-like)
    gripper_lag    the reading moves toward the last command by at most 0.0142
                   per step (MuJoCo-like)

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import numpy as np

SLEW_PER_STEP = 0.0142
HOLD_READING = 0.18  # the jaws stop here on the cube in both simulators
STATE_EDITS = ("gripper_track", "gripper_lag")


class GripperStateEditor:
    def __init__(self, name: str) -> None:
        if name not in STATE_EDITS:
            raise ValueError(f"unknown state_edit {name!r}; choose from {STATE_EDITS}")
        self.name = name
        self._reading: float | None = None

    def reset(self) -> None:
        self._reading = None

    def __call__(self, state: np.ndarray, last_command: float | None) -> np.ndarray:
        """`state` (joint positions, gripper last) with the gripper reading replaced."""
        edited = np.array(state, dtype=np.float32, copy=True)
        if self._reading is None:
            self._reading = float(edited[-1])
        if last_command is not None:
            if self.name == "gripper_track":
                self._reading = float(last_command)
            else:
                step = np.clip(
                    last_command - self._reading, -SLEW_PER_STEP, SLEW_PER_STEP
                )
                self._reading += float(step)
        edited[-1] = (
            max(self._reading, HOLD_READING) if last_command is not None else edited[-1]
        )
        return edited
