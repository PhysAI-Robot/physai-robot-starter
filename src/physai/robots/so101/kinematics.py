"""SO-101 forward kinematics and IK: the shared arm solver with this arm's joints and pinch offset."""

from __future__ import annotations

import numpy as np

from ..arm_kinematics import (
    APPROACH_AXIS,
    PINCH_AXIS,
    TOP_DOWN,
    IKResult,
    top_down_quat,
)
from ..arm_kinematics import (
    ArmKinematics as _ArmKinematics,
)
from .contracts import ARM_JOINT_NAMES

PINCH_OFFSET = np.array([-0.0042, -0.0043, 0.0154])


class SO101Kinematics(_ArmKinematics):
    """FK/IK over the five arm joints of a compiled SO-101 model."""

    def __init__(
        self,
        model,
        ee_site: str = "gripperframe",
        joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
        pinch_offset=PINCH_OFFSET,
    ) -> None:
        super().__init__(model, ee_site, joint_names, pinch_offset)


# The name the scripts, tests and plugins import.
ArmKinematics = SO101Kinematics

__all__ = [
    "APPROACH_AXIS",
    "PINCH_AXIS",
    "PINCH_OFFSET",
    "TOP_DOWN",
    "ArmKinematics",
    "IKResult",
    "SO101Kinematics",
    "top_down_quat",
]
