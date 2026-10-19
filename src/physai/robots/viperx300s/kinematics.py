"""Forward kinematics and damped-least-squares IK for the ALOHA follower arm.

The shared arm solver (`..arm_kinematics.ArmKinematics`) is joint-count agnostic; this
subclass only fixes what is specific to the ViperX 300 S: six joints, the `pinch`
site at the fingertip centre (so the pinch offset is zero), and a top-down grasp
whose fingers close along the world y axis.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..arm_kinematics import TOP_DOWN, ArmKinematics, IKResult, top_down_quat
from .contracts import ARM_JOINT_NAMES

# The `pinch` site is already at the fingertip centre, so there is no offset to add.
PINCH_OFFSET = np.zeros(3)


class ViperX300SKinematics(ArmKinematics):
    """FK/IK over the six arm joints of a compiled ViperX 300 S model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site: str = "pinch",
        joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    ) -> None:
        super().__init__(model, ee_site, joint_names, PINCH_OFFSET)

    def ik_pinch(
        self,
        object_center,
        approach_dir=TOP_DOWN,
        q_init=None,
        *,
        pinch_offset=None,
        yaw: float | None = 0.0,
        **ik_kwargs,
    ) -> IKResult:
        """IK that puts the pinch point on `object_center`, gripper pointing down.

        With `yaw` set (the default, 0) the full orientation is pinned: fingers close
        along the world y axis, so a cube lying square to the table is gripped on
        faces, whatever the arm's waist angle. `yaw=None` constrains only the
        approach direction and leaves the roll about it free.
        """
        if yaw is not None and approach_dir is not None:
            ik_kwargs.setdefault("target_quat_wxyz", top_down_quat(yaw))
        return super().ik_pinch(
            object_center,
            approach_dir,
            q_init=q_init,
            pinch_offset=pinch_offset,
            **ik_kwargs,
        )


__all__ = [
    "PINCH_OFFSET",
    "TOP_DOWN",
    "IKResult",
    "ViperX300SKinematics",
    "top_down_quat",
]
