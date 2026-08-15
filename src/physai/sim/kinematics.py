"""Forward kinematics helpers and a damped-least-squares IK for the SO-101.

The IK solves only for the 5 arm joints (the gripper joint is commanded
separately), driving the `gripperframe` site to a target pose. It is a
Jacobian-based iterative solver run on a scratch MjData, so it never disturbs
the live simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..contracts import ARM_JOINT_NAMES, Header, Pose, PoseStamped, Quaternion, Vector3

# Measured from so101_new_calib.xml at the `gripperframe` site:
#   +x points out of the jaws (approach direction)
#   +z is the jaw opening/closing direction
#   +y completes the frame
APPROACH_AXIS = "x"
PINCH_AXIS = "z"
TOP_DOWN = np.array([0.0, 0.0, -1.0])

# Vector from the site origin to the midpoint between the jaw faces, in
# site-local coordinates, measured on the composed scene at the grasp aperture.
# The pads are not centred on the site, so this is not a pure +z offset.
PINCH_OFFSET = np.array([-0.0042, -0.0043, 0.0154])


@dataclass
class IKResult:
    qpos: np.ndarray          # (5,) arm joint targets
    position_error: float     # metres
    orientation_error: float  # radians
    iterations: int
    converged: bool
    site_rotation: np.ndarray = None  # (3,3) EE site frame at the solution

    def axis(self, which: str) -> np.ndarray:
        """World-frame unit vector of the site's local x/y/z at the solution."""
        return self.site_rotation[:, "xyz".index(which)].copy()


class ArmKinematics:
    """FK/IK over the 5 arm joints of a compiled SO-101 model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site: str = "gripperframe",
        joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    ) -> None:
        self.model = model
        self.joint_names = joint_names
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self.site_id < 0:
            raise KeyError(f"site {ee_site!r} not in model")

        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        )
        if (self.joint_ids < 0).any():
            missing = [n for n, i in zip(joint_names, self.joint_ids) if i < 0]
            raise KeyError(f"joints not in model: {missing}")
        self.qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.dof_adr = model.jnt_dofadr[self.joint_ids]
        self.limits = model.jnt_range[self.joint_ids].copy()

        self._scratch = mujoco.MjData(model)

    # -- forward -------------------------------------------------------
    def fk(self, data: mujoco.MjData) -> PoseStamped:
        """End-effector pose in the world/base frame from a live MjData."""
        pos = data.site_xpos[self.site_id].copy()
        mat = data.site_xmat[self.site_id].reshape(9).copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return PoseStamped(
            pose=Pose(
                position=Vector3.from_array(pos),
                orientation=Quaternion.from_mujoco(quat),
            ),
            header=Header(frame_id="base"),
        )

    def pinch_center(self, data: mujoco.MjData, offset=PINCH_OFFSET) -> np.ndarray:
        """Point midway between the jaw faces — where a grasped object sits."""
        R = data.site_xmat[self.site_id].reshape(3, 3)
        return data.site_xpos[self.site_id] + R @ np.asarray(offset, dtype=np.float64)

    def site_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        """(6, 5) Jacobian of the EE site w.r.t. the arm joints."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

    # -- inverse -------------------------------------------------------
    def ik(
        self,
        target_pos,
        approach_dir=None,
        q_init=None,
        *,
        approach_axis: str = APPROACH_AXIS,
        target_quat_wxyz=None,
        max_iters: int = 150,
        pos_tol: float = 1e-3,
        rot_tol: float = 3e-2,
        damping: float = 5e-2,
        step_scale: float = 0.6,
        pos_weight: float = 1.0,
        rot_weight: float = 0.3,
    ) -> IKResult:
        """Damped least squares IK for the 5 arm joints.

        Three orientation modes, in order of usefulness on this arm:

        * ``approach_dir=[0,0,-1]`` — align the gripper's approach axis (+x of
          the ``gripperframe`` site) with a world direction. Two rotational
          constraints, leaving the roll about that axis free. **This is the
          right mode for a 5-DoF arm** and what the scripted expert uses.
        * ``approach_dir=None`` — position only.
        * ``target_quat_wxyz=...`` — full 6-DoF pose. Generally unreachable on
          5 DoF; ``rot_weight`` is < 1 so the solver degrades gracefully
          instead of oscillating.
        """
        model, d = self.model, self._scratch
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        axis_col = "xyz".index(approach_axis)

        if approach_dir is not None:
            approach_dir = np.asarray(approach_dir, dtype=np.float64).reshape(3)
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

        mujoco.mj_resetData(model, d)
        if q_init is not None:
            d.qpos[self.qpos_adr] = np.asarray(q_init, dtype=np.float64).reshape(5)

        err = np.zeros(6)
        pos_err = rot_err = np.inf
        it = 0
        for it in range(1, max_iters + 1):
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)

            err[:3] = target_pos - d.site_xpos[self.site_id]
            pos_err = float(np.linalg.norm(err[:3]))
            R = d.site_xmat[self.site_id].reshape(3, 3)

            if target_quat_wxyz is not None:
                cur_q = np.zeros(4)
                mujoco.mju_mat2Quat(cur_q, d.site_xmat[self.site_id].reshape(9))
                neg = np.zeros(4)
                mujoco.mju_negQuat(neg, cur_q)
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, np.asarray(target_quat_wxyz, dtype=np.float64), neg)
                vel = np.zeros(3)
                mujoco.mju_quat2Vel(vel, dq, 1.0)
                err[3:] = vel
            elif approach_dir is not None:
                # Rotation that carries the current approach axis onto the
                # desired direction: omega = a x d, magnitude = angle.
                a = R[:, axis_col]
                cross = np.cross(a, approach_dir)
                s, c = np.linalg.norm(cross), float(np.dot(a, approach_dir))
                angle = float(np.arctan2(s, c))
                err[3:] = (cross / s * angle) if s > 1e-9 else np.zeros(3)
            else:
                err[3:] = 0.0
            rot_err = float(np.linalg.norm(err[3:]))

            if pos_err < pos_tol and rot_err < rot_tol:
                break

            J = self.site_jacobian(d)
            W = np.diag([pos_weight] * 3 + [rot_weight] * 3)
            Jw, ew = W @ J, W @ err
            # dq = J^T (J J^T + λ²I)^-1 e
            JJt = Jw @ Jw.T + (damping ** 2) * np.eye(6)
            dq = Jw.T @ np.linalg.solve(JJt, ew)

            q = d.qpos[self.qpos_adr] + step_scale * dq
            d.qpos[self.qpos_adr] = np.clip(q, self.limits[:, 0], self.limits[:, 1])

        return IKResult(
            qpos=d.qpos[self.qpos_adr].copy(),
            position_error=pos_err,
            orientation_error=rot_err,
            iterations=it,
            converged=pos_err < pos_tol and rot_err < rot_tol,
            site_rotation=d.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def ik_pinch(
        self,
        object_center,
        approach_dir=TOP_DOWN,
        q_init=None,
        *,
        pinch_offset=PINCH_OFFSET,
        **ik_kwargs,
    ) -> IKResult:
        """IK that centres an object between the jaw faces, not on the EE site.

        The `gripperframe` site sits on the *fixed* jaw, so aiming it at an
        object's centre drives that jaw straight into the object. This solves
        for the site pose that puts the pinch centre on the target instead.

        Two passes: solve once to learn the site's roll about the approach axis
        (free on a 5-DoF arm), then re-solve against the corrected target.
        """
        object_center = np.asarray(object_center, dtype=np.float64).reshape(3)
        offset = np.asarray(pinch_offset, dtype=np.float64)
        first = self.ik(object_center, approach_dir, q_init=q_init, **ik_kwargs)
        target = object_center - first.site_rotation @ offset
        return self.ik(target, approach_dir, q_init=first.qpos, **ik_kwargs)

    def clip_to_limits(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(5)
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])


def top_down_quat(yaw: float = 0.0) -> np.ndarray:
    """Full quaternion (w,x,y,z) for approach=-Z with the jaws opening along Y.

    Only useful with ``ik(..., target_quat_wxyz=...)``. On a 5-DoF arm this is
    over-constrained — prefer ``ik(pos, approach_dir=TOP_DOWN)``.
    """
    # Bring site +x onto world -z (rotate +90 deg about world +y),
    # then spin about the approach axis by `yaw`.
    q_down = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_down, np.array([0.0, 1.0, 0.0]), np.pi / 2)
    q_yaw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), yaw)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_yaw, q_down)
    return out
