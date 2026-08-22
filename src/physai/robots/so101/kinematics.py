"""Forward kinematics and damped-least-squares IK for the SO-101."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ...contracts import ARM_JOINT_NAMES, Header, Pose, PoseStamped, Quaternion, Vector3

APPROACH_AXIS = "x"
PINCH_AXIS = "z"
TOP_DOWN = np.array([0.0, 0.0, -1.0])
PINCH_OFFSET = np.array([-0.0042, -0.0043, 0.0154])


@dataclass
class IKResult:
    qpos: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int
    converged: bool
    site_rotation: np.ndarray = None

    def axis(self, which: str) -> np.ndarray:
        return self.site_rotation[:, "xyz".index(which)].copy()


class ArmKinematics:
    """FK/IK over the five arm joints of a compiled SO-101 model."""

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

    def fk(self, data: mujoco.MjData) -> PoseStamped:
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
        rotation = data.site_xmat[self.site_id].reshape(3, 3)
        return data.site_xpos[self.site_id] + rotation @ np.asarray(offset, dtype=np.float64)

    def site_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

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
        model, data = self.model, self._scratch
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        axis_col = "xyz".index(approach_axis)
        if approach_dir is not None:
            approach_dir = np.asarray(approach_dir, dtype=np.float64).reshape(3)
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

        mujoco.mj_resetData(model, data)
        if q_init is not None:
            data.qpos[self.qpos_adr] = np.asarray(q_init, dtype=np.float64).reshape(5)

        error = np.zeros(6)
        position_error = rotation_error = np.inf
        iterations = 0
        for iterations in range(1, max_iters + 1):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            error[:3] = target_pos - data.site_xpos[self.site_id]
            position_error = float(np.linalg.norm(error[:3]))
            rotation = data.site_xmat[self.site_id].reshape(3, 3)

            if target_quat_wxyz is not None:
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, data.site_xmat[self.site_id].reshape(9))
                negative = np.zeros(4)
                mujoco.mju_negQuat(negative, current_quat)
                delta_quat = np.zeros(4)
                mujoco.mju_mulQuat(delta_quat, np.asarray(target_quat_wxyz), negative)
                velocity = np.zeros(3)
                mujoco.mju_quat2Vel(velocity, delta_quat, 1.0)
                error[3:] = velocity
            elif approach_dir is not None:
                current_axis = rotation[:, axis_col]
                cross = np.cross(current_axis, approach_dir)
                sine, cosine = np.linalg.norm(cross), float(np.dot(current_axis, approach_dir))
                angle = float(np.arctan2(sine, cosine))
                error[3:] = cross / sine * angle if sine > 1e-9 else 0.0
            else:
                error[3:] = 0.0
            rotation_error = float(np.linalg.norm(error[3:]))
            if position_error < pos_tol and rotation_error < rot_tol:
                break

            jacobian = self.site_jacobian(data)
            weights = np.diag([pos_weight] * 3 + [rot_weight] * 3)
            weighted_jacobian, weighted_error = weights @ jacobian, weights @ error
            system = weighted_jacobian @ weighted_jacobian.T + damping**2 * np.eye(6)
            delta = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
            target = data.qpos[self.qpos_adr] + step_scale * delta
            data.qpos[self.qpos_adr] = np.clip(target, self.limits[:, 0], self.limits[:, 1])

        return IKResult(
            qpos=data.qpos[self.qpos_adr].copy(),
            position_error=position_error,
            orientation_error=rotation_error,
            iterations=iterations,
            converged=position_error < pos_tol and rotation_error < rot_tol,
            site_rotation=data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def ik_pinch(self, object_center, approach_dir=TOP_DOWN, q_init=None,
                 *, pinch_offset=PINCH_OFFSET, **ik_kwargs) -> IKResult:
        object_center = np.asarray(object_center, dtype=np.float64).reshape(3)
        offset = np.asarray(pinch_offset, dtype=np.float64)
        first = self.ik(object_center, approach_dir, q_init=q_init, **ik_kwargs)
        target = object_center - first.site_rotation @ offset
        return self.ik(target, approach_dir, q_init=first.qpos, **ik_kwargs)

    def clip_to_limits(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(5)
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])


def top_down_quat(yaw: float = 0.0) -> np.ndarray:
    q_down = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_down, np.array([0.0, 1.0, 0.0]), np.pi / 2)
    q_yaw = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), yaw)
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, q_yaw, q_down)
    return result
