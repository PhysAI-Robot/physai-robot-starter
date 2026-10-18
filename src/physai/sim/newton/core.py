"""Newton physics for an env that keeps its state in a MuJoCo `MjData`.

Newton's `SolverMuJoCo` runs the same MJCF the MuJoCo backend simulates, so a
Newton env is its MuJoCo env with `mj_step` replaced: every substep runs on the
GPU, and the result is copied back into `data` so sites, kinematics, contacts
and cameras keep reading MuJoCo. Newton and MuJoCo agree on coordinates except
for the free-joint quaternion, which Newton stores as xyzw.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np


def _free_qadrs(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[joint])
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    ]


def qpos_to_newton(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """MuJoCo `qpos` as Newton `joint_q` (free-joint quaternion wxyz -> xyzw)."""
    q = np.asarray(qpos, dtype=np.float32).copy()
    for adr in _free_qadrs(model):
        q[adr + 3 : adr + 7] = qpos[[adr + 4, adr + 5, adr + 6, adr + 3]]
    return q


def qpos_from_newton(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Newton `joint_q` as MuJoCo `qpos` (free-joint quaternion xyzw -> wxyz)."""
    qpos = np.asarray(q, dtype=np.float64).copy()
    for adr in _free_qadrs(model):
        qpos[adr + 3 : adr + 7] = q[[adr + 6, adr + 3, adr + 4, adr + 5]]
    return qpos


@contextmanager
def scene_mjcf(
    spec: mujoco.MjSpec, model: mujoco.MjModel, robot_dir: Path
) -> Iterator[str]:
    """The scene as an MJCF file beside the robot's meshes, for Newton's parser.

    Mesh paths in the MJCF are relative to the robot's own directory, so the
    file goes there, under a per-process name (sharded evaluations run several
    processes against the same assets) and is removed after the load. The
    compiled `model`'s actuator force limits are copied onto the spec first:
    envs set them after compiling (a servo's torque cap), and the spec alone
    would export the model as shipped.
    """
    for index, actuator in enumerate(spec.actuators):
        actuator.forcerange = model.actuator_forcerange[index]
        actuator.forcelimited = int(model.actuator_forcelimited[index])
    path = Path(robot_dir) / f"_newton_{os.getpid()}.xml"
    path.write_text(spec.to_xml(), encoding="utf-8")
    try:
        yield str(path)
    finally:
        path.unlink(missing_ok=True)


class NewtonPhysics:
    """Mixin for a MuJoCo env: step with Newton, keep `self.data` current.

    Put it before the MuJoCo env in the bases and call `_start_newton()` at the
    end of `__init__`, after the env has finished editing its model. `reset`
    pushes the freshly reset `data` into Newton; `step_simulation` runs one
    control period there and copies the result back.
    """

    def _start_newton(self, robot_dir: Path) -> None:
        import newton
        import warp as wp

        # Domain randomization edits the MjModel after this point; Newton would
        # not see it. ponytail: rebuild the Newton model on each randomized
        # reset if physics randomization on Newton is ever needed.
        if self.cfg.domain_randomization.enabled:
            raise ValueError(
                "Newton physics does not support domain_randomization; "
                "use --sim mujoco for randomized physics"
            )
        with scene_mjcf(self.spec, self.model, robot_dir) as path:
            builder = newton.ModelBuilder()
            builder.add_mjcf(path, verbose=False, ctrl_direct=True)
        self._newton_model = builder.finalize()
        self._newton_solver = newton.solvers.SolverMuJoCo(
            self._newton_model,
            use_mujoco_contacts=True,
            nconmax=2048,
            njmax=4096,
        )
        if self._newton_model.joint_coord_count != self.model.nq:
            raise RuntimeError(
                "Newton and MuJoCo disagree on the scene's coordinates "
                f"({self._newton_model.joint_coord_count} != {self.model.nq})"
            )
        self._wp, self._newton = wp, newton
        self._s0, self._s1 = self._newton_model.state(), self._newton_model.state()
        self._newton_control = self._newton_model.control()
        self._newton_dt = float(self.model.opt.timestep)
        self._graph = None

    def _newton_substeps(self) -> None:
        for i in range(self.n_substeps):
            a, b = (self._s0, self._s1) if i % 2 == 0 else (self._s1, self._s0)
            self._newton_solver.step(a, b, self._newton_control, None, self._newton_dt)
        if self.n_substeps % 2:  # the result landed in s1; keep s0 the live state
            self._wp.copy(self._s0.joint_q, self._s1.joint_q)
            self._wp.copy(self._s0.joint_qd, self._s1.joint_qd)

    def _push_newton(self) -> None:
        self._s0.joint_q.assign(qpos_to_newton(self.model, self.data.qpos))
        self._s0.joint_qd.assign(self.data.qvel.astype(np.float32))
        self._newton.eval_fk(
            self._newton_model, self._s0.joint_q, self._s0.joint_qd, self._s0
        )

    def reset(self, seed: int | None = None):
        observation = super().reset(seed)
        self._push_newton()
        return observation

    def step_simulation(self) -> None:
        self._newton_control.mujoco.ctrl.assign(self.data.ctrl.astype(np.float32))
        if self._graph is not None:
            self._wp.capture_launch(self._graph)
        else:
            self._newton_substeps()  # the first run compiles kernels; capture the next
            if self._wp.get_device().is_cuda:
                with self._wp.ScopedCapture() as capture:
                    self._newton_substeps()
                self._graph = capture.graph
        self.data.qpos[:] = qpos_from_newton(self.model, self._s0.joint_q.numpy())
        # ponytail: a free joint's qvel is Newton's (world-frame) velocity, not
        # MuJoCo's body-frame one; nothing reads the cube's velocity.
        self.data.qvel[:] = self._s0.joint_qd.numpy()
        self.data.time += self.control_dt
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
