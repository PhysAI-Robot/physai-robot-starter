"""SO-101 pick-and-place environment.

Gym-like API (`reset` / `step`) but the observation and action types are the
ROS2-shaped dataclasses from `physai.contracts`, not raw arrays. That is the
whole point of Phase 0: the policy and planner code you write here talks the
same language it will talk to a ROS2 node in Phase 1.

Action space is absolute joint position targets (5 arm joints in radians) plus a
normalised gripper aperture — matching what LeRobot records on the real SO-101,
so demonstrations collected here are directly comparable to real ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    Action,
    GRIPPER_JOINT_NAME,
    GripperCommand,
    Header,
    ImageFrame,
    JointState,
    Observation,
)
from ..tasks import create_task
from ..robots.so101.kinematics import ArmKinematics
from .scene import EE_SITE, SceneConfig, build_model

HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)


@dataclass
class EnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    control_hz: float = 25.0
    render: bool = True
    cameras: tuple[str, ...] = ("front", "wrist")
    max_steps: int = 400
    task: str = "pick_place"

    # Domain randomisation. The default band is where the scripted expert is
    # actually reliable, not the full IK-reachable area — top-down grasps solve
    # over a much wider region than they succeed in, because the descent path
    # and the grasp itself add constraints IK does not model. Widen it if you
    # want harder demos, and run scripts/workspace_map.py first.
    randomize_cube: bool = True
    cube_x_range: tuple[float, float] = (0.20, 0.24)
    cube_y_range: tuple[float, float] = (0.05, 0.13)
    randomize_target: bool = False
    target_x_range: tuple[float, float] = (0.16, 0.26)
    target_y_range: tuple[float, float] = (-0.13, -0.04)

    success_xy_tol: float = 0.04
    success_hold_steps: int = 10
    seed: int | None = None


class SO101PickPlaceEnv:
    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.task = create_task(self.cfg.task, success_xy_tol=self.cfg.success_xy_tol)
        self.model, self.spec = build_model(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n_substeps = max(1, round((1.0 / self.cfg.control_hz) / self.model.opt.timestep))
        self.control_dt = self.n_substeps * self.model.opt.timestep

        # --- index bookkeeping -------------------------------------------
        def jid(name: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"joint {name!r} missing from model")
            return i

        def aid(name: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise KeyError(f"actuator {name!r} missing from model")
            return i

        self.arm_joint_ids = np.array([jid(n) for n in ARM_JOINT_NAMES])
        self.gripper_joint_id = jid(GRIPPER_JOINT_NAME)
        self.arm_qadr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_vadr = self.model.jnt_dofadr[self.arm_joint_ids]
        self.grip_qadr = int(self.model.jnt_qposadr[self.gripper_joint_id])
        self.grip_vadr = int(self.model.jnt_dofadr[self.gripper_joint_id])

        self.arm_act_ids = np.array([aid(n) for n in ARM_JOINT_NAMES])
        self.grip_act_id = aid(GRIPPER_JOINT_NAME)
        self.arm_limits = self.model.jnt_range[self.arm_joint_ids].copy()
        self.grip_limits = self.model.jnt_range[self.gripper_joint_id].copy()

        # Multi-cube (sorting) bookkeeping is additive: when num_cubes == 1 the
        # scene has no "cube_red" etc. bodies, so self.sorting_cubes stays
        # empty and every sorting-specific code path below is simply unused —
        # the original single-cube "cube" body and its properties (cube_pos,
        # cube_bid, ...) are untouched.
        self.sorting_cubes: dict[str, tuple[int, int]] = {}  # color -> (body_id, qadr)
        if self.cfg.scene.num_cubes == len(self.cfg.scene.sorting_cube_names):
            for name in self.cfg.scene.sorting_cube_names:
                color = name.removeprefix("cube_")
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                qadr = int(self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
                ])
                self.sorting_cubes[color] = (bid, qadr)
            self.cube_bid = self.cube_qadr = None
        else:
            self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            self.cube_qadr = int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
            ])
        self.target_color: str | None = None
        self.target_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self.kin = ArmKinematics(self.model, ee_site=EE_SITE)

        # --- rendering ----------------------------------------------------
        self._renderer: mujoco.Renderer | None = None
        if self.cfg.render:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.cfg.scene.camera_height,
                width=self.cfg.scene.camera_width,
            )

        self.step_count = 0
        self._success_streak = 0
        self._last_action = Action(joint_position=HOME_QPOS.copy())

    @property
    def robot_spec(self):
        """Generic embodiment metadata exposed to registries and tooling."""
        from ..robots.base import RobotSpec

        return RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=ALL_JOINT_NAMES,
            action_modes=("joint_position",),
            observation_modalities=("state", "images", "ee_pose"),
            capabilities=("joint_position", "arm_kinematics", "gripper", "images"),
            metadata={"control_hz": self.cfg.control_hz},
        )

    # ------------------------------------------------------------------
    # gripper unit conversion
    # ------------------------------------------------------------------
    def gripper_to_joint(self, normalized: float) -> float:
        lo, hi = self.grip_limits
        return float(lo + np.clip(normalized, 0.0, 1.0) * (hi - lo))

    def joint_to_gripper(self, q: float) -> float:
        lo, hi = self.grip_limits
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self.data.qpos[self.grip_qadr] = self.gripper_to_joint(1.0)

        if self.sorting_cubes:
            # Three non-overlapping y-bands at the historically-reliable grasp
            # x-range, shuffled per episode so the expert/policy can't learn a
            # fixed color->slot shortcut instead of actually looking.
            base_z = self.cfg.scene.cube_pos[2]
            y_bands = [(0.00, 0.02), (0.06, 0.08), (0.12, 0.14)]
            colors = list(self.sorting_cubes.keys())
            self.rng.shuffle(colors)
            for color, (y_lo, y_hi) in zip(colors, y_bands):
                bid, qadr = self.sorting_cubes[color]
                pos = np.array([0.0, 0.0, base_z], dtype=np.float64)
                if self.cfg.randomize_cube:
                    pos[0] = self.rng.uniform(*self.cfg.cube_x_range)
                    pos[1] = self.rng.uniform(y_lo, y_hi)
                else:
                    pos[0] = self.cfg.scene.cube_pos[0]
                    pos[1] = (y_lo + y_hi) / 2
                self.data.qpos[qadr:qadr + 3] = pos
                self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            self.target_color = self.rng.choice(list(self.sorting_cubes.keys()))
        else:
            cube_pos = np.array(self.cfg.scene.cube_pos, dtype=np.float64)
            if self.cfg.randomize_cube:
                cube_pos[0] = self.rng.uniform(*self.cfg.cube_x_range)
                cube_pos[1] = self.rng.uniform(*self.cfg.cube_y_range)
            self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = cube_pos
            self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [1, 0, 0, 0]

        if self.cfg.randomize_target:
            tp = self.model.site_pos[self.target_sid].copy()
            tp[0] = self.rng.uniform(*self.cfg.target_x_range)
            tp[1] = self.rng.uniform(*self.cfg.target_y_range)
            self.model.site_pos[self.target_sid] = tp
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_pad")
            self.model.geom_pos[gid] = tp

        self.data.ctrl[self.arm_act_ids] = HOME_QPOS
        self.data.ctrl[self.grip_act_id] = self.gripper_to_joint(1.0)

        mujoco.mj_forward(self.model, self.data)
        self.task.reset(self, self.rng)
        self.step_count = 0
        self._success_streak = 0
        self._last_action = Action(joint_position=HOME_QPOS.copy(),
                                   gripper=GripperCommand(position=1.0))
        return self.observe()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        if action.joint_position is None:
            raise ValueError(
                "Action.joint_position is required by this env. Convert a Twist "
                "command with physai.control.TwistToJointResolver first."
            )
        q_arm = np.clip(action.joint_position.reshape(5),
                        self.arm_limits[:, 0], self.arm_limits[:, 1])
        self.data.ctrl[self.arm_act_ids] = q_arm
        self.data.ctrl[self.grip_act_id] = self.gripper_to_joint(action.gripper.clipped())

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._last_action = action

        obs = self.observe()
        info = self.task_state()
        reward = self.reward(info)

        self._success_streak = self._success_streak + 1 if info["at_target"] else 0
        info["success"] = self._success_streak >= self.cfg.success_hold_steps

        terminated = bool(info["success"]) or bool(info["cube_dropped"])
        truncated = self.step_count >= self.cfg.max_steps
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def joint_state(self) -> JointState:
        pos = np.concatenate([self.data.qpos[self.arm_qadr],
                              [self.data.qpos[self.grip_qadr]]])
        vel = np.concatenate([self.data.qvel[self.arm_vadr],
                              [self.data.qvel[self.grip_vadr]]])
        eff = np.concatenate([self.data.actuator_force[self.arm_act_ids],
                              [self.data.actuator_force[self.grip_act_id]]])
        return JointState(
            name=ALL_JOINT_NAMES,
            position=pos,
            velocity=vel,
            effort=eff,
            header=Header(stamp=float(self.data.time), frame_id="base"),
        )

    def render_camera(self, name: str) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("env constructed with render=False")
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def observe(self) -> Observation:
        images: dict[str, ImageFrame] = {}
        if self._renderer is not None:
            for cam in self.cfg.cameras:
                images[cam] = ImageFrame(
                    data=self.render_camera(cam),
                    camera_name=cam,
                    header=Header(stamp=float(self.data.time), frame_id=cam),
                )
        return Observation(
            joint_state=self.joint_state(),
            images=images,
            ee_pose=self.kin.fk(self.data),
            step=self.step_count,
            sim_time=float(self.data.time),
        )

    # ------------------------------------------------------------------
    # task
    # ------------------------------------------------------------------
    @property
    def cube_pos(self) -> np.ndarray:
        if self.sorting_cubes:
            bid, _ = self.sorting_cubes[self.target_color]
            return self.data.xpos[bid].copy()
        return self.data.xpos[self.cube_bid].copy()

    @property
    def cube_positions(self) -> dict[str, np.ndarray]:
        """color -> world position, sorting mode only (empty dict otherwise)."""
        return {color: self.data.xpos[bid].copy() for color, (bid, _) in self.sorting_cubes.items()}

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_sid].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.kin.site_id].copy()

    @property
    def table_top(self) -> float:
        return self.cfg.scene.table_pos[2] + self.cfg.scene.table_size[2]

    @property
    def cube_half(self) -> float:
        return self.cfg.scene.cube_half

    def task_state(self) -> dict:
        return self.task.evaluate(self)

    def reward(self, info: dict | None = None) -> float:
        return self.task.reward(self, info)
