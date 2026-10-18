"""Newton camera renderer for a scene whose state lives in a MuJoCo `MjData`.

The renderer owns a Newton model built from the scene's MJCF, with the floor
replaced by 1/6 m tiles in the studio's two colours (the checker floor is a cue
learned policies read). Each `render` copies the MuJoCo state in (joint
positions, the target pad, the camera pose), so any simulator that keeps
`data` current, MuJoCo, Newton or Isaac through its display mirror, can be
drawn by it.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..workspace import VIDEO_CAMERA, ViewCamera
from .core import qpos_to_newton, scene_mjcf

TILE = 1 / 6
FLOOR_LIGHT, FLOOR_DARK = (0.8902, 0.9137, 0.8941), (0.6235, 0.6902, 0.6588)
SKY = (223, 230, 226)
KEY_LIGHT_DIRECTION = (-0.35, 0.45, -1.0)


class NewtonRenderer:
    def __init__(
        self,
        spec: mujoco.MjSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        robot_dir: Path,
        width: int,
        height: int,
        view: ViewCamera | None = None,
    ) -> None:
        import newton
        import warp as wp
        from newton.sensors import SensorTiledCamera

        self._newton, self._wp = newton, wp
        self.model, self.data = model, data
        self.width, self.height = width, height
        self._view = view
        builder = newton.ModelBuilder()
        with scene_mjcf(spec, model, robot_dir) as path:
            builder.add_mjcf(
                path,
                verbose=False,
                force_show_colliders=True,  # the table, cube and target are plain geoms
                ignore_names=["physai_floor"],
            )
        solid = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        reach = round(2.0 / TILE)
        for i in range(-reach, reach):
            for j in range(-reach, reach):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(
                        (TILE * (i + 0.5), TILE * (j + 0.5), -0.005),
                        wp.quat_identity(),
                    ),
                    hx=TILE / 2,
                    hy=TILE / 2,
                    hz=0.005,
                    color=FLOOR_DARK if (i + j) % 2 == 0 else FLOOR_LIGHT,
                    cfg=solid,
                )
        self._newton_model = builder.finalize()
        self._state = self._newton_model.state()
        labels = list(self._newton_model.shape_label)
        self._pad = next(
            (i for i, name in enumerate(labels) if name.endswith("target_pad")), None
        )
        self._pad_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_pad"
        )
        self._sensor = SensorTiledCamera(self._newton_model)
        self._sensor.utils.create_default_light(
            enable_shadows=True, direction=wp.vec3f(*KEY_LIGHT_DIRECTION)
        )
        self._color = self._sensor.utils.create_color_image_output(width, height)
        self._rays: dict[float, object] = {}
        self._clear = SensorTiledCamera.ClearData(
            clear_color=SKY[0] | (SKY[1] << 8) | (SKY[2] << 16) | (255 << 24)
        )
        self._target: np.ndarray | None = None

    def render(self, name: str) -> np.ndarray:
        wp, newton = self._wp, self._newton
        mujoco.mj_camlight(self.model, self.data)  # cam_xpos/cam_xmat are current
        state = self._state
        state.joint_q.assign(qpos_to_newton(self.model, self.data.qpos))
        newton.eval_fk(self._newton_model, state.joint_q, state.joint_qd, state)
        self._move_target(self.model.geom_pos[self._pad_geom])
        self._newton_model.bvh_refit_shapes(state)
        if name == VIDEO_CAMERA and self._view is not None:
            fovy = self._view.fovy
            position, xyaxes = self._view.pose()
            # z, the backward axis, completes the right-handed frame.
            axes = np.stack([xyaxes[:3], xyaxes[3:], np.cross(xyaxes[:3], xyaxes[3:])])
            matrix = axes.T.reshape(9)
        else:
            camera = self.model.camera(name).id
            fovy = float(self.model.cam_fovy[camera])
            position, matrix = self.data.cam_xpos[camera], self.data.cam_xmat[camera]
        if fovy not in self._rays:
            self._rays[fovy] = self._sensor.utils.compute_camera_rays_pinhole(
                self.width, self.height, camera_fovs=float(np.radians(fovy))
            )
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, matrix)
        w, x, y, z = quat
        pose = wp.array(
            [[wp.transformf(wp.vec3f(*position), wp.quatf(x, y, z, w))]],
            dtype=wp.transformf,
        )
        self._sensor.update(
            state,
            pose,
            self._rays[fovy],
            color_image=self._color,
            clear_data=self._clear,
        )
        rgba = self._sensor.utils.to_rgba_from_color(self._color).numpy()
        return rgba.reshape(self.height, self.width, 4)[:, :, :3].copy()

    def _move_target(self, target: np.ndarray) -> None:
        if self._pad is None or (
            self._target is not None and np.allclose(target, self._target)
        ):
            return
        transforms = self._newton_model.shape_transform.numpy()
        transforms[self._pad, :3] = target
        self._newton_model.shape_transform.assign(transforms)
        self._target = np.array(target, dtype=float)

    def close(self) -> None:
        pass
