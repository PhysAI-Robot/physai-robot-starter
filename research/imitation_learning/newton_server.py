"""Render the front and wrist cameras with Newton, for a MuJoCo state sent over a pipe.

Runs in its own Python (`newton` pins `warp-lang`, which the Isaac Sim environment also
ships), so it imports nothing from this repository. `newton_eval.py` starts it, sends one
pickled request per camera frame and reads back the image:

    request  {"name", "qpos" (13), "target" (3), "pos" (3), "mat" (3x3), "fovy"}
    reply    uint8 (H, W, 3)

The scene is the MJCF `newton_eval.py` exports from the running MuJoCo scene. Newton draws
the arm, table, cube and target pad from it; the floor plane is replaced by 1/6 m tiles
in the studio's two colours, because the checker floor is a cue ACT reads
(`research/imitation_learning/FINDINGS.md`, M5). MuJoCo's camera poses are passed in, so
the views match to the pixel and only the renderer differs.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import pickle
import struct
import sys

# Warp prints its start-up banner to stdout, which carries the replies here.
PROTOCOL_OUT, sys.stdout = sys.stdout.buffer, sys.stderr

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import newton  # noqa: E402
from newton.sensors import SensorTiledCamera  # noqa: E402

TILE = 1 / 6
FLOOR_LIGHT, FLOOR_DARK = (0.8902, 0.9137, 0.8941), (0.6235, 0.6902, 0.6588)
SKY = (223, 230, 226)
KEY_LIGHT_DIRECTION = (-0.35, 0.45, -1.0)


class Renderer:
    def __init__(self, scene: str, width: int, height: int) -> None:
        self.width, self.height = width, height
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            scene,
            verbose=False,
            force_show_colliders=True,  # the table, cube and target are plain geoms
            ignore_names=["physai_floor"],
        )
        solid = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        for i in range(-12, 12):
            for j in range(-12, 12):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(
                        (TILE * (i + 0.5), TILE * (j + 0.5), -0.005), wp.quat_identity()
                    ),
                    hx=TILE / 2,
                    hy=TILE / 2,
                    hz=0.005,
                    color=FLOOR_DARK if (i + j) % 2 == 0 else FLOOR_LIGHT,
                    cfg=solid,
                )
        self.model = builder.finalize()
        self.state = self.model.state()
        labels = list(self.model.shape_label)
        self.pad = next(
            i for i, name in enumerate(labels) if name.endswith("target_pad")
        )
        self.sensor = SensorTiledCamera(self.model)
        self.sensor.utils.create_default_light(
            enable_shadows=True, direction=wp.vec3f(*KEY_LIGHT_DIRECTION)
        )
        self.color = self.sensor.utils.create_color_image_output(width, height)
        self.rays: dict[float, object] = {}
        self.clear = SensorTiledCamera.ClearData(
            clear_color=SKY[0] | (SKY[1] << 8) | (SKY[2] << 16) | (255 << 24)
        )
        self._target = None

    def render(self, request: dict) -> np.ndarray:
        self.state.joint_q = wp.array(
            request["qpos"].astype(np.float32), dtype=wp.float32
        )
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self._move_target(request["target"])
        self.model.bvh_refit_shapes(self.state)
        fovy = float(request["fovy"])
        if fovy not in self.rays:
            self.rays[fovy] = self.sensor.utils.compute_camera_rays_pinhole(
                self.width, self.height, camera_fovs=float(np.radians(fovy))
            )
        x, y, z, w = Rotation.from_matrix(request["mat"]).as_quat()
        pose = wp.array(
            [[wp.transformf(wp.vec3f(*request["pos"]), wp.quatf(x, y, z, w))]],
            dtype=wp.transformf,
        )
        self.sensor.update(
            self.state,
            pose,
            self.rays[fovy],
            color_image=self.color,
            clear_data=self.clear,
        )
        rgba = self.sensor.utils.to_rgba_from_color(self.color).numpy()
        return rgba.reshape(self.height, self.width, 4)[:, :, :3].copy()

    def _move_target(self, target) -> None:
        if self._target is not None and np.allclose(target, self._target):
            return
        transforms = self.model.shape_transform.numpy()
        transforms[self.pad, :3] = target
        self.model.shape_transform.assign(transforms)
        self._target = np.array(target, dtype=float)


def main() -> int:
    scene, width, height = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    renderer = Renderer(scene, width, height)
    stdin, stdout = sys.stdin.buffer, PROTOCOL_OUT
    stdout.write(b"ready\n")
    stdout.flush()
    while header := stdin.read(4):
        request = pickle.loads(stdin.read(struct.unpack("<I", header)[0]))
        reply = pickle.dumps(renderer.render(request))
        stdout.write(struct.pack("<I", len(reply)) + reply)
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
