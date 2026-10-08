"""Edit one region of a camera image at evaluation time, to find what a policy relies on.

`eval_policy.py --policy lerobot --policy-arg image_edit=NAME` applies the edit to
every frame the policy sees (`vla_adapter.LeRobotPolicy.build_batch`). The front camera
is fixed, so its regions come from geometry (the projected table top) and from colour
(the yellow arm, the red cube, the green target disc); everything else is background.
The wrist camera moves, so it only gets whole-image edits.

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

GRAY = 128

# Mean colour ratio Isaac Sim / MuJoCo per region of the front camera, and a per-channel
# gain for the wrist camera, measured on the two renders of one scene at one arm pose
# (`scripts/compare_cameras.py --dump`; see FINDINGS, M4).
FRONT_GAINS = {
    "background": (0.67, 0.70, 0.67),
    "table": (0.93, 0.94, 0.93),
    "cube": (1.16, 1.19, 1.16),
    "arm": (1.03, 0.77, 0.48),  # Isaac renders the yellow arm orange
}
WRIST_GAIN = (0.81, 0.78, 0.75)
# Gaussian sigma (pixels) that gives a MuJoCo frame Isaac Sim's Laplacian variance
# (sharpness): Isaac renders the wrist camera far smoother, the front camera a little.
BLUR_SIGMA = {"front": 0.5, "wrist": 1.0}


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0)).astype(np.uint8)


def robot_mask(image: np.ndarray) -> np.ndarray:
    """The yellow arm, found by hue so it does not depend on how bright it is."""
    r, g, b = (image[..., i].astype(float) for i in range(3))
    saturated = (r - b) / np.maximum(r, 1.0) >= 0.4
    yellow = (g - b) / np.maximum(r - b, 1.0) >= 0.5
    return (r >= 60) & saturated & yellow


def cube_mask(image: np.ndarray) -> np.ndarray:
    """Reddish pixels that are not the arm."""
    r, g, b = (image[..., i].astype(int) for i in range(3))
    return (r > g + 40) & (r > b + 30) & ~robot_mask(image)


def disc_mask(image: np.ndarray) -> np.ndarray:
    """The green target disc: green well above red and blue."""
    r, g, b = (image[..., i].astype(int) for i in range(3))
    return (g - np.maximum(r, b)) >= 30


def table_polygon_mask(
    shape: tuple[int, int], intrinsics, extrinsics, table_pos, table_size
) -> np.ndarray:
    """Pixels covered by the table top, from the camera calibration and the table box.

    `extrinsics` is the camera pose in the base frame in the optical convention
    (x right, y down, z forward), as `ImageFrame.extrinsics` carries it.
    """
    rotation = extrinsics.orientation.to_matrix()
    origin = extrinsics.position.as_array()
    top = table_pos[2] + table_size[2]
    corners = [
        (table_pos[0] + sx * table_size[0], table_pos[1] + sy * table_size[1], top)
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]
    pixels = []
    for corner in corners:
        x, y, z = rotation.T @ (np.asarray(corner) - origin)
        z = max(z, 1e-3)  # a corner behind the camera is clamped to the image edge
        pixels.append(
            (
                intrinsics.fx * x / z + intrinsics.cx,
                intrinsics.fy * y / z + intrinsics.cy,
            )
        )
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon(pixels, fill=1)
    return np.asarray(canvas, dtype=bool)


def front_regions(image: np.ndarray, table: np.ndarray) -> dict[str, np.ndarray]:
    """Disjoint masks: arm, cube, disc, very dark pixels, table and background."""
    arm, cube, disc = robot_mask(image), cube_mask(image), disc_mask(image)
    dark = image.max(axis=-1) < 70
    free = ~arm & ~cube & ~disc & ~dark
    return {
        "arm": arm,
        "cube": cube,
        "disc": disc,
        "table": free & table,
        "background": free & ~table,
    }


def _scaled(image: np.ndarray, mask: np.ndarray, gain) -> None:
    image[mask] = np.clip(image[mask] * np.asarray(gain, dtype=np.float32), 0, 255)


def edit_front(name: str, image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """`image` (H, W, 3 uint8) with the named edit applied; a new array."""
    if name == "blank_front":
        return np.full_like(image, GRAY)
    if name in ("blur", "blur_front"):
        return _blur(image, BLUR_SIGMA["front"])
    regions = front_regions(image, table)
    out = image.astype(np.float32)
    if name == "bg_dim":
        _scaled(out, regions["background"], FRONT_GAINS["background"])
    elif name == "table_dim":
        _scaled(out, regions["table"], FRONT_GAINS["table"])
    elif name == "cube_gain":
        _scaled(out, regions["cube"], FRONT_GAINS["cube"])
    elif name == "arm_tint":
        _scaled(out, regions["arm"], FRONT_GAINS["arm"])
    elif name == "bg_gray":
        out[regions["background"]] = GRAY
    elif name == "bg_flat":
        mask = regions["background"]
        if mask.any():
            out[mask] = out[mask].mean(axis=0)
    elif name == "isaac_like":
        for region, gain in FRONT_GAINS.items():
            _scaled(out, regions[region], gain)
    else:
        return image
    return np.clip(out, 0, 255).astype(np.uint8)


def edit_wrist(name: str, image: np.ndarray) -> np.ndarray:
    if name == "blank_wrist":
        return np.full_like(image, GRAY)
    if name in ("blur", "blur_wrist"):
        return _blur(image, BLUR_SIGMA["wrist"])
    if name == "arm_tint":
        out = image.astype(np.float32)
        _scaled(out, robot_mask(image), FRONT_GAINS["arm"])
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == "isaac_like":
        return np.clip(image * np.asarray(WRIST_GAIN, dtype=np.float32), 0, 255).astype(
            np.uint8
        )
    return image


EDITS = (
    "bg_dim",
    "bg_gray",
    "bg_flat",
    "table_dim",
    "cube_gain",
    "arm_tint",
    "isaac_like",
    "blank_front",
    "blank_wrist",
    "blur",
    "blur_front",
    "blur_wrist",
)


class ImageEditor:
    """Applies one named edit to the frames of an observation's cameras."""

    def __init__(self, name: str, table_pos, table_size) -> None:
        if name not in EDITS:
            raise ValueError(f"unknown image_edit {name!r}; choose from {EDITS}")
        self.name = name
        self.table_pos, self.table_size = table_pos, table_size
        self._table: np.ndarray | None = None

    def __call__(self, camera: str, frame) -> np.ndarray:
        """The frame's pixels after the edit (`frame` is an `ImageFrame`)."""
        if camera == "front":
            if self._table is None:
                self._table = table_polygon_mask(
                    frame.data.shape[:2],
                    frame.intrinsics,
                    frame.extrinsics,
                    self.table_pos,
                    self.table_size,
                )
            return edit_front(self.name, frame.data, self._table)
        return edit_wrist(self.name, frame.data)
