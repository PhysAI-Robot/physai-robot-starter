"""The look every simulator's scene shares: sky, floor, and table colours.

Neutral module: `physai.sim.mujoco` and `physai.sim.isaac` stay independent
backends (an import-linter contract), so anything both must agree on lives
here instead of in either of them. Camera-based policies see these colours, so
a difference between simulators is a perception gap, not just cosmetics.
"""

from __future__ import annotations

# Matches the web viewer's Three.js scene background exactly
# (`scene.background = new THREE.Color(0xdfe6e2)` in
# src/physai/web/static/js/scene.js), so a render (the native viewer's free
# camera, a captured camera frame, an exported video) and the browser viewer
# show the same background colour instead of a simulator's own default sky.
STUDIO_SKY_RGB: tuple[float, float, float] = (0.8745, 0.9020, 0.8863)

# Matches the web viewer's checker-textured floor exactly (`checkerTexture`
# in scene.js uses the same two hex colours, #e3e9e4 / #9fb0a8). A
# robot-mounted camera (e.g. so101's "front") often frames the floor rather
# than open sky, so the floor's own texture, not just the skybox, needs to be
# in the same palette. The gap between the two tiles is wider than the
# sky/floor gap so the checker pattern stays legible without leaving the light
# palette.
STUDIO_FLOOR_RGB1: tuple[float, float, float] = (0.8902, 0.9137, 0.8941)
STUDIO_FLOOR_RGB2: tuple[float, float, float] = (0.6235, 0.6902, 0.6588)

# MuJoCo's floor is an infinite plane whose checker squares are 1/6 m wide, with
# their edges at multiples of 1/6 m from the origin; the square containing the
# origin is the dark one (`STUDIO_FLOOR_RGB2`). Measured from a front-camera
# render (scripts/compare_cameras.py), so another simulator can draw the same.
FLOOR_TILE_M: float = 1.0 / 6.0

# Isaac Sim light intensities (with linear, gamma-free output; see
# `sim.isaac.core.configure_render_output`), fitted against MuJoCo renders
# with scripts/compare_cameras.py on what a camera policy sees: the table
# colour (224, 216, 197 against MuJoCo's 226, 217, 199) and the cube's faces
# (wrist-camera cube mean 100, 30, 24, the same as MuJoCo's). Table brightness
# is about linear in dome + key (a sum near 1400 matches; much above it the
# table clips to white), the shadows' depth in their ratio. The far floor
# near the horizon still renders darker than MuJoCo's; no policy reads it.
ISAAC_DOME_INTENSITY: float = 780.0
ISAAC_KEY_INTENSITY: float = 600.0

# The manipulation scene's table top (`build_manipulation_spec`'s table geom).
TABLE_RGBA: tuple[float, float, float, float] = (0.75, 0.72, 0.66, 1.0)

__all__ = [
    "FLOOR_TILE_M",
    "STUDIO_FLOOR_RGB1",
    "STUDIO_FLOOR_RGB2",
    "STUDIO_SKY_RGB",
    "TABLE_RGBA",
]
