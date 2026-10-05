# 17. Sim-neutral workspace description

## Status

Accepted

## Context

The table, target pad, cubes and front camera were described by
`ManipulationSceneConfig` inside `physai.sim.mujoco`, mixed with MuJoCo-only
construction (`build_spec()`/`build_model()`, `robot_xml`, `timestep`). Isaac
Sim may not import `physai.sim.mujoco`, so `SO101IsaacEnv` took the scene as
`Any`, read it by attribute name, and translated it into its own
`TableConfig`, `GraspCubeConfig` and `FrontCameraConfig`, whose defaults
"mirrored" values MuJoCo hardcoded (table and cube friction, target colour,
front-camera field of view). Nothing enforced the mirror, and the robot env
built the table, pad, cube and camera prims itself.

## Decision

Add `physai.sim.workspace`, a neutral module in the style of
[`sim/studio.py`](../../src/physai/sim/studio.py) and
[ADR 11](0011-sim-neutral-robot-description.md)'s `RobotDescription`:
`WorkspaceConfig` (table, target, front camera, camera resolution, and
`cubes()`), `CubeSpec`, and the shared constants (`TABLE_FRICTION`,
`CUBE_FRICTION`, `TARGET_RGBA`, `FRONT_CAMERA_FOVY_DEG`). It imports neither
simulator; an import-linter contract in `pyproject.toml` enforces that.

`WorldSceneConfig` inherits `WorkspaceConfig` and keeps only the MuJoCo-only
fields, so `ManipulationSceneConfig`, the two task scenes and every field name
are unchanged: manifest `scene.overrides` and dataset `scene_config` metadata
keep working. Each task scene describes its cubes through `cubes()`.

Each backend has one builder that reads the description:
`sim.mujoco.scenes.common.build_manipulation_spec` and
`sim.isaac.scene.add_workspace`. The Isaac env only calls the builder and
resets the cube each episode. `IsaacEnvConfig.scene` is typed, and the
`cube`/`table`/`front_camera`/`target_pos` override fields and their config
types are removed: a bare test scene is a default scene config with
`randomize_cube=False`.

The shared values are module constants rather than config fields: nothing
varies them per run, so a field would add keys to dataset metadata and the
manifest surface without a use.

## Consequences

A value both engines must agree on is written once, and a test checks the
compiled MuJoCo scene against it. A new simulator adds one builder for
`WorkspaceConfig` and no translation layer. The scene config classes stay in
`physai.sim.mujoco`, and `SO101IsaacEnv` still builds a MuJoCo model of the
scene for its viewer mirror and kinematics, so it types `scene` as
`ManipulationSceneConfig`; only `physai.sim.isaac` itself is free of MuJoCo.
Moving the scene registry out of `sim/mujoco/scenes/` is left for a separate
decision.
