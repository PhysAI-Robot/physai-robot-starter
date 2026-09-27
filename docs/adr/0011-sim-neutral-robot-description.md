# 11. Sim-neutral robot description

## Status

Accepted

## Context

SO-101's grasp pads, wristframe site, wrist camera, and jaw-collision
disabling only existed as MuJoCo `MjSpec` calls in `sim/scenes/common.py`,
with the gripper-angle-dependent pad orientation re-derived from MuJoCo
forward kinematics on every build. Solver tuning (`solref`/`solimp`/
`condim`/`noslip_iterations`) lived in the same `ManipulationSceneConfig`
dataclass as portable geometry and friction. A second simulator (Isaac Sim)
could not reuse any of this without re-deriving the same numbers by hand,
and there was no single place recording where they came from.

## Decision

Add `physai.robots.description.RobotDescription`: a frozen dataclass schema
(frames, cameras, contact pads, actuators, per-simulator `sim_overrides`)
loaded from one `description.yaml` per robot, owned by the robot's own
package (`robots/so101/description.yaml`) rather than by any simulator.
Poses are given in the parent link's own frame — meters, `wxyz`
quaternions, radians — so the same numbers drive a MuJoCo `MjSpec`, a URDF
importer, or a USD stage unchanged. This module must never import a
simulator SDK (`mujoco`, `isaacsim`); `pyproject.toml`'s import-linter
contract enforces it.

Each simulator gets one generic builder that reads a `RobotDescription`:
`sim.scenes.common.apply_description()` for MuJoCo,
`isaac.description.{apply_actuators,apply_cameras,apply_frames,
apply_contact_friction}` for Isaac Sim. Solver-specific tuning that has no
cross-simulator equivalent (MuJoCo's `solref`/`solimp`, a future PhysX
material policy) lives under `sim_overrides`, keyed by simulator name, and
is read only by that simulator's own builder.

The SO-101 pad quaternions are baked FK output, not hand-picked: a
`derivation` block in the YAML records the reference gripper angle and
tilt they were computed at, and a test re-derives them independently and
checks they match to 1e-6, so a future re-tune stays traceable.

## Consequences

Adding a robot's sim-neutral data is one YAML file plus the existing
per-robot `scene_defaults()` factory; no simulator builder needs a
robot-name branch. Moving the pad/camera/frame numbers out of
`ManipulationSceneConfig` removed roughly a dozen dataclass fields that only
ever had a value for SO-101, plus the runtime-recomputed pad quaternions
that used to be re-derived from MuJoCo forward kinematics on every scene
build. Solver tuning does not carry over between simulators automatically —
a simulator without a matching contact model (PhysX vs. MuJoCo's soft
constraints) needs its own `sim_overrides` entry tuned separately; that
re-tuning is itself part of the sim-to-sim evaluation this schema exists
to enable, not a follow-up.
