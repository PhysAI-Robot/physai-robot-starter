# Contract and description decisions

ADRs 11, 12 and 17: data that every simulator and robot reads, kept free of any
simulator SDK.

## ADR 11: Sim-neutral robot description

Status: accepted. Module paths were moved by
[ADR 14](simulators.md#adr-14-simulator-package-layout-and-adapter-names)
(`sim.scenes.common` is now `sim.mujoco.scenes.common`, `isaac.description` is
now `sim.isaac.description`); the decision is unchanged.

**Context.** SO-101's grasp pads, wrist frame, wrist camera and jaw-collision
settings existed only as MuJoCo `MjSpec` calls, with solver tuning mixed into
the same dataclass as portable geometry. A second simulator could not reuse any
of it without re-deriving the numbers.

**Decision.** `physai.robots.description.RobotDescription` is a frozen schema
(frames, cameras, contact pads, actuators, per-simulator `sim_overrides`) loaded
from one `description.yaml` owned by the robot's package. Poses are in the
parent link's frame (metres, `wxyz` quaternions, radians). The module never
imports a simulator SDK, enforced by import-linter. Each simulator has one
builder: `apply_description()` for MuJoCo, `apply_actuators`, `apply_cameras`,
`apply_frames` and `apply_contact_friction` for Isaac. Tuning with no
cross-simulator equivalent (MuJoCo `solref`/`solimp`) lives under
`sim_overrides` and is read only by that simulator. The SO-101 pad quaternions
are baked forward-kinematics output; a `derivation` block records how, and a
test re-derives them to 1e-6.

**Consequences.** A robot's sim-neutral data is one YAML file plus its
`scene_defaults()`; no builder needs a robot-name branch. Solver tuning does not
carry across engines and is re-tuned per engine, which is part of the sim-to-sim
evaluation, not a follow-up.

## ADR 12: `ImageFrame` carries camera intrinsics and extrinsics

Status: accepted.

**Context.** The visual-servo policy derived its calibration from `env.model` and
`env.data`, which no other `RobotPort` (ROS2-bridged, Isaac) can provide.
`contracts.py` is frozen, so changing it needs an ADR.

**Decision.** Add `CameraIntrinsics` (`fx`, `fy`, `cx`, `cy`) and
`Quaternion.to_matrix()`. `ImageFrame` gains optional `intrinsics` and
`extrinsics: Pose | None`, both defaulting to `None`. `extrinsics` uses the ROS
optical-frame convention (x right, y down, z forward), so a backend converts
once from its own convention (MuJoCo: x right, y up, z backward). The policy
reads calibration off the observation.

**Consequences.** Defaulting to `None` meant no version marker or deprecation
window. A vision policy works against any backend that fills the fields; the web
host's camera worker had to call the robot's `camera_calibration()` itself,
because its frames do not pass through `observe()`.

## ADR 17: Sim-neutral workspace description

Status: accepted.

**Context.** The table, target, cubes and front camera were described by
`ManipulationSceneConfig` inside `physai.sim.mujoco`, mixed with MuJoCo-only
construction. Isaac may not import that package, so the Isaac env read the scene
as `Any`, translated it into its own table, cube and camera config types, and
mirrored MuJoCo's hardcoded friction, colour and field-of-view values with
nothing enforcing the match.

**Decision.** `physai.sim.workspace` (neutral, like `sim/studio.py`) holds
`WorkspaceConfig` (table, target, front camera, resolution, `cubes()`),
`CubeSpec`, and the shared constants `TABLE_FRICTION`, `CUBE_FRICTION`,
`TARGET_RGBA`, `FRONT_CAMERA_FOVY_DEG`. An import-linter contract keeps it free
of both simulators. `WorldSceneConfig` inherits it and keeps only MuJoCo-only
fields, so scene class names, import paths and field names are unchanged
(manifest overrides and dataset metadata keep working). Each task scene
describes its cubes through `cubes()`. Each backend has one builder reading the
description: `build_manipulation_spec` and `sim.isaac.scene.add_workspace`. The
Isaac env only calls the builder and resets the cube; its table, cube and
camera config types and override fields are removed. The shared values are
constants, not config fields, because nothing varies them per run.

**Consequences.** A value both engines must agree on is written once, and a test
checks the compiled MuJoCo scene against it. A new engine adds one builder and
no translation layer. `SO101IsaacEnv` still builds a MuJoCo model of the scene
for its viewer mirror and kinematics, so its `scene` is typed
`ManipulationSceneConfig`; only `physai.sim.isaac` itself is free of MuJoCo.
Moving the scene registry out of `sim/mujoco/scenes/` is left as a separate
decision.
