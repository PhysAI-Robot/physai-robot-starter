# 12. ImageFrame carries camera intrinsics and extrinsics

## Status

Accepted

## Context

`research/classical_control/so101_visual_servo.py` derived its camera
calibration (pinhole intrinsics, camera-to-base rotation and translation)
by reaching into `env.model`/`env.data` directly — MuJoCo internals a
policy should not depend on, and something no other `RobotPort` backend
(a ROS2-bridged robot, a future Isaac-backed one) can supply the same way.
`ImageFrame`'s own docstring already claimed to carry "the bits of
CameraInfo we care about," but had no intrinsics or extrinsics fields to
back that claim.

`contracts.py` is a frozen contract (`docs/ARCHITECTURE.md`'s "Frozen vs
free to change" table): changing it needs an ADR and a version marker,
with a deprecation window if anything existing depends on the old shape.

## Decision

Add `CameraIntrinsics` (`fx`, `fy`, `cx`, `cy`) and `Quaternion.to_matrix()`
(pure numpy, no simulator dependency) to `contracts.py`. `ImageFrame` gains
optional `intrinsics: CameraIntrinsics | None` and
`extrinsics: Pose | None` fields, both defaulting to `None`. `extrinsics`
is in the ROS optical-frame convention (REP 103: x right, y down, z
forward) — the same convention `sensor_msgs` consumers assume — so a
backend that reads a simulator's own camera-axis convention (MuJoCo: x
right, y up, z backward) converts once, in the one place that already
knows that convention, rather than every consumer needing to know it.

`SO101Env.camera_calibration()` performs that MuJoCo-to-ROS conversion.
`so101_visual_servo.py` reads calibration off the observation's `ImageFrame`
instead of the backend directly, so it works against any `RobotPort` that
fills those fields in.

Because both new fields default to `None`, no version marker or
deprecation window was needed: every existing `ImageFrame` construction and
consumer is unchanged, and nothing before this ADR read either field.

## Consequences

A vision-dependent policy no longer assumes a direct MuJoCo environment for
calibration; it works unchanged against any backend that fills in
`intrinsics`/`extrinsics` on its `ImageFrame`s (verified: `SO101IsaacEnv`
does not populate them yet, since it has no camera-consuming policy to
serve today). A host-driven session's async camera worker (`web/host.py`)
also had to be taught to fill in the new fields via a duck-typed
`camera_calibration()` lookup on the robot port — camera frames render on a
separate thread there and were not going through `SO101Env.observe()` at
all, a real gap this change surfaced rather than introduced.
