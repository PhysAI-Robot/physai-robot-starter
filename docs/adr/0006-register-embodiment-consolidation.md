# 6. Consolidate per-robot registration into one call

## Status

Accepted

## Context

Adding a robot to the direct-MuJoCo path required 3-6 separate registration
calls inside `robots/registry.py`'s `_load_builtins()` (factory, scene
defaults, env config, and optionally a ROS2 node, navigation baseline, and
per-robot policies) rather than the "one registration line" the architecture
aims for.

## Decision

Introduce a `RobotDescriptor` dataclass that bundles all of a robot's
factories (env factory, kind, scene-defaults factory, env-config factory,
optional ROS2-node factory, optional navigation factory, optional
jog-resolver factory, and a policies mapping) and one `register_embodiment(name,
descriptor)` call that registers all of them at once. `_load_builtins()`
builds one `RobotDescriptor` per robot and calls `register_embodiment` once
per robot instead of calling several `register_*` functions.

## Consequences

Adding a robot becomes: write the `robots/<name>/` package, build one
`RobotDescriptor`, call `register_embodiment` once. All existing public
lookup functions (`create_robot`, `robot_kind`, `scene_defaults`,
`create_ros2_node`, `navigate`, `create_robot_policy`, `available_robots`)
keep their signatures — this is an internal data-structure change, not a
public API break.
