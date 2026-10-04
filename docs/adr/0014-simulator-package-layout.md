# 14. Simulator package layout and transport-adapter naming

## Status

Accepted

## Context

`physai.sim` held MuJoCo's simulation core, shared world, and scene
orchestration under a name that said nothing about which engine it was;
`physai.isaac` (added by [ADR 13](0013-isaac-sim-optional-backend.md)) sat
next to it as a flat, unrelated-looking sibling instead of its visible peer.
Nothing about the source tree said "these two are the same kind of thing."

Separately, `robots.adapters.DirectMuJoCoAdapter` and
`bridge.adapters.ROS2MuJoCoAdapter` both wrap any `RobotPort` — neither has
ever contained MuJoCo-specific code — but their names claimed otherwise.
ADR 13 already relied on this fact (`DirectMuJoCoAdapter` wraps
`SO101IsaacEnv` unchanged, with no `direct_isaac` adapter), so the names were
actively misleading about an abstraction the code already had right. The
registered adapter names (`direct_mujoco`, `ros2_mujoco`, `ros2_hardware`)
also didn't match the manifest's own `backend` vocabulary
(`direct`/`ros2_sim`/`ros2_real`), which are two names for what should be one
concept.

## Decision

Nest both engine backends as peer subpackages of one namespace:

```
physai/sim/
├── __init__.py   docstring only — importing physai.sim must not require
│                 either engine's SDK
├── mujoco/       was physai/sim/*.py (core, world, domain_randomization,
│                 scenes/) — unchanged behavior, only the path grew a
│                 segment
└── isaac/        was physai/isaac/* — unchanged behavior
```

`robots/so101/env.py` is renamed `robots/so101/mujoco_env.py`, matching its
sibling `isaac_env.py`; `SO101Env`'s own name and code are unchanged, this is
a file rename only. `robots/turtlebot/env.py` is untouched — it has one
engine, so a rename would document a distinction that doesn't exist yet.

Rename the transport adapters to match the manifest's `backend` field
one-to-one, since a transport adapter is not simulator-specific:

| Before | After | Registered as |
| --- | --- | --- |
| `DirectMuJoCoAdapter` | `DirectAdapter` | `direct` (was `direct_mujoco`) |
| `ROS2MuJoCoAdapter` | `ROS2SimAdapter` | `ros2_sim` (was `ros2_mujoco`) |
| `ROS2HardwareAdapter` | `ROS2HardwareAdapter` (unchanged) | `ros2_real` (was `ros2_hardware`) |

No compatibility alias is kept for either the module paths or the adapter
names: every import site and every `adapter=`/`create_adapter()` caller in
`src/`, `scripts/`, and `tests/` is updated in the same change, and this is a
pre-1.0 internal rename, not a published contract change to a deployed
manifest.

`pyproject.toml`'s import-linter contracts are updated for the new paths and
gain two new rules: `physai.sim.mujoco` and `physai.sim.isaac` must not
import each other (an `independence` contract — they are peer backends, not
one implementing the other), and `physai.sim.isaac` must not import
`mujoco`.

## Consequences

The folder tree change touches a frozen boundary
(`docs/ARCHITECTURE.md`'s "Frozen vs. free to change" table), which is why it
needs this ADR. `sim/`'s own shape — one subpackage per engine, itself
frozen going forward — becomes the pattern a third engine (if one is ever
added) follows, rather than a second flat sibling.

The adapter rename means `robots.registry.create_robot`'s default parameter
(`adapter="direct"`) and every robot factory's default changed together; a
caller passing the old string names (`"direct_mujoco"`, `"ros2_mujoco"`,
`"ros2_hardware"`) now gets the same `unknown adapter` error as any other
typo, with the current names listed. No manifest or `RobotDescriptor` field
changes — `SessionManifest.backend`'s three accepted values
(`direct`/`ros2_sim`/`ros2_real`) already matched the target adapter names,
so this rename only removed a translation that never needed to exist.

This ADR does not make the simulator engine (`mujoco` vs. `isaac`) a
registry-level or manifest-level concept — that remains `so101.factory
.make_so101`'s own `simulator=` kwarg, tracked as a gap in
`docs/ARCHITECTURE.md`'s extension-seam table pending follow-up work.
