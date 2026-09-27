# 13. Isaac Sim as an optional backend

## Status

Accepted. The package layout and adapter names below were superseded by
[ADR 14](0014-simulator-package-layout.md): `physai.isaac` moved to
`physai.sim.isaac`, and `DirectMuJoCoAdapter` was renamed `DirectAdapter`.
[ADR 15](0015-simulator-engine-selection.md) superseded the "no manifest
schema change" and "no new registry seam" parts below: `simulator` is now a
top-level manifest field and a `RobotDescriptor.simulators` registry check,
not only a `config:`-routed factory kwarg. The rest of the decision (install
recipe, no `pyproject.toml` extra, `SO101IsaacEnv`'s narrower scope) still
holds.

## Context

The project's stated goal (`ROADMAP.md` Phase 4) includes cross-simulator
portability — evaluating a policy tuned in MuJoCo (`visual_servo`, ACT)
against a second simulator to measure the sim-to-sim gap, the same way
`ROADMAP.md`'s 2C compares backends. Isaac Sim was the target, contingent
on Python 3.12 support and GPU availability (this project pins
`requires-python = ">=3.12,<3.13"`).

A spike against Isaac Sim 6.1.0.0 on an RTX 3060 (see the plan this ADR
accompanies for the full log) confirmed Python 3.12 support and validated
an in-process integration end to end: URDF import, physics-variant
selection, actuator gain/limit mapping, closed-loop joint control, and
camera rendering all verified against real Isaac Sim, not just written.

## Decision

Add `physai.isaac` (SimulationApp lifecycle, `RobotDescription` → USD
application, generic world extras) and `robots.so101.isaac_env
.SO101IsaacEnv`, mirroring `physai.sim`'s role for MuJoCo. Only these two
locations may import `isaacsim`/`omni`/`pxr`; `pyproject.toml`'s
import-linter contract enforces it, and every such import is function-local
(Isaac Sim's own modules are not import-safe before a `SimulationApp`
exists), so importing `physai.isaac` or `physai.robots.so101.isaac_env`
never requires Isaac Sim to be installed — only constructing a
`SO101IsaacEnv` does.

No new registry seam was added. `DirectMuJoCoAdapter` only wraps a generic
`RobotPort` (observe/reset/step/send_action/close plus a safety gate) and
has no MuJoCo-specific behavior of its own, so it wraps an Isaac-backed
port exactly as it wraps a MuJoCo one; there is no `direct_isaac` adapter.
`so101.factory.make_so101()` gained a `simulator: "mujoco" | "isaac"`
parameter that picks which port it builds, defaulting to `"mujoco"`.

No manifest schema change was needed either. A robot instance's `config:`
mapping already reaches `create_robot(**kwargs)` unfiltered
(`runtime.session._robot_fields`), so `config: {simulator: isaac}` in an
existing manifest already routes to `make_so101(simulator="isaac", ...)`
today, verified by a test. `schema_version` is unchanged.

Isaac Sim is deliberately **not** a `pyproject.toml` extra: `isaacsim
==6.1.0.0` pins `numpy==2.3.1`, which directly conflicts with this
project's own `numpy>=2.0,<2.3` — a conflict against a base dependency
present in every environment, which `[tool.uv] conflicts` (for mutually
exclusive extras) cannot express or fix. It also needs an NVIDIA package
index and a pre-release allowance this project does not enable globally.
Install it into a separate virtual environment instead (never this
project's own `.venv`), with `PYTHONPATH` pointed at this repo's `src/`;
see `pyproject.toml`'s comment and `README.md` for the exact recipe.

`SO101IsaacEnv` is deliberately narrower than `SO101Env` for now: no
Cartesian jog, IK, or task/scene objects. `ArmKinematics`
(`robots/so101/kinematics.py`) is still a MuJoCo-specific kinematic model
(changing its API to take joint positions instead of `MjData`, so an Isaac
env could reuse it, is tracked separately — deferred deliberately: it
touches 25+ call sites with no second consumer yet to validate the new
shape against).

## Consequences

A robot-only Isaac-backed `RobotPort` exists and is verified end to end
(URDF import, joint control tracking a step to within measured tolerance,
camera rendering), but nothing that needs task/scene composition
(`TaskRuntime`, a graspable object, `ArmKinematics`-based IK) works with it
yet — that is follow-up work, not a limitation this ADR resolves. Isaac's
own physics-variant mechanism (a USD variant set: `mujoco` | `physx` |
`physics` | `none` on the imported robot, backed by Isaac Sim 6.1's own
`mujoco-usd-converter`) is a more direct sim-to-sim bridge than assumed
before the spike; `physai.isaac.description.import_robot()` defaults to
the `physx` variant, but the `mujoco` variant remains available for a
future comparison that wants Isaac to run the MuJoCo solver itself rather
than PhysX.
