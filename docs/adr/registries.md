# Registry decisions

ADRs 6 and 7. The seam table is in
[ARCHITECTURE.md](../ARCHITECTURE.md#extension-seams).

## ADR 6: One call registers a whole robot

Status: accepted.

**Context.** Adding a robot needed three to six separate registration calls in
`robots/registry.py` (factory, scene defaults, env config, and optionally a ROS2
node, navigation baseline and policies).

**Decision.** A `RobotDescriptor` dataclass bundles every factory a robot owns
(env factory, kind, scene defaults, env config, and optional ROS2 node,
navigation, jog resolver, shared-world factories, supported simulators), and
`register_embodiment(name, descriptor)` registers it at once. Robot-owned
policies still register separately through `register_robot_policy()`, since
research modules register them independently and later.

**Consequences.** Adding a robot is one package, one descriptor and one call.
Lookup functions (`create_robot`, `robot_kind`, `scene_defaults`,
`create_ros2_node`, `navigate`, `create_robot_policy`, `available_robots`) kept
their signatures. The per-field `register_*` helpers that filled one descriptor
field later were unused and have been removed.

## ADR 7: Planner and backend registries

Status: accepted. The adapter names were changed by
[ADR 14](simulators.md#adr-14-simulator-package-layout-and-adapter-names):
`direct_mujoco`, `ros2_mujoco` and `ros2_hardware` are now `direct`, `ros2_sim`
and `ros2_real`.

**Context.** Robots, scenes, tasks and policies had register/create pairs, but
planners were constructed directly and backend selection was a string switch,
so two of the extension seams did not meet "one new file plus one registration
call".

**Decision.** `planner/registry.py` (`register_planner`, `create_planner`,
`available_planners`) mirrors `tasks/registry.py`; `robots/adapters.py` gains
`register_adapter` / `create_adapter`, replacing the switch.

**Consequences.** A new planner or backend is a registration call like every
other seam. `ScriptedPlanner` registers from core; `SortingPlanner` registers
itself from `research/vlm_planners/` on import, without core importing it.
