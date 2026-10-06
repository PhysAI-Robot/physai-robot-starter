# Architecture decision records

Decisions behind the frozen design, grouped by topic. Each decision keeps its
number (`## ADR N`), so a reference such as "ADR 4" resolves to the file below. A
later ADR may supersede part of an earlier one; the entry says so.

| ADR | Decision | File |
| --- | --- | --- |
| 1 | One `Host` for single-robot and shared-world sessions | [web-host.md](web-host.md) |
| 4 | `--viewer` is MuJoCo's own viewer, frozen in scope | [web-host.md](web-host.md) |
| 9 | Record and replay web sessions through the existing data path | [web-host.md](web-host.md) |
| 2 | One session manifest schema | [config-and-sessions.md](config-and-sessions.md) |
| 10 | The manifest becomes the run description | [config-and-sessions.md](config-and-sessions.md) |
| 6 | One call registers a whole robot | [registries.md](registries.md) |
| 7 | Planner and backend registries | [registries.md](registries.md) |
| 11 | Sim-neutral robot description | [contracts-and-descriptions.md](contracts-and-descriptions.md) |
| 12 | `ImageFrame` carries camera intrinsics and extrinsics | [contracts-and-descriptions.md](contracts-and-descriptions.md) |
| 17 | Sim-neutral workspace description | [contracts-and-descriptions.md](contracts-and-descriptions.md) |
| 13 | Isaac Sim as an optional backend (partly superseded by 14-16) | [simulators.md](simulators.md) |
| 14 | Simulator package layout and adapter names | [simulators.md](simulators.md) |
| 15 | Simulator engine selection | [simulators.md](simulators.md) |
| 16 | `isaacsim` as a project extra (replaces 13's install recipe) | [simulators.md](simulators.md) |
| 18 | One environment for Isaac Sim and LeRobot (supersedes 16's `vla` conflict) | [simulators.md](simulators.md) |
| 19 | One manifest runs on both simulators (supersedes 15's robot-only manifest) | [simulators.md](simulators.md) |
| 20 | `run_sim` and `eval_policy` share one session and one rollout (narrows 10) | [config-and-sessions.md](config-and-sessions.md) |
| 21 | One flag name per concept across the scripts (renames flags in 20) | [config-and-sessions.md](config-and-sessions.md) |
| 3 | Research code lives outside the core package | [repo-scope.md](repo-scope.md) |
| 5 | TurtleBot4 stays as the second embodiment | [repo-scope.md](repo-scope.md) |
| 8 | Enforce dependency direction with import-linter | [repo-scope.md](repo-scope.md) |

A new decision goes into the file of its topic with the next free number, or into a
new topic file if none fits, and gets a row here.
