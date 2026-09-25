# Migration record

This records the core-architecture-freeze restructuring: every moved module,
what replaced it, the phase it landed in, and the tests that guarded it. All
steps below are **done** — this is a historical record, not a plan to
execute. Unlisted modules did not move.

## Research extraction (Phases 1–3)

| Old path | New path | Guarded by |
| --- | --- | --- |
| `src/physai/robots/so101/expert.py` | `research/scripted_experts/so101_pick_place_expert.py` | `tests/research/scripted_experts/test_so101_pick_place_expert.py` |
| `src/physai/robots/so101/visual_servo.py` | `research/classical_control/so101_visual_servo.py` | `tests/research/classical_control/test_so101_visual_servo*.py` |
| `src/physai/robots/so101/policy.py` | Removed — `make_scripted_policy`/`make_visual_servo_policy` moved into the two files above and self-register with `robots.registry.register_robot_policy()` | `tests/unit/test_robot_registry.py`, `tests/unit/test_policy_registry.py` |
| `src/physai/policy/vla_adapter.py` (`VLAPolicy`, `ReplayPolicy`) | `src/physai/policy/replay.py` (stays core) | `tests/policy/test_replay.py` |
| `src/physai/policy/vla_adapter.py` (`LeRobotPolicy`) | `research/imitation_learning/vla_adapter.py`; self-registers `"lerobot"` with `policy.registry` | `tests/unit/test_policy_registry.py::test_lerobot_policy_registers_itself_when_its_research_module_is_imported` |
| `src/physai/policy/act_dataset.py` | `research/imitation_learning/act_dataset.py` | (exercised via `research/imitation_learning/train_act.py`) |
| `scripts/train_act.py` | `research/imitation_learning/train_act.py` | manual invocation (training script, no CI test) |
| `src/physai/planner/base.py` (`SortingPlanner`) | `research/vlm_planners/sorting_planner.py`; self-registers `"sorting_planner"` with `planner.registry` | `tests/unit/test_planner_registry.py` |
| `tests/unit/test_visual_servo.py` | `tests/research/classical_control/test_so101_visual_servo.py` | — |
| `tests/acceptance/so101/test_visual_servo_acceptance.py` | `tests/research/classical_control/test_so101_visual_servo_acceptance.py` | — |
| `tests/policy/test_scripted_policy.py` (expert tests) | `tests/research/scripted_experts/test_so101_pick_place_expert.py` | — |
| `tests/policy/test_scripted_policy.py` (`PlanRunner` test) | `tests/policy/test_plan_runner.py` (renamed, stays core) | — |

Scripts that use the built-in `scripted`/`visual_servo`/`lerobot` policies
(`run_sim.py`, `eval_policy.py`, `eval_randomization.py`,
`render_docs_media.py`, `collect_demos.py`) import the relevant `research/`
module explicitly to trigger self-registration, since core no longer imports
it for them.

## Registry consolidation (Phases 4–5)

| Change | Where | Guarded by |
| --- | --- | --- |
| `RobotDescriptor` + `register_embodiment()` replace one `register_*` call per factory kind | `src/physai/robots/registry.py` | `tests/unit/test_robot_registry.py::test_register_embodiment_wires_every_factory_in_one_call` |
| `planner/registry.py` added (mirrors `tasks/registry.py`) | `src/physai/planner/registry.py` | `tests/unit/test_planner_registry.py` |
| `register_adapter()`/`create_adapter()` added; `select_adapter()` becomes a thin wrapper | `src/physai/robots/adapters.py` | `tests/unit/test_adapter_registry.py` |

## Shared-world instance dispatch (Phase 7)

| Old | New | Guarded by |
| --- | --- | --- |
| `web/world_runtime.py`'s `SharedRobotInstance` (hardcoded `if robot_name == "so101"/"turtlebot4"`) | `robots/so101/shared.py`'s `SO101SharedInstance`, `robots/turtlebot/shared.py`'s `TurtleBot4SharedInstance`, dispatched via `robots.registry.create_shared_instance()` | `tests/unit/test_shared_world.py`, `tests/unit/test_host.py` |
| `sim/world.py`'s inline so101 camera-attach branch | `SharedWorld(..., shared_attach=...)` calling the registered `so101_shared_attach()` hook | `tests/unit/test_shared_world.py::test_shared_so101_instance_exposes_front_and_wrist_cameras` |

## Host unification (Phase 7)

| Old | New | Guarded by |
| --- | --- | --- |
| `src/physai/web/runtime.py` (`SimulationHost`) | `src/physai/web/host.py` (`Host.for_robot()`) | `tests/unit/test_host.py` |
| `src/physai/web/world_runtime.py` (`SharedWorldHost`) | `src/physai/web/host.py` (`Host.for_world()`) | `tests/unit/test_host.py` |
| `action_from_payload` (was in `web/runtime.py`) | `src/physai/web/actions.py` | `tests/unit/test_web_telemetry.py` |
| `scripts/run_sim.py`'s Tk viewer camera-panel list (introspected `env.model` directly) | Driven by `env.robot_spec.camera_frames` | `tests/unit/test_viewer_photo.py` |

End-to-end HTTP smoke tests (single-robot `--serve` with camera streaming,
and `--world configs/worlds/heterogeneous.yaml --serve`) were run manually
against real MuJoCo assets during this phase, in addition to the automated
suite.

## Session manifest (Phase 6)

`src/physai/config.py` became a package: `config/legacy.py` (unchanged
content, moved) and `config/manifest.py` (new). `config/__init__.py`
re-exports both, so every existing `physai.config.load_*` import is
unaffected. Guarded by `tests/unit/test_config.py` (legacy loaders) and
`tests/unit/test_manifest.py` (new schema).

## Import-boundary enforcement (Phase 9)

`import-linter` added as a dev-only dependency; contracts in
`pyproject.toml`'s `[tool.importlinter]`; gated in the normal test run via
`tests/boundaries/test_import_boundaries.py`.

## Documentation

`docs/ARCHITECTURE.md` was rewritten as the frozen reference (module
ownership, dependency-direction table, extension seams, frozen-vs-free list,
session manifest, unified host API, research boundary, known remaining
gaps). `docs/WEB_VIEWER_RUNBOOK.md` was trimmed to operational-only content;
its host-threading-model, control-lease, and API-surface sections moved into
`docs/ARCHITECTURE.md#host--client-api`. Eight ADRs record the underlying
decisions: see `docs/adr/`.

## Not done in this pass

Tracked in `docs/ARCHITECTURE.md`'s "Known remaining gaps" rather than here,
since they describe current-state gaps, not completed moves:

- Jog-resolver registry field on `RobotDescriptor` (resolver construction is
  still duplicated across `web/host.py` and `scripts/teleop_keyboard.py`).
- Routing the remaining utility scripts (`workspace_map.py`,
  `benchmark_ik.py`, `render_docs_media.py`, `teleop_keyboard.py`, and
  `run_sim.py`'s task/policy assembly) through
  `physai.runtime.create_runtime()` instead of hand-assembling `EnvConfig`/
  env objects directly.
- Wiring `physai.config.manifest.load_manifest()` into `run_sim.py` or
  `create_runtime()` — the schema exists and is tested, but no script
  consumes it yet.
