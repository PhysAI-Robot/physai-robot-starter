# 1. Merge SimulationHost and SharedWorldHost into one host

## Status

Accepted

## Context

`SimulationHost` (single-robot) and `SharedWorldHost` (multi-robot) each
independently implemented a physics thread, camera worker, control-lease
arbitration, and publish loop, with no shared base class.
`src/physai/web/app.py` branched on `isinstance(host, SharedWorldHost)`
throughout instead of calling one polymorphic API. Adding a host feature
meant changing it twice, and single-robot was not actually "a world with one
instance" — it was a structurally different class.

## Decision

Replace both with one `Host` class (`src/physai/web/host.py`) where a
single-robot session is simply `instances = {default_id: <the one
instance>}`. Every method (`scene`, `latest_state`, `list_robots`,
`camera_jpeg`, `submit`, `reset`, `set_paused`, `release_control`) is
instance-keyed; callers with one robot may omit the id. Reset and pause stay
world-atomic per the existing multi-robot contract.

## Consequences

`web/app.py` no longer branches on host type. `web/runtime.py` and
`web/world_runtime.py` are removed once `host.py` reaches behavioral parity,
guarded by tests that exercise both the n=1 and n>1 cases against the same
class.
