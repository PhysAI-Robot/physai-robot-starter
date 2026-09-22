# 4. Freeze the Tk viewer's scope

## Status

Accepted

## Context

There are two client UIs: a Tk desktop viewer (`--viewer`) and a FastAPI +
Three.js web viewer (`--serve`). If both grow features, every new capability
has to be built and maintained twice.

## Decision

The Tk viewer's scope is frozen at: simulation render, live camera panels,
and basic status. It never grows new features — all new UI functionality
(jog controls, instance selection, telemetry, overlays) goes into the web
viewer only. The Tk viewer's *implementation* still has to become a thin
client of the unified host API (it currently reaches into MuJoCo directly),
but its *scope* does not expand.

## Consequences

Reviewers can reject any PR that adds a feature to the Tk viewer on scope
grounds alone. The web viewer remains the one place sophistication
accumulates.
