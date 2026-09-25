# 5. Keep TurtleBot4 as the second embodiment

## Status

Accepted

## Context

SO-101 is the current development focus. TurtleBot4 is a different
embodiment kind (mobile base, twist actions, no arm) already registered and
working. Removing it would shrink the multi-embodiment surface the
architecture is meant to prove; keeping it under active development would
compete with SO-101 for attention it does not need.

## Decision

TurtleBot4 stays in the repository as the proof that the `RobotSpec`
capability abstraction generalizes beyond a single arm. It is explicitly not
a development focus: its existing navigation baseline
(`robots/turtlebot/navigation.py`) stays in core as its only registered
capability, and no new TurtleBot4-specific research work is planned.

## Consequences

Existing TurtleBot4 tests and the ROS2/Nav2 acceptance path keep passing
untouched. Reviewers should not read TurtleBot4's thin feature set as
neglect — it is intentional scope.
