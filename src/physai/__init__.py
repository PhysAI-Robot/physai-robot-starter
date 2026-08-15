"""physai — Phase 0 starter for SO-101 physical-AI research.

Layers, mirroring the architecture diagram:

    planner/   VLM  — natural language -> sub-goal waypoints (PoseStamped)
    policy/    VLA  — waypoint + images + joint state -> joint commands
    control/   IK / joint controller — Twist or PoseStamped -> joint targets
    sim/       MuJoCo — rigid body physics and contact
    bridge/    ROS2 topic contract for Phase 1

Phase 0 runs all of it in-process; Phase 1 splits the same interfaces across
ROS2 nodes.
"""

__version__ = "0.0.1"
