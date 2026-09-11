"""physai — robotics contracts and adapters for physical-AI research.

Layers, mirroring the architecture diagram:

    planner/   VLM  — natural language -> sub-goal waypoints (PoseStamped)
    policy/    VLA  — waypoint + images + joint state -> joint commands
    control/   IK / joint controller — Twist or PoseStamped -> joint targets
    sim/       MuJoCo — rigid body physics and contact
    bridge/    ROS2 transport, topic contracts, and adapters

Direct MuJoCo and ROS2-backed execution share the same application contracts;
ROS2 message conversion remains at the bridge boundary.
"""

__version__ = "0.0.1"
