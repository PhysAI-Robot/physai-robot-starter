"""Print the Phase 1 ROS2 topic contract that Phase 0 code is written against.

    python scripts/show_ros2_contract.py
"""

import _bootstrap  # noqa: F401

from physai.bridge import describe

if __name__ == "__main__":
    print(describe())
