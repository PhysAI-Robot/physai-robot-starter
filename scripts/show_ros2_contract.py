"""Print the ROS2 topic contract used by the supported robot adapters.

    python scripts/show_ros2_contract.py
"""

import _bootstrap  # noqa: F401

from physai.bridge import describe

if __name__ == "__main__":
    print(describe())
