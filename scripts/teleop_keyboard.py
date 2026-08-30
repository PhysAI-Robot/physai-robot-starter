"""Keyboard teleop in the MuJoCo viewer — Cartesian jogging via the Twist path.

Exercises the same `Twist -> TwistToJointResolver -> Action` chain that a VLA
policy will use, so if teleop feels right, the closed-loop path is wired right.

    python scripts/teleop_keyboard.py

Keys (press inside the viewer window):
    W / S   +x / -x        A / D   +y / -y        Q / E   +z / -z
    O / C   open / close gripper
    R       reset episode
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from physai.contracts import GripperCommand, Twist, Vector3
from physai.control import TwistToJointResolver
from physai.robots.so101 import EnvConfig, SO101Env

KEYMAP = {
    ord("W"): ("lin", 0, +1.0), ord("S"): ("lin", 0, -1.0),
    ord("A"): ("lin", 1, +1.0), ord("D"): ("lin", 1, -1.0),
    ord("Q"): ("lin", 2, +1.0), ord("E"): ("lin", 2, -1.0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.06, help="m/s per key press")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco.viewer

    env = SO101Env(EnvConfig(seed=args.seed, render=False, max_steps=10**9))
    obs = env.reset()
    resolver = TwistToJointResolver(env.kin, env.data, dt=env.control_dt)

    state = {"lin": np.zeros(3), "grip": 1.0, "reset": False}

    def on_key(keycode: int) -> None:
        if keycode in KEYMAP:
            _, axis, sign = KEYMAP[keycode]
            state["lin"][axis] = sign * args.speed
        elif keycode == ord("O"):
            state["grip"] = min(1.0, state["grip"] + 0.1)
        elif keycode == ord("C"):
            state["grip"] = max(0.0, state["grip"] - 0.1)
        elif keycode == ord("R"):
            state["reset"] = True

    print(__doc__)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            if state["reset"]:
                obs = env.reset()
                state.update(lin=np.zeros(3), grip=1.0, reset=False)

            twist = Twist(linear=Vector3.from_array(state["lin"]))
            action = resolver(twist, obs.joint_state,
                              GripperCommand(position=state["grip"]))
            obs, *_rest = env.step(action)
            # Key presses are momentary: decay the command so the arm stops.
            state["lin"] *= 0.6
            viewer.sync()

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
