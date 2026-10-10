"""Run `eval_policy.py` on MuJoCo with the camera frames drawn by Newton instead.

    python research/imitation_learning/newton_eval.py --newton-python PATH_TO_NEWTON_PYTHON \
        -- --manifest configs/manifests/so101_randomized_pick_place.yaml --policy lerobot \
        --checkpoint outputs/act_200 --seed 1000 --episodes 100

MuJoCo keeps the physics, the state and the camera poses; `SO101Env.render_camera` sends
the state to `newton_server.py` (a separate Python with `newton` installed) and returns its
picture. Everything after `--` is `eval_policy.py`'s own command line, so any policy,
seed range or `--json` output works as for MuJoCo.

Research module: nothing in `src/physai` imports it; this patches the env at run time.
"""

from __future__ import annotations

import argparse
import atexit
import pickle
import struct
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "src", _ROOT, _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402

SERVER = Path(__file__).with_name("newton_server.py")
SCENE = _ROOT / "assets" / "so101" / "_newton_scene.xml"  # meshes resolve beside it


class NewtonService:
    def __init__(self, python: str, env) -> None:
        from physai.sim.mujoco import export_xml

        scene = env.cfg.scene
        export_xml(SCENE, scene)
        self._process = subprocess.Popen(
            [
                python,
                str(SERVER),
                str(SCENE),
                str(scene.camera_width),
                str(scene.camera_height),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        if self._process.stdout.readline().strip() != b"ready":
            raise RuntimeError("the Newton server did not start")
        atexit.register(self.close)

    def render(self, request: dict) -> np.ndarray:
        payload = pickle.dumps(request)
        self._process.stdin.write(struct.pack("<I", len(payload)) + payload)
        self._process.stdin.flush()
        (size,) = struct.unpack("<I", self._process.stdout.read(4))
        return pickle.loads(self._process.stdout.read(size))

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=30)


def install(python: str) -> None:
    import mujoco

    from physai.robots.so101.mujoco_env import SO101Env

    service: list[NewtonService] = []

    def render_camera(self, name: str) -> np.ndarray:
        if not service:
            service.append(NewtonService(python, self))
        camera = self.model.camera(name).id
        pad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_pad")
        return service[0].render(
            {
                "qpos": self.data.qpos[:13].copy(),
                "target": self.model.geom_pos[pad].copy(),
                "pos": self.data.cam_xpos[camera].copy(),
                "mat": self.data.cam_xmat[camera].reshape(3, 3).copy(),
                "fovy": float(self.model.cam_fovy[camera]),
            }
        )

    SO101Env.render_camera = render_camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--newton-python", required=True, help="Python with newton installed"
    )
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    install(args.newton_python)
    import eval_policy

    sys.argv = ["eval_policy.py", *rest]
    return eval_policy.main()


if __name__ == "__main__":
    raise SystemExit(main())
