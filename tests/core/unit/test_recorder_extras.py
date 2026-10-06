import json
from types import SimpleNamespace

import numpy as np

from physai.contracts import Action, JointState, Observation
from physai.data import EpisodeRecorder, load_episode
from physai.data.extras import collect_extras


def _obs():
    return Observation(
        joint_state=JointState(name=("a",), position=np.zeros(1), velocity=np.ones(1))
    )


def test_collect_extras_gathers_robot_policy_and_info():
    robot = SimpleNamespace(gripper_contact_force=lambda: 1.5)
    got = collect_extras(robot, None, _obs(), {"ok": True, "skip": [1]})
    assert set(got) >= {"joint_velocity", "gripper_force_n", "info.ok"}
    assert "info.skip" not in got
    # dataclass metrics are expanded, a missing detection becomes NaN
    import dataclasses

    @dataclasses.dataclass
    class M:
        phase: str = "LIFT"
        feature_px: tuple | None = None

    got = collect_extras(None, SimpleNamespace(metrics=M()), _obs(), None)
    assert got["policy.phase"] == "LIFT" and np.isnan(got["policy.feature_px"])


def test_sparse_extras_are_padded_and_listed_in_meta(tmp_path):
    rec = EpisodeRecorder(tmp_path)
    rec.start_episode()
    action = Action(joint_position=np.zeros(1))
    rec.record(_obs(), action, extras={"force": 2.0, "tag": "x"})
    rec.record(_obs(), action, extras={})
    rec.record(_obs(), action, extras={"force": 3.0})
    path = rec.end_episode(True, path=tmp_path / "flat.npz")
    meta = rec.write_meta(tmp_path / "flat.json")
    data = load_episode(path)
    assert data["extras.force"].shape == (3,) and np.isnan(data["extras.force"][1])
    assert list(data["extras.tag"]) == ["x", "", ""]
    features = json.loads(meta.read_text())["features"]
    assert features["extras.force"]["dtype"] == "float64"
