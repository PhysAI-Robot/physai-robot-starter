import numpy as np
import pytest

from conftest import requires_assets

pytestmark = requires_assets

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _render_env():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(EnvConfig(seed=0, render=True, max_steps=200))
    return TaskRuntime(robot, create_task("pick_place"))


@requires_assets
@requires_torch
def test_act_dataset_padding_and_shapes(tmp_path):
    """The last few timesteps of an episode need padded action chunks — this
    is the part of act_dataset.py most likely to silently produce garbage
    (wrong pad value, wrong mask) without an explicit test."""
    from physai.data import EpisodeRecorder
    from physai.policy import ScriptedPickPlace
    from physai.policy.act_dataset import ACTEpisodeDataset

    env = _render_env()
    try:
        rec = EpisodeRecorder(tmp_path, fps=env.cfg.control_hz)
        obs = env.reset(seed=0)
        policy = ScriptedPickPlace(env.kin, env)
        policy.reset(obs)
        rec.start_episode()
        n_steps = 40
        for _ in range(n_steps):
            action = policy.act(obs)
            prev = obs
            obs, reward, term, trunc, _ = env.step(action)
            rec.record(prev, action, reward=reward, done=term or trunc,
                       phase=policy.phase.name,
                       gripper_joint=env.gripper_to_joint(action.gripper.clipped()))
        rec.end_episode(success=False)
        rec.write_meta()
    finally:
        env.close()

    chunk_size = 10
    ds = ACTEpisodeDataset(tmp_path, chunk_size=chunk_size, image_size=32, task="test task")
    assert len(ds) == n_steps

    # A sample near the start: full chunk, no padding.
    early = ds[0]
    assert early["action"].shape == (chunk_size, 6)
    assert not early["action_is_pad"].any()

    # The last sample: position 0 is still the real final action (an episode
    # of length T has a valid sample at t=T-1); every position after that is
    # padded, and repeats the final action (not zeros — zero would be a real,
    # very wrong, joint target).
    last = ds[n_steps - 1]
    assert not last["action_is_pad"][0]
    assert last["action_is_pad"][1:].all()
    last_real_action = torch.from_numpy(
        np.load(tmp_path / "episode_00000.npz")["action"][-1]
    ).float()
    torch.testing.assert_close(last["action"][0], last_real_action)
    torch.testing.assert_close(last["action"][-1], last["action"][0])

    for cam in ("front", "wrist"):
        img = early[f"observation.images.{cam}"]
        assert img.shape == (3, 32, 32)
        assert img.min() >= 0.0 and img.max() <= 1.0


@requires_assets
@requires_torch
def test_act_dataset_stats_shapes(tmp_path):
    from physai.data import EpisodeRecorder
    from physai.policy import ScriptedPickPlace
    from physai.policy.act_dataset import ACTEpisodeDataset

    env = _render_env()
    try:
        rec = EpisodeRecorder(tmp_path, fps=env.cfg.control_hz)
        for seed in (0, 1):
            obs = env.reset(seed=seed)
            policy = ScriptedPickPlace(env.kin, env)
            policy.reset(obs)
            rec.start_episode()
            for _ in range(20):
                action = policy.act(obs)
                prev = obs
                obs, reward, term, trunc, _ = env.step(action)
                rec.record(prev, action, reward=reward, done=term or trunc,
                           phase=policy.phase.name,
                           gripper_joint=env.gripper_to_joint(action.gripper.clipped()))
            rec.end_episode(success=False)
        rec.write_meta()
    finally:
        env.close()

    ds = ACTEpisodeDataset(tmp_path, chunk_size=5, image_size=32, task="test")
    stats = ds.compute_stats()

    assert stats.per_key["action"]["mean"].__len__() == 6
    assert stats.per_key["observation.state"]["std"].__len__() == 6
    for cam in ("front", "wrist"):
        s = stats.per_key[f"observation.images.{cam}"]
        assert len(s["mean"]) == 3 and len(s["std"]) == 3
        assert all(v > 0 for v in s["std"]), "zero std would divide-by-zero at normalization"

    out = tmp_path / "stats.json"
    stats.to_json(out)
    reloaded = type(stats).from_json(out)
    assert reloaded.per_key == stats.per_key
