from types import SimpleNamespace

import numpy as np
import pytest

from physai.contracts import Action, JointState, Observation
from physai.control import SafetyViolation
from physai.runtime import EpisodeObserver, RenderGlitch, run_episode


def _obs():
    return Observation(joint_state=JointState(name=("a",), position=np.zeros(1)))


class FakeRuntime:
    """Steps succeed until `refuse_at`, which the safety gate refuses."""

    def __init__(self, *, done_after=None, refuse_at=None, success_at=None, limit=5):
        self.steps = 0
        self.policy = SimpleNamespace(act=lambda obs: Action(), done=False)
        self._done_after, self._refuse_at = done_after, refuse_at
        self._success_at = success_at
        self.robot = SimpleNamespace(cfg=SimpleNamespace(max_steps=limit))
        self.resets = []

    def reset(self, seed=None):
        self.resets.append(seed)
        self.steps = 0
        return _obs()

    def step(self, action):
        if self.steps == self._refuse_at:
            raise SafetyViolation("too fast")
        self.steps += 1
        if self._done_after == self.steps:
            self.policy.done = True
        info = {"success": self.steps == self._success_at}
        return _obs(), 1.0, False, self.steps >= self.robot.cfg.max_steps, info


class Spy(EpisodeObserver):
    def __init__(self):
        self.calls = []

    def on_reset(self, observation):
        self.calls.append("reset")

    def before_step(self, observation, action, info):
        self.calls.append("before")

    def after_step(self, observation, action, next_observation, reward, done, info):
        self.calls.append("after")


def test_runs_to_the_robot_limit_and_reports_truncation():
    runtime = FakeRuntime(limit=4)
    result = run_episode(runtime, seed=7)
    assert (result.steps, result.reward, result.truncated) == (4, 4.0, True)
    assert runtime.resets == [7]


def test_policy_done_ends_the_episode_early():
    result = run_episode(FakeRuntime(done_after=2, limit=10), seed=0)
    assert result.steps == 2 and not result.truncated


def test_a_refused_action_is_recorded_not_raised():
    result = run_episode(FakeRuntime(refuse_at=2), seed=0)
    assert result.violation == "too fast"
    assert result.steps == 2 and not result.success


def test_success_comes_from_the_last_step_info():
    assert run_episode(FakeRuntime(success_at=5, limit=5), seed=0).success


def test_observers_see_reset_then_each_step():
    spy = Spy()
    run_episode(FakeRuntime(limit=2), seed=0, observers=(spy,))
    assert spy.calls == ["reset", "before", "after", "before", "after"]


def test_isaac_render_glitch_is_raised_before_any_step():
    runtime = FakeRuntime()
    runtime.robot.robot_is_rendered = lambda: False
    with pytest.raises(RenderGlitch):
        run_episode(runtime, seed=3)
