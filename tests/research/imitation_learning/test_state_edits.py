import numpy as np
import pytest

from research.imitation_learning.state_edits import (
    HOLD_READING,
    SLEW_PER_STEP,
    GripperStateEditor,
)


def _state(gripper):
    return np.array([0.0, 0.1, 0.2, 0.3, 0.4, gripper], dtype=np.float32)


def test_lag_moves_toward_the_command_at_the_measured_rate():
    editor = GripperStateEditor("gripper_lag")
    editor(_state(1.745), None)  # the first reading is the real one
    out = [editor(_state(0.5), 0.88)[-1] for _ in range(3)]
    assert out[0] == pytest.approx(1.745 - SLEW_PER_STEP, abs=1e-4)
    assert out[2] == pytest.approx(1.745 - 3 * SLEW_PER_STEP, abs=1e-4)


def test_track_follows_the_command_but_not_below_the_hold_reading():
    editor = GripperStateEditor("gripper_track")
    assert editor(_state(1.745), 0.881)[-1] == pytest.approx(0.881)
    assert editor(_state(0.2), -0.059)[-1] == pytest.approx(HOLD_READING)


def test_other_joints_are_untouched_and_reset_forgets_the_history():
    editor = GripperStateEditor("gripper_lag")
    state = _state(1.745)
    assert (editor(state, 0.2)[:5] == state[:5]).all()
    editor.reset()
    assert editor(_state(1.0), 0.2)[-1] == pytest.approx(1.0 - SLEW_PER_STEP, abs=1e-4)


def test_an_unknown_state_edit_is_refused():
    with pytest.raises(ValueError, match="unknown state_edit"):
        GripperStateEditor("nope")
