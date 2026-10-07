from types import SimpleNamespace

import pytest

pytest.importorskip("lerobot")


def _policy(chunk_size=30):
    return SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            temporal_ensemble_coeff=None,
        )
    )


def test_n_action_steps_shortens_the_open_loop_and_is_bounded():
    from research.imitation_learning.vla_adapter import _set_execution

    policy = _policy()
    _set_execution(policy, n_action_steps=10, temporal_ensemble_coeff=None)
    assert policy.config.n_action_steps == 10
    with pytest.raises(ValueError, match="1..30"):
        _set_execution(policy, n_action_steps=31, temporal_ensemble_coeff=None)


def test_temporal_ensembling_predicts_every_step():
    from research.imitation_learning.vla_adapter import _set_execution

    policy = _policy()
    _set_execution(policy, n_action_steps=None, temporal_ensemble_coeff=0.01)
    assert policy.config.n_action_steps == 1
    assert policy.temporal_ensembler is not None
    with pytest.raises(ValueError, match="n_action_steps=1"):
        _set_execution(_policy(), n_action_steps=10, temporal_ensemble_coeff=0.01)
