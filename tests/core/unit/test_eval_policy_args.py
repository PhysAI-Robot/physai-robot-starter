import argparse
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _common_args  # noqa: E402
import eval_policy  # noqa: E402


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("squeeze_grip=False", ("squeeze_grip", False)),
        ("grasp_offset_xy=(0.0, -0.006)", ("grasp_offset_xy", (0.0, -0.006))),
        ("kp=2", ("kp", 2)),
        ("camera=wrist", ("camera", "wrist")),
    ],
)
def test_policy_arg_reads_python_literals_and_keeps_bare_words(text, expected):
    assert _common_args.parse_policy_arg(text) == expected


@pytest.mark.parametrize("text", ["no_equals", "=1"])
def test_policy_arg_rejects_a_missing_key_or_separator(text):
    with pytest.raises(argparse.ArgumentTypeError):
        _common_args.parse_policy_arg(text)


def test_seed_list_parses_distinct_integers():
    assert eval_policy._parse_seed_list("5,13, 28") == [5, 13, 28]


@pytest.mark.parametrize("text", ["5,x", "5,5"])
def test_seed_list_rejects_bad_input(text):
    with pytest.raises(argparse.ArgumentTypeError):
        eval_policy._parse_seed_list(text)


def test_policy_kwargs_merge_policy_args_and_require_a_checkpoint_for_lerobot():
    args = argparse.Namespace(policy_arg=[("kp", 2)], checkpoint=None)
    assert _common_args.policy_kwargs(args, "visual_servo") == {"kp": 2}
    with pytest.raises(ValueError, match="--checkpoint"):
        _common_args.policy_kwargs(args, "lerobot")
    args.checkpoint = Path("ckpt")
    assert _common_args.policy_kwargs(args, "lerobot") == {
        "kp": 2,
        "checkpoint": Path("ckpt"),
    }
