import argparse
import sys
import types
from pathlib import Path

import pytest

from physai.policy.checkpoints import parse_hf_reference, resolve_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import _common_args  # noqa: E402


def _fake_hub(monkeypatch, root: Path) -> list[dict]:
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(root)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    return calls


def test_a_path_is_used_as_given():
    assert resolve_checkpoint("outputs/act") == Path("outputs/act")
    assert resolve_checkpoint(Path("outputs/act")) == Path("outputs/act")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("hf:owner/repo/model", ("owner/repo", "model", None)),
        ("hf:owner/repo/model@abc123", ("owner/repo", "model", "abc123")),
        ("hf:owner/repo/group/model@v1", ("owner/repo", "group/model", "v1")),
    ],
)
def test_a_reference_splits_into_repo_subfolder_and_revision(spec, expected):
    assert parse_hf_reference(spec) == expected


@pytest.mark.parametrize("spec", ["hf:owner/repo", "hf:owner", "hf:"])
def test_a_reference_without_a_subfolder_is_refused(spec):
    with pytest.raises(ValueError, match="expected hf:"):
        parse_hf_reference(spec)


def test_a_reference_downloads_only_its_subfolder_and_returns_the_local_folder(
    tmp_path, monkeypatch
):
    (tmp_path / "model").mkdir()
    calls = _fake_hub(monkeypatch, tmp_path)
    folder = resolve_checkpoint("hf:owner/repo/model@abc123")
    assert folder == tmp_path / "model"
    assert calls == [
        {"repo_id": "owner/repo", "revision": "abc123", "allow_patterns": ["model/*"]}
    ]


def test_a_missing_subfolder_is_named_in_the_error(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="no folder 'model'"):
        resolve_checkpoint("hf:owner/repo/model")


def test_a_missing_huggingface_hub_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="huggingface-hub"):
        resolve_checkpoint("hf:owner/repo/model")


def test_the_checkpoint_argument_resolves_a_reference(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    _fake_hub(monkeypatch, tmp_path)
    args = argparse.Namespace(policy_arg=[], checkpoint="hf:owner/repo/model")
    assert _common_args.policy_kwargs(args) == {"checkpoint": tmp_path / "model"}
