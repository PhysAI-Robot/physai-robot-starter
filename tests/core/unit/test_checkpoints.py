import argparse
import sys
import types
from pathlib import Path

import pytest

from physai.policy import checkpoints
from physai.policy.checkpoints import parse_hf_reference, resolve_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import _common_args  # noqa: E402


def _fake_hub(monkeypatch, make_folder: str | None = None) -> list[dict]:
    """A `huggingface_hub` whose download writes `make_folder` under local_dir (or nothing)."""
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if make_folder:
            (Path(kwargs["local_dir"]) / make_folder).mkdir(parents=True)
        return kwargs["local_dir"]

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


def test_a_reference_downloads_only_its_subfolder_into_the_checkpoint_folder(
    tmp_path, monkeypatch
):
    calls = _fake_hub(monkeypatch, make_folder="model")
    folder = resolve_checkpoint("hf:owner/repo/model@abc123", root=tmp_path)
    assert folder == tmp_path / "owner" / "repo" / "model"
    assert calls == [
        {
            "repo_id": "owner/repo",
            "revision": "abc123",
            "allow_patterns": ["model/*"],
            "local_dir": str(tmp_path / "owner" / "repo"),
        }
    ]


def test_the_default_download_folder_is_checkpoints_at_the_repository_root():
    assert checkpoints.CHECKPOINT_DIR == checkpoints.REPO_ROOT / "checkpoints"
    assert (checkpoints.REPO_ROOT / "pyproject.toml").exists()


def test_a_missing_subfolder_is_named_in_the_error(tmp_path, monkeypatch):
    _fake_hub(monkeypatch)
    with pytest.raises(FileNotFoundError, match="no folder 'model'"):
        resolve_checkpoint("hf:owner/repo/model", root=tmp_path)


def test_a_missing_huggingface_hub_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(RuntimeError, match="huggingface-hub"):
        resolve_checkpoint("hf:owner/repo/model")


def test_the_checkpoint_argument_resolves_a_reference(tmp_path, monkeypatch):
    _fake_hub(monkeypatch, make_folder="model")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_DIR", tmp_path)
    args = argparse.Namespace(policy_arg=[], checkpoint="hf:owner/repo/model")
    assert _common_args.policy_kwargs(args) == {
        "checkpoint": tmp_path / "owner" / "repo" / "model"
    }


def test_the_fetch_script_downloads_each_reference_and_prints_its_folder(
    tmp_path, monkeypatch, capsys
):
    import fetch_checkpoints

    _fake_hub(monkeypatch, make_folder="model")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fetch_checkpoints.py", "hf:owner/repo/model", "--out", str(tmp_path)],
    )
    assert fetch_checkpoints.main() == 0
    assert str(tmp_path / "owner" / "repo" / "model") in capsys.readouterr().out
