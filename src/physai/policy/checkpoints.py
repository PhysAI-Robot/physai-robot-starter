"""Resolve a `--checkpoint` value to a local folder.

A value is either a path, used as given, or a Hugging Face reference
`hf:<owner>/<repo>/<subfolder>[@<revision>]`, which downloads only that subfolder of the repository into
`checkpoints/<owner>/<repo>/<subfolder>` at the repository root (ignored by git, like `assets/`) and returns
that folder. A revision pins a commit or tag; without one the repository's `main` is used, so a published
result should name one. `scripts/fetch_checkpoints.py` does the same download ahead of time. `huggingface_hub`
is imported only when a reference is used (the `training` extra installs it through lerobot). Weights are not
part of this repository; see docs/DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path

HF_PREFIX = "hf:"
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_hf_reference(spec: str) -> tuple[str, str, str | None]:
    """`hf:owner/repo/sub/folder@rev` -> (`owner/repo`, `sub/folder`, `rev`); revision is None if absent."""
    body, _, revision = spec.removeprefix(HF_PREFIX).partition("@")
    parts = [part for part in body.split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            f"{spec!r}: expected hf:<owner>/<repo>/<subfolder>[@<revision>], "
            "e.g. hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0"
        )
    return "/".join(parts[:2]), "/".join(parts[2:]), revision or None


def fetch_checkpoint(
    repo_id: str,
    subfolder: str,
    revision: str | None = None,
    root: Path | None = None,
) -> Path:
    """Download one subfolder of a Hugging Face repository into `<root>/<repo_id>/<subfolder>`.

    Files that are already there and unchanged are not downloaded again. `root` defaults to
    `checkpoints/` at the repository root.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "downloading a checkpoint needs huggingface-hub "
            "(uv sync --extra training installs it)"
        ) from exc
    repo_dir = (Path(root) if root is not None else CHECKPOINT_DIR) / repo_id
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(repo_dir),
    )
    folder = repo_dir / subfolder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{repo_id} has no folder {subfolder!r}"
            + (f" at revision {revision!r}" if revision else "")
        )
    return folder


def resolve_checkpoint(spec: str | Path, root: Path | None = None) -> Path:
    """The local checkpoint folder for a path or an `hf:` reference."""
    if not str(spec).startswith(HF_PREFIX):
        return Path(spec)
    repo_id, subfolder, revision = parse_hf_reference(str(spec))
    return fetch_checkpoint(repo_id, subfolder, revision, root)
