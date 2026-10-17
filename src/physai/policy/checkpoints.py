"""Resolve a `--checkpoint` value to a local folder.

A value is either a path, used as given, or a Hugging Face reference
`hf:<owner>/<repo>/<subfolder>[@<revision>]`, which downloads only that subfolder of the repository
into the Hugging Face cache (never into this repository) and returns the local folder. A revision pins
a commit or tag; without one the repository's `main` is used, so a published result should name one.
`huggingface_hub` is imported only when a reference is used (the `training` extra installs it through
lerobot). Weights are not part of this repository; see docs/DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path

HF_PREFIX = "hf:"


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


def resolve_checkpoint(spec: str | Path) -> Path:
    """The local checkpoint folder for a path or an `hf:` reference."""
    if not str(spec).startswith(HF_PREFIX):
        return Path(spec)
    repo_id, subfolder, revision = parse_hf_reference(str(spec))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "downloading a checkpoint needs huggingface-hub "
            "(uv sync --extra training installs it)"
        ) from exc
    root = Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=[f"{subfolder}/*"]
        )
    )
    folder = root / subfolder
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{repo_id} has no folder {subfolder!r}"
            + (f" at revision {revision!r}" if revision else "")
        )
    return folder
