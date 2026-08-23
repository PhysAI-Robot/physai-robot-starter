"""Local Hugging Face model storage and path resolution."""

from __future__ import annotations

from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"


def model_dir(model_name: str) -> Path:
    """Return the local directory used for a downloaded model name."""
    return MODEL_ROOT / model_name


def resolve_local_model(model: str, *, model_name: str | None = None) -> Path:
    """Resolve a model path and reject unresolved Hugging Face IDs."""
    path = Path(model).expanduser()
    if path.exists():
        return path
    local = model_dir(model_name or model.rsplit("/", 1)[-1])
    if local.exists():
        return local
    raise FileNotFoundError(
        f"local model not found: {local}. "
        f"Run `python scripts/download_models.py --repo {model}` first."
    )
