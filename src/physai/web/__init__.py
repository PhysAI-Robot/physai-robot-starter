"""Browser-facing runtime for headless MuJoCo sessions."""

from .telemetry import build_scene_manifest, build_state_snapshot

__all__ = ["build_scene_manifest", "build_state_snapshot"]
