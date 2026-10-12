"""Load the plugin modules a session manifest names.

A plugin is any importable module (a user's own research folder, for example
`my_lab.plugin`) that registers robots, scenes, tasks, policies or planners
with the core registries when it is imported. Core never imports a plugin by
name itself; the manifest's `plugins:` list is the only way one gets loaded,
so every plugin a run depends on is recorded with the run.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable


def load_plugins(modules: Iterable[str]) -> None:
    """Import each module so it registers itself; a failing import is an error."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # Only the plugin itself (or a parent package) missing gets the
            # friendlier message; a dependency missing inside it re-raises as is.
            if exc.name is not None and f"{name}.".startswith(f"{exc.name}."):
                raise ModuleNotFoundError(
                    f"plugin {name!r} is not importable; put its folder on the "
                    "Python path (the repository root already is for scripts)",
                    name=exc.name,
                ) from exc
            raise


__all__ = ["load_plugins"]
