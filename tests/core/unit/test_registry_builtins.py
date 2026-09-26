"""An extension may register before a registry's first lookup.

Each registry lazily loads its built-ins; registering something else first
must not make them disappear (a research planner registers itself when its
module is imported, and a new backend is meant to be additive).
"""

import importlib

import pytest

REGISTRIES = [
    (
        "physai.planner.registry",
        "register_planner",
        "available_planners",
        "scripted_planner",
    ),
    ("physai.tasks.registry", "register_task", "available_tasks", "pick_place"),
    (
        "physai.robots.adapters",
        "register_adapter",
        "available_adapters",
        "direct_mujoco",
    ),
]


def test_builtins_survive_an_extension_registering_first(monkeypatch):
    for module_name, register, available, builtin in REGISTRIES:
        module = importlib.import_module(module_name)
        table = "_ADAPTERS" if module_name.endswith("adapters") else "_FACTORIES"
        monkeypatch.setattr(module, table, {})
        monkeypatch.setattr(module, "_BUILTINS_LOADED", False)

        getattr(module, register)("_extension_registered_first", lambda **_: None)

        assert builtin in getattr(module, available)(), module_name
        assert "_extension_registered_first" in getattr(module, available)()
