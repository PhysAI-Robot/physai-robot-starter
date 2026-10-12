import sys

import pytest
import yaml


def test_load_plugins_imports_each_module_and_fails_loudly_on_a_missing_one(
    tmp_path, monkeypatch
):
    from physai.plugins import load_plugins

    (tmp_path / "physai_test_plugin.py").write_text("LOADED = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    load_plugins(["physai_test_plugin"])
    assert sys.modules["physai_test_plugin"].LOADED

    with pytest.raises(ModuleNotFoundError, match="physai_no_such_plugin"):
        load_plugins(["physai_no_such_plugin"])


def test_a_manifest_loads_its_plugins_before_it_validates_names(tmp_path, monkeypatch):
    from physai.config import load_manifest

    (tmp_path / "physai_policy_plugin.py").write_text(
        "from physai.policy import register_policy\n"
        "from physai.policy.base import ConstantPolicy\n"
        "register_policy('physai_plugin_hold', lambda **_: ConstantPolicy())\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    data = {
        "schema_version": 1,
        "plugins": ["physai_policy_plugin"],
        "robots": [{"id": "a", "robot": "so101", "policy": "physai_plugin_hold"}],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.plugins == ("physai_policy_plugin",)
    assert manifest.policy_for(manifest.robots[0]) == "physai_plugin_hold"

    data["robots"][0]["policy"] = "physai_unregistered_policy"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="plugins"):
        load_manifest(path)
