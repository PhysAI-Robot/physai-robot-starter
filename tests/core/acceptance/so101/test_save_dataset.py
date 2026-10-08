"""`eval_policy.py --save-dataset` writes the same dataset layout `collect_demos.py` does."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import requires_assets

pytestmark = [pytest.mark.acceptance, requires_assets]

ROOT = Path(__file__).resolve().parents[4]


def test_eval_policy_saves_a_training_dataset(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_policy.py"),
            "--policy",
            "scripted",
            "--episodes",
            "1",
            "--save-dataset",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_episodes"] == 1
    assert meta["task_name"] == "single_cube_place"
    assert set(meta["camera_config"]) == {"front", "wrist"}
    assert (tmp_path / meta["episodes"][0]["file"]).exists()
