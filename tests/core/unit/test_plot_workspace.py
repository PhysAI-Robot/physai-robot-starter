import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))


def test_plot_workspace_writes_an_image_from_evaluation_json(tmp_path):
    from plot_workspace import plot

    rows = [
        {"success": True, "cube_start": [0.2, 0.05], "target_pos": [0.18, -0.1]},
        {"success": False, "cube_start": [0.17, -0.02], "target_pos": [0.22, 0.1]},
    ]
    results = tmp_path / "eval.json"
    results.write_text(json.dumps({"results": rows}), encoding="utf-8")
    plot([results], ["demo"], tmp_path / "out.png")
    assert (tmp_path / "out.png").stat().st_size > 1000
