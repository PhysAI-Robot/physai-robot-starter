"""The scripts that drive a robot take any robot's manifest, not one robot's module."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest
from tests.conftest import requires_assets, requires_viperx300s_assets

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
MANIFESTS = [
    pytest.param(
        "configs/manifests/so101_single_cube_fixed_place.yaml", 5, marks=requires_assets
    ),
    pytest.param(
        "configs/manifests/viperx300s_single_cube_fixed_place.yaml",
        6,
        marks=requires_viperx300s_assets,
    ),
]


def _script(monkeypatch, name):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return __import__(name)


@pytest.mark.parametrize(("manifest", "joints"), MANIFESTS)
def test_benchmark_ik_runs_on_the_manifests_arm(monkeypatch, manifest, joints):
    benchmark_ik = _script(monkeypatch, "benchmark_ik")
    report = benchmark_ik.benchmark(
        Namespace(manifest=Path(manifest), targets=3, seed=0)
    )
    assert report["targets"] == 3
    assert report["jacobian_shape"] == [6, joints]
    assert report["success_count"] == 3


@pytest.mark.parametrize(("manifest", "joints"), MANIFESTS)
def test_teleop_keyboard_jogs_the_manifests_arm(monkeypatch, manifest, joints):
    import mujoco.viewer

    teleop = _script(monkeypatch, "teleop_keyboard")

    class FakeViewer:
        frames = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def is_running(self):
            FakeViewer.frames += 1
            return FakeViewer.frames <= 3

        def sync(self):
            pass

    monkeypatch.setattr(
        mujoco.viewer, "launch_passive", lambda *args, **kwargs: FakeViewer()
    )
    monkeypatch.setattr(sys, "argv", ["teleop_keyboard.py", "--manifest", manifest])
    assert teleop.main() == 0
    assert FakeViewer.frames == 4


@pytest.mark.parametrize(("manifest", "joints"), MANIFESTS)
def test_plan_task_plans_on_the_manifests_scene(monkeypatch, capsys, manifest, joints):
    plan_task = _script(monkeypatch, "plan_task")
    monkeypatch.setattr(
        sys, "argv", ["plan_task.py", "--manifest", manifest, "--dry-run"]
    )
    assert plan_task.main() == 0
    assert "planner: scripted_planner" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("manifest", "robot"),
    [
        pytest.param(
            "configs/manifests/so101_single_cube_fixed_place.yaml",
            "so101",
            marks=requires_assets,
        ),
        pytest.param(
            "configs/manifests/viperx300s_single_cube_fixed_place.yaml",
            "viperx300s",
            marks=requires_viperx300s_assets,
        ),
    ],
)
def test_export_scene_writes_the_manifests_scene(
    monkeypatch, capsys, tmp_path, manifest, robot
):
    export_scene = _script(monkeypatch, "export_scene")
    out = tmp_path / "scene.xml"
    monkeypatch.setattr(
        sys, "argv", ["export_scene.py", "--manifest", manifest, "--out", str(out)]
    )
    assert export_scene.main() == 0
    assert "<mujoco" in out.read_text(encoding="utf-8")
    assert f"assets/{robot}/" in capsys.readouterr().out


def test_export_scene_says_so_when_the_robot_has_no_scene(monkeypatch):
    export_scene = _script(monkeypatch, "export_scene")
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_scene.py", "--manifest", "configs/manifests/turtlebot4.yaml"],
    )
    with pytest.raises(SystemExit) as stopped:
        export_scene.main()
    assert stopped.value.code == 2
