import sys
from pathlib import Path

import pytest
from conftest import requires_assets

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def run_sim(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    return run_sim


def capture_viewer(run_sim, monkeypatch, argv):
    """Run main() up to the viewer, returning what it would have been given."""
    captured = {}

    def fake_run_viewer(args, manifest):
        captured["args"] = args
        captured["manifest"] = manifest
        return 0

    monkeypatch.setattr(run_sim, "run_viewer", fake_run_viewer)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", *argv])
    assert run_sim.main() == 0
    return captured


def test_serve_alone_runs_headless_without_desktop_window(run_sim, monkeypatch):
    captured = capture_viewer(run_sim, monkeypatch, ["--serve"])

    assert (captured["args"].viewer, captured["args"].serve) == (False, True)


def test_turtlebot_viewer_gets_a_resolved_step_limit(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim, monkeypatch, ["--robot", "turtlebot4", "--serve"]
    )

    (robot,) = captured["manifest"].robots
    assert robot.robot == "turtlebot4"
    assert isinstance(robot.config["max_steps"], int)


def test_a_bare_manipulator_run_keeps_its_task_and_wide_cameras(run_sim, monkeypatch):
    captured = capture_viewer(run_sim, monkeypatch, ["--serve"])

    manifest = captured["manifest"]
    assert manifest.task == "pick_place"
    assert manifest.scene.overrides["camera_width"] == 640


def test_command_line_overrides_reach_the_manifest(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--serve", "--seed", "4", "--max-steps", "77", "--camera-size", "96"],
    )

    manifest = captured["manifest"]
    assert manifest.simulation.seed == 4
    assert manifest.robots[0].config["max_steps"] == 77
    assert manifest.scene.overrides["camera_height"] == 96


def test_a_manifest_file_selects_the_run(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--manifest", "configs/manifests/so101_pick_place.yaml", "--serve"],
    )

    manifest = captured["manifest"]
    assert manifest.robots[0].id == "so101"
    assert manifest.robots[0].config["max_steps"] == 400


def test_a_deprecated_config_still_works_and_says_so(run_sim, monkeypatch, capsys):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--config", "configs/tasks/so101/pick_place.yaml", "--serve"],
    )

    assert captured["manifest"].scene.name == "pick_place_minimal"
    assert "--config is deprecated" in capsys.readouterr().err


def test_a_world_file_becomes_a_shared_world_manifest(run_sim, monkeypatch, capsys):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--world", "configs/worlds/heterogeneous.yaml", "--serve"],
    )

    assert captured["manifest"].world is not None
    assert "--world is deprecated" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--manifest", "m.yaml", "--config", "c.yaml"], "cannot be combined"),
        (["--manifest", "m.yaml", "--robot", "so101"], "cannot be combined"),
        (["--world", "w.yaml", "--robot", "so101"], "cannot be combined"),
        (["--record-dir", "d"], "requires --serve"),
        (
            ["--world", "configs/worlds/heterogeneous.yaml"],
            "requires --viewer or --serve",
        ),
        (
            [
                "--world",
                "configs/worlds/heterogeneous.yaml",
                "--serve",
                "--policy",
                "constant",
            ],
            "cannot be used with a shared world",
        ),
        (
            [
                "--world",
                "configs/worlds/heterogeneous.yaml",
                "--serve",
                "--record-dir",
                "d",
            ],
            "not available with a shared world",
        ),
        (
            [
                "--config",
                "configs/tasks/so101/pick_place.yaml",
                "--robot",
                "turtlebot4",
            ],
            "does not match",
        ),
    ],
)
def test_incompatible_flags_are_rejected(run_sim, monkeypatch, capsys, argv, message):
    monkeypatch.setattr(sys, "argv", ["run_sim.py", *argv])

    with pytest.raises(SystemExit) as exit_info:
        run_sim.main()

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


@requires_assets
def test_default_policy_names_the_video_file(run_sim, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sim.py",
            "--video",
            "--max-steps",
            "3",
            "--camera-size",
            "64",
            "--out",
            str(tmp_path),
        ],
    )

    assert run_sim.main() == 0

    assert [path.stem for path in tmp_path.iterdir()] == ["scripted_ep000"]


@requires_assets
def test_a_manifest_runs_headless_episodes(run_sim, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sim.py",
            "--manifest",
            "configs/manifests/so101_pick_place.yaml",
            "--max-steps",
            "3",
            "--episodes",
            "2",
            "--out",
            str(tmp_path),
        ],
    )

    assert run_sim.main() == 0

    assert "0/2 successful" in capsys.readouterr().out
