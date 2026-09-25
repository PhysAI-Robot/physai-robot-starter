import sys
from pathlib import Path

import pytest
from conftest import requires_assets

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
WORLD = "configs/worlds/heterogeneous.yaml"
TASK = "configs/tasks/so101/pick_place.yaml"


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


def test_serve_alone_runs_headless_and_overrides_reach_the_manifest(
    run_sim, monkeypatch
):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--serve", "--seed", "4", "--max-steps", "77", "--camera-size", "96"],
    )

    assert (captured["args"].viewer, captured["args"].serve) == (False, True)
    manifest = captured["manifest"]
    assert manifest.task == "pick_place"
    assert manifest.simulation.seed == 4
    assert manifest.robots[0].config["max_steps"] == 77
    assert manifest.scene.overrides["camera_height"] == 96


def test_turtlebot_viewer_gets_a_resolved_step_limit(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim, monkeypatch, ["--robot", "turtlebot4", "--serve"]
    )

    (robot,) = captured["manifest"].robots
    assert robot.robot == "turtlebot4"
    assert isinstance(robot.config["max_steps"], int)


def test_a_manifest_selects_the_run_and_the_old_flags_work_with_a_notice(
    run_sim, monkeypatch, capsys
):
    manifest = capture_viewer(
        run_sim,
        monkeypatch,
        ["--manifest", "configs/manifests/so101_pick_place.yaml", "--serve"],
    )["manifest"]
    assert manifest.robots[0].id == "so101"
    assert manifest.robots[0].config["max_steps"] == 400

    config = capture_viewer(run_sim, monkeypatch, ["--config", TASK, "--serve"])
    assert config["manifest"].scene.name == "pick_place_minimal"
    assert "--config is deprecated" in capsys.readouterr().err

    world = capture_viewer(run_sim, monkeypatch, ["--world", WORLD, "--serve"])
    assert world["manifest"].world is not None
    assert "--world is deprecated" in capsys.readouterr().err


def test_incompatible_flags_are_rejected(run_sim, monkeypatch, capsys):
    cases = [
        (["--manifest", "m.yaml", "--config", "c.yaml"], "cannot be combined"),
        (["--manifest", "m.yaml", "--robot", "so101"], "cannot be combined"),
        (["--world", "w.yaml", "--robot", "so101"], "cannot be combined"),
        (["--record-dir", "d"], "requires --serve"),
        (["--world", WORLD], "requires --viewer or --serve"),
        (
            ["--world", WORLD, "--serve", "--policy", "constant"],
            "cannot be used with a shared world",
        ),
        (
            ["--world", WORLD, "--serve", "--record-dir", "d"],
            "not available with a shared world",
        ),
        (["--config", TASK, "--robot", "turtlebot4"], "does not match"),
    ]

    for argv, message in cases:
        monkeypatch.setattr(sys, "argv", ["run_sim.py", *argv])
        with pytest.raises(SystemExit) as exit_info:
            run_sim.main()
        assert exit_info.value.code == 2, argv
        assert message in capsys.readouterr().err, argv


@requires_assets
def test_headless_episodes_run_from_a_manifest_and_name_their_video(
    run_sim, monkeypatch, tmp_path, capsys
):
    manifest_run = [
        "run_sim.py",
        "--manifest",
        "configs/manifests/so101_pick_place.yaml",
        "--max-steps",
        "3",
        "--episodes",
        "2",
        "--out",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", manifest_run)
    assert run_sim.main() == 0
    assert "0/2 successful" in capsys.readouterr().out

    # with no policy named, the video is named after the one actually run
    videos = tmp_path / "videos"
    video_run = ["run_sim.py", "--video", "--max-steps", "3", "--camera-size", "64"]
    monkeypatch.setattr(sys, "argv", [*video_run, "--out", str(videos)])
    assert run_sim.main() == 0
    assert [path.stem for path in videos.iterdir()] == ["scripted_ep000"]
