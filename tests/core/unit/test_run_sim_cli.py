import sys
from pathlib import Path

import pytest
from tests.conftest import requires_assets

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
WORLD = "configs/manifests/heterogeneous_world.yaml"
ISAAC = "configs/manifests/so101_single_cube_fixed_place.yaml"


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
        ["--serve", "--seed", "4", "--max-steps", "77"],
    )

    assert (captured["args"].viewer, captured["args"].serve) == (False, True)
    manifest = captured["manifest"]
    assert manifest.task == "single_cube_fixed_place"
    assert manifest.simulation.seed == 4
    assert manifest.robots[0].config["max_steps"] == 77


def test_isaac_serve_reaches_the_viewer_path(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim, monkeypatch, ["--sim", "isaac", "--manifest", ISAAC, "--serve"]
    )

    assert captured["manifest"].simulator == "isaac"


def test_camera_res_flag_reaches_the_manifest_and_rejects_other_sizes(
    run_sim, monkeypatch
):
    captured = capture_viewer(
        run_sim, monkeypatch, ["--serve", "--camera-res", "640x480"]
    )
    assert captured["manifest"].simulation.camera_resolution == "640x480"

    default = capture_viewer(run_sim, monkeypatch, ["--serve"])
    assert default["manifest"].simulation.camera_resolution == "320x240"

    monkeypatch.setattr(sys, "argv", ["run_sim.py", "--camera-res", "64x64"])
    with pytest.raises(SystemExit):
        run_sim.main()


def test_turtlebot_viewer_gets_a_resolved_step_limit(run_sim, monkeypatch):
    captured = capture_viewer(
        run_sim,
        monkeypatch,
        ["--manifest", "configs/manifests/turtlebot4.yaml", "--serve"],
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
        [
            "--manifest",
            "configs/manifests/so101_single_cube_fixed_place.yaml",
            "--serve",
        ],
    )["manifest"]
    assert manifest.robots[0].id == "so101"
    assert manifest.robots[0].config["max_steps"] == 600

    world = capture_viewer(run_sim, monkeypatch, ["--manifest", WORLD, "--serve"])
    assert world["manifest"].world is not None


def test_incompatible_flags_are_rejected(run_sim, monkeypatch, capsys):
    cases = [
        (["--record", "--serve"], "use --dataset"),
        (["--dataset", "d"], "needs --serve"),
        (["--manifest", WORLD], "requires --viewer or --serve"),
        (
            ["--manifest", WORLD, "--serve", "--policy", "constant"],
            "cannot be used with a shared world",
        ),
        (
            ["--manifest", WORLD, "--serve", "--dataset", "d"],
            "not available with a shared world",
        ),
        (
            ["--sim", "isaac", "--manifest", ISAAC, "--viewer"],
            "--viewer is MuJoCo-only",
        ),
        (
            ["--sim", "isaac", "--manifest", ISAAC, "--serve", "--dataset", "d"],
            "--dataset is MuJoCo-only",
        ),
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
        "configs/manifests/so101_single_cube_fixed_place.yaml",
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

    # with no policy named, the video is named after the simulator, the robot, the
    # policy actually run and the seed
    out = tmp_path / "run"
    video_run = ["run_sim.py", "--video", "--record", "--max-steps", "3"]
    monkeypatch.setattr(sys, "argv", [*video_run, "--out", str(out)])
    assert run_sim.main() == 0
    assert [path.stem for path in (out / "videos").iterdir()] == [
        "mujoco_so101_scripted_seed0000"
    ]
    # the recording is named like the video, with its meta next to it
    assert sorted(path.name for path in (out / "recordings").iterdir()) == [
        "mujoco_so101_scripted_seed0000.json",
        "mujoco_so101_scripted_seed0000.npz",
    ]

    # --name replaces the automatic prefix for both
    monkeypatch.setattr(sys, "argv", [*video_run, "--name", "mine", "--out", str(out)])
    assert run_sim.main() == 0
    assert (out / "videos" / "mine_seed0000.mp4").exists()
    assert (out / "recordings" / "mine_seed0000.npz").exists()


def test_without_a_manifest_the_single_cube_session_runs(run_sim, monkeypatch):
    manifest = capture_viewer(run_sim, monkeypatch, ["--serve"])["manifest"]
    assert manifest.scene.name == "single_cube_fixed_place"


@requires_assets
def test_a_robot_without_a_scripted_expert_runs_headless_on_the_constant_policy(
    run_sim, monkeypatch, capsys
):
    argv = ["run_sim.py", "--manifest", "configs/manifests/turtlebot4.yaml"]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-steps", "3"])
    assert run_sim.main() == 0
    assert "episode 0:" in capsys.readouterr().out
