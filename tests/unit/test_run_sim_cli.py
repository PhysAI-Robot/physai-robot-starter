import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_serve_alone_runs_headless_without_desktop_window(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    captured = {}

    def fake_run_viewer(args, task_config, seed, max_steps, domain_randomization):
        del task_config, seed, max_steps, domain_randomization
        captured["viewer"] = args.viewer
        captured["serve"] = args.serve
        return 0

    monkeypatch.setattr(run_sim, "run_viewer", fake_run_viewer)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", "--serve"])

    code = run_sim.main()

    assert code == 0
    assert captured == {"viewer": False, "serve": True}


class _StopBuilding(Exception):
    pass


def test_turtlebot_viewer_gets_a_resolved_step_limit(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    captured = {}

    def fake_create_robot(name, *, config):
        captured["name"] = name
        captured["max_steps"] = config.max_steps
        raise _StopBuilding

    monkeypatch.setattr(run_sim, "create_robot", fake_create_robot)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", "--robot", "turtlebot4", "--serve"])

    try:
        run_sim.main()
    except _StopBuilding:
        pass

    assert captured["name"] == "turtlebot4"
    assert isinstance(captured["max_steps"], int)
