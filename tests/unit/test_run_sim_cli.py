import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def run_main(monkeypatch, capsys, *argv: str) -> tuple[int, str]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    monkeypatch.setattr(sys, "argv", ["run_sim.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run_sim.main()
    return exit_info.value.code, capsys.readouterr().err


def test_serve_alone_runs_headless_without_desktop_window(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    captured = {}

    def fake_run_viewer(args, task_config, seed, max_steps, domain_randomization):
        del task_config, seed, max_steps, domain_randomization
        captured["viewer"] = args.viewer
        captured["serve"] = args.serve
        captured["headless"] = args.headless
        return 0

    monkeypatch.setattr(run_sim, "run_viewer", fake_run_viewer)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", "--serve"])

    code = run_sim.main()

    assert code == 0
    assert captured == {"viewer": False, "serve": True, "headless": False}


def test_headless_and_bare_serve_are_equivalent(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    captured = {}

    def fake_run_viewer(args, task_config, seed, max_steps, domain_randomization):
        del task_config, seed, max_steps, domain_randomization
        captured["viewer"] = args.viewer
        return 0

    monkeypatch.setattr(run_sim, "run_viewer", fake_run_viewer)
    monkeypatch.setattr(sys, "argv", ["run_sim.py", "--headless", "--serve"])

    code = run_sim.main()

    assert code == 0
    assert captured == {"viewer": False}


def test_headless_requires_serve(monkeypatch, capsys):
    code, err = run_main(monkeypatch, capsys, "--headless")

    assert code == 2
    assert "--headless requires --serve" in err


def test_headless_and_viewer_are_mutually_exclusive(monkeypatch, capsys):
    code, err = run_main(monkeypatch, capsys, "--headless", "--viewer", "--serve")

    assert code == 2
    assert "mutually exclusive" in err
