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


def test_serve_alone_names_both_ways_to_host(monkeypatch, capsys):
    code, err = run_main(monkeypatch, capsys, "--serve")

    assert code == 2
    assert "--serve requires --viewer or --headless" in err


def test_headless_requires_serve(monkeypatch, capsys):
    code, err = run_main(monkeypatch, capsys, "--headless")

    assert code == 2
    assert "--headless requires --serve" in err


def test_headless_and_viewer_are_mutually_exclusive(monkeypatch, capsys):
    code, err = run_main(monkeypatch, capsys, "--headless", "--viewer", "--serve")

    assert code == 2
    assert "mutually exclusive" in err
