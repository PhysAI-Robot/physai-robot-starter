"""Every CLI in scripts/ must at least import and print its --help.

The scripts are run in one subprocess, not in this process: importing them
registers research modules into the global registries, which would leak into
tests that expect those registrations to happen only on an explicit import.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

DRIVER = """
import contextlib, io, json, runpy, sys
from pathlib import Path

scripts = Path(sys.argv[1])
sys.path.insert(0, str(scripts))
failures = {}
for script in sorted(scripts.glob("*.py")):
    if script.name.startswith("_"):
        continue
    sys.argv = [script.name, "--help"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            # A script without arguments (show_ros2_contract.py) just runs.
            runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None) or "usage" not in out.getvalue().lower():
            failures[script.name] = f"exit {exc.code!r}"
    except ModuleNotFoundError as exc:
        if exc.name != "rclpy":  # ROS2 scripts need a ROS2 install
            failures[script.name] = f"ModuleNotFoundError: {exc}"
    except BaseException as exc:
        failures[script.name] = f"{type(exc).__name__}: {exc}"
print(json.dumps(failures))
"""


def test_every_script_prints_help():
    result = subprocess.run(
        [sys.executable, "-c", DRIVER, str(SCRIPTS)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {}
