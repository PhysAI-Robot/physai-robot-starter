import tomllib
from pathlib import Path

import physai

ROOT = Path(__file__).resolve().parents[3]


def test_the_package_version_matches_pyproject_and_the_viewer_header_shows_it():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert physai.__version__ == pyproject["project"]["version"]
    index = (ROOT / "src/physai/web/static/index.html").read_text(encoding="utf-8")
    assert "v{{version}}" in index  # web/app.py fills it from physai.__version__
