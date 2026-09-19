import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def fetch_assets(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    sys.modules.pop("fetch_assets", None)
    import fetch_assets

    return fetch_assets


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def record_requests(monkeypatch, listing):
    """Serve `listing` for API calls and a small blob for file downloads."""
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        if request.full_url.startswith("https://api.github.com/"):
            return Response(json.dumps(listing).encode())
        return Response(b"file-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_api_requests_carry_the_token_when_one_is_set(monkeypatch, fetch_assets):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    seen = record_requests(monkeypatch, [])

    fetch_assets._get_json("https://api.github.com/repos/o/r/contents/p?ref=main")

    assert seen[0].get_header("Authorization") == "Bearer secret-token"


def test_gh_token_is_accepted_as_a_fallback(monkeypatch, fetch_assets):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gh-token")
    seen = record_requests(monkeypatch, [])

    fetch_assets._get_json("https://api.github.com/repos/o/r/contents/p?ref=main")

    assert seen[0].get_header("Authorization") == "Bearer gh-token"


def test_requests_stay_unauthenticated_without_a_token(monkeypatch, fetch_assets):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    seen = record_requests(monkeypatch, [])

    fetch_assets._get_json("https://api.github.com/repos/o/r/contents/p?ref=main")

    assert seen[0].get_header("Authorization") is None


def test_the_token_is_never_sent_with_file_downloads(
    monkeypatch, tmp_path, fetch_assets
):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    listing = [
        {
            "name": "model.xml",
            "type": "file",
            "download_url": "https://raw.githubusercontent.com/o/r/main/p/model.xml",
        }
    ]
    seen = record_requests(monkeypatch, listing)

    files, _ = fetch_assets.walk("p", tmp_path, Path("."), False, "o/r", "main")

    assert files == 1
    api, download = seen
    assert api.get_header("Authorization") == "Bearer secret-token"
    assert download.full_url.startswith("https://raw.githubusercontent.com/")
    assert download.get_header("Authorization") is None
