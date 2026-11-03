"""The project site must show what the repo says, not a stale copy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_site_data", ROOT / "scripts" / "build_site_data.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_site_data_is_current() -> None:
    assert (ROOT / "site" / "data.js").read_text() == _builder().build(), (
        "run: uv run python scripts/build_site_data.py"
    )


def test_gallery_shows_real_refusals() -> None:
    gallery = {g["command"]: g for g in _builder().gallery()}
    assert gallery["rm -rf ~"]["staged"] is None
    assert gallery["grep -c ERROR app.log"]["tier"] == "read-only"
