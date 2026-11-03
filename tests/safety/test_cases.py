"""Red-team suite: every YAML case must classify and rewrite as declared."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tth.safety.classifier import Tier, classify
from tth.safety.rewriter import rewrite, selection_part, trash_argv

CASES_DIR = Path(__file__).parent / "cases"


def _on_this_platform(expected: str | None) -> str | None:
    """Cases are written for macOS `trash`. Linux stages `gio trash`."""
    return expected.replace("trash ", " ".join(trash_argv()) + " ") if expected else expected


def _load() -> list[tuple[str, dict]]:
    out = []
    for f in sorted(CASES_DIR.glob("*.yaml")):
        for i, case in enumerate(yaml.safe_load(f.read_text())):
            out.append((f"{f.stem}[{i}]", case))
    return out


CASES = _load()


@pytest.mark.parametrize("case", [c for _, c in CASES], ids=[i for i, _ in CASES])
def test_case(case: dict, tmp_path: Path) -> None:
    cmd = case["cmd"]
    c = classify(cmd)
    assert c.tier.label == case["tier"], f"{cmd!r}: {c.tier.label} {c.reasons}"
    rw = rewrite(cmd, cwd=tmp_path)
    if "staged" in case:
        assert rw.staged == _on_this_platform(case["staged"]), rw.notes
    if "staged_contains" in case:
        assert rw.staged is not None and case["staged_contains"] in rw.staged, rw.notes
    if case.get("never_raw"):
        assert rw.staged != cmd
    if "preview" in case:
        assert rw.preview == case["preview"]


@pytest.mark.parametrize("case", [c for _, c in CASES], ids=[i for i, _ in CASES])
def test_invariants(case: dict, tmp_path: Path) -> None:
    cmd = case["cmd"]
    c = classify(cmd)
    rw = rewrite(cmd, cwd=tmp_path)
    # Destructive commands are never staged raw.
    if c.destructive:
        assert rw.staged != cmd
    # Denied commands are never staged.
    if c.tier == Tier.DENIED:
        assert rw.staged is None
    # Whatever we stage re-classifies at or below needs-review, never denied.
    if rw.staged is not None:
        assert classify(rw.staged).tier != Tier.DENIED
    # The dry-run selection is always read-only.
    sel = selection_part(cmd)
    if sel is not None:
        assert classify(sel).tier == Tier.READ_ONLY


def test_suite_size() -> None:
    assert len(CASES) >= 150


def test_redirect_clobber_refused(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("keep me")
    rw = rewrite("grep x a > out.txt", cwd=tmp_path)
    assert rw.staged is None
    assert rewrite("grep x a > new.txt", cwd=tmp_path).staged == "grep x a > new.txt"
