"""Invariant 1: tth never executes the final command.

Process-spawning calls may only appear in the sandbox runner (probes) and in the
CLI helper that starts the expert server. Nothing else in src/tth may spawn.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tth.cli.main import stage_output

SRC = Path(__file__).resolve().parents[1] / "src" / "tth"
SPAWN = {("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"), ("subprocess", "check_call"),
         ("subprocess", "check_output"), ("os", "system"), ("os", "popen"), ("os", "execv"), ("os", "execvp"),
         ("os", "execl"), ("os", "spawnv"), ("os", "posix_spawn")}  # fmt: skip
ALLOWED = {("sandbox/runner.py", "run"), ("cli/main.py", "_start_expert")}


def _spawns(tree: ast.AST) -> list[tuple[str, int]]:
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            ):
                if (node.func.value.id, node.func.attr) in SPAWN:
                    found.append((fn.name, node.lineno))
    return found


def test_only_allowed_spawn_sites() -> None:
    offenders = []
    for f in SRC.rglob("*.py"):
        rel = f.relative_to(SRC).as_posix()
        for fn, line in _spawns(ast.parse(f.read_text())):
            if (rel, fn) not in ALLOWED:
                offenders.append(f"{rel}:{line} in {fn}")
    assert not offenders, offenders


def test_stage_writes_never_runs(tmp_path: Path) -> None:
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    out = tmp_path / "out"
    stage_output(f"trash {victim}", "changes-files", [], str(out), "test")
    assert out.read_text() == f"trash {victim}"
    assert victim.exists()


def test_product_never_uses_disposable_mode(monkeypatch, tmp_path: Path) -> None:
    """Invariant 2: even with the env var and a hosted-VM marker set, the product toolbox sandboxes."""
    from tth.sandbox.runner import sandbox_prefix
    from tth.tools.shell_tools import ShellToolbox

    monkeypatch.setenv("TTH_SANDBOX", "disposable-container")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert ShellToolbox(tmp_path).allow_disposable is False
    import sys

    if sys.platform == "darwin":
        assert sandbox_prefix(str(tmp_path), str(tmp_path))[0].endswith("sandbox-exec")
    assert sandbox_prefix(str(tmp_path), str(tmp_path), allow_disposable=True) == []
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE")
    if sys.platform == "darwin":
        assert sandbox_prefix(str(tmp_path), str(tmp_path), allow_disposable=True)[0].endswith("sandbox-exec")
