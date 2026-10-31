"""The sandbox must block writes, network and secret reads, and enforce limits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tth.sandbox.runner import bwrap_argv, macos_profile, run, run_shell

needs_macos = pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")


@needs_macos
def test_reads_work(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    r = run(["cat", "a.txt"], tmp_path)
    assert r.ok and r.stdout == "hello\n"


@needs_macos
def test_writes_blocked(tmp_path: Path) -> None:
    r = run_shell("echo x > pwned.txt", tmp_path)
    assert not r.ok
    assert not (tmp_path / "pwned.txt").exists()
    r = run(["touch", str(tmp_path / "t")], tmp_path)
    assert not (tmp_path / "t").exists()


@needs_macos
def test_delete_blocked(tmp_path: Path) -> None:
    f = tmp_path / "keep.txt"
    f.write_text("x")
    run(["rm", "-f", str(f)], tmp_path)
    assert f.exists()


@needs_macos
def test_network_blocked(tmp_path: Path) -> None:
    r = run(["curl", "-sS", "-m", "2", "https://example.com"], tmp_path)
    assert not r.ok


@needs_macos
def test_secret_reads_blocked(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=abc\n")
    r = run(["cat", ".env"], tmp_path)
    assert "abc" not in r.stdout and not r.ok
    r = run_shell("ls ~/.ssh", tmp_path)
    assert not r.ok or r.stdout == ""


@needs_macos
def test_timeout_and_cap(tmp_path: Path) -> None:
    r = run(["sleep", "10"], tmp_path, timeout=0.5)
    assert r.timed_out and r.seconds < 3
    r = run(["yes"], tmp_path, timeout=1, cap=1000)
    assert r.truncated and len(r.stdout) <= 1000


def test_profile_mentions_secrets(tmp_path: Path) -> None:
    prof = macos_profile(str(tmp_path))
    assert "(deny network*)" in prof and "(deny file-write*)" in prof and ".ssh" in prof


def test_bwrap_argv_shape(tmp_path: Path) -> None:
    argv = bwrap_argv(str(tmp_path), str(tmp_path))
    assert argv[:5] == ["bwrap", "--ro-bind", "/", "/", "--dev"]
    assert "--unshare-net" in argv and argv[-2:] == ["--chdir", str(tmp_path)]
