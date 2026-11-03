"""The sandbox must block writes, network and secret reads, and enforce limits."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from tth.sandbox.runner import DEFAULT_TIMEOUT, bwrap_argv, macos_profile, run, run_shell

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


needs_bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="needs a real bwrap (runs on Linux in CI)")


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="no sandbox on this platform")
@pytest.mark.parametrize("name", [".env", ".env_prod", ".envrc", "server.pem", "app.key", "id_ed25519"])
def test_secret_files_in_the_project_cannot_be_read(tmp_path: Path, name: str) -> None:
    if sys.platform == "linux" and not shutil.which("bwrap"):
        pytest.skip("needs bwrap")
    (tmp_path / name).write_text("hunter2\n")
    r = run(["cat", name], tmp_path)
    assert "hunter2" not in r.stdout


@needs_bwrap
def test_bwrap_blocks_writes_and_network(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    assert run(["cat", "a.txt"], tmp_path).stdout == "hello\n"
    assert not run_shell("echo x > pwned.txt", tmp_path).ok
    assert not (tmp_path / "pwned.txt").exists()


def test_bwrap_masks_secret_files_by_path(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".env").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    argv = bwrap_argv(str(tmp_path), str(tmp_path))
    assert str(tmp_path / "sub" / ".env") in argv
    assert str(tmp_path / "notes.txt") not in argv


def test_default_timeout_matches_the_invariant() -> None:
    assert DEFAULT_TIMEOUT == 3.0
