"""Run probes read-only, offline, time-limited and output-capped.

macOS: sandbox-exec with a deny-write / deny-network / deny-secret-read profile.
Linux: bwrap with a read-only root, no network, and tmpfs over secret dirs.
Probes never run unsandboxed: if no sandbox is available, `run` refuses.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tth.safety.secrets import secret_dirs

DEFAULT_TIMEOUT = 3.0
DEFAULT_CAP = 8 * 1024


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class ProbeResult:
    exit_code: int | None
    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool
    seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _sb_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def macos_profile(tmpdir: str, writable: list[str] | None = None) -> str:
    deny_reads = "\n".join(
        f"  (subpath {_sb_quote(str(d))})" for d in secret_dirs()
    )  # fmt: skip
    extra = "".join(f" (subpath {_sb_quote(os.path.realpath(w))})" for w in writable or [])
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper")
  (regex #"^/dev/tty") (regex #"^/dev/fd/")
  (subpath {_sb_quote(tmpdir)}) (subpath {_sb_quote(os.path.realpath(tmpdir))}){extra})
(deny file-read*
{deny_reads}
  (regex #"/\\.env(\\.[^/]*)?$"))
"""


def bwrap_argv(cwd: str, tmpdir: str, writable: list[str] | None = None) -> list[str]:
    argv = [
        "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--bind", tmpdir, tmpdir, "--unshare-net", "--unshare-pid", "--unshare-ipc",
        "--die-with-parent", "--new-session",
    ]  # fmt: skip
    for d in secret_dirs():
        if d.is_dir():
            argv += ["--tmpfs", str(d)]
        elif d.exists():
            argv += ["--ro-bind", "/dev/null", str(d)]
    for w in writable or []:
        argv += ["--bind", w, w]
    argv += ["--chdir", cwd]
    return argv


DISPOSABLE = "disposable-container"


def disposable_host() -> bool:
    """True only inside a throwaway Kaggle/Colab VM with TTH_SANDBOX=disposable-container set."""
    hosted = any(
        k in os.environ for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG")
    ) or os.path.isdir("/kaggle/working")
    return os.environ.get("TTH_SANDBOX") == DISPOSABLE and hosted


def sandbox_prefix(
    cwd: str, tmpdir: str, writable: list[str] | None = None, allow_disposable: bool = False
) -> list[str]:
    # Training/eval envs on a disposable hosted VM (no user namespaces there) may run probes without
    # bwrap. The product never passes allow_disposable, so `tth` on a user's machine always sandboxes.
    if allow_disposable and disposable_host():
        return []
    if sys.platform == "darwin":
        exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        if not os.path.exists(exe):
            raise SandboxUnavailable("sandbox-exec not found")
        return [exe, "-p", macos_profile(tmpdir, writable)]
    if sys.platform.startswith("linux"):
        if not shutil.which("bwrap"):
            raise SandboxUnavailable("bwrap not installed (apt install bubblewrap)")
        return bwrap_argv(cwd, tmpdir, writable)
    raise SandboxUnavailable(f"no sandbox for platform {sys.platform}")


def _read_capped(stream, cap: int, sink: list[bytes], flag: list[bool]) -> None:
    total = 0
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        if total < cap:
            sink.append(chunk[: cap - total])
        total += len(chunk)
        if total > cap:
            flag[0] = True


def run(
    argv: list[str],
    cwd: str | os.PathLike[str],
    timeout: float = DEFAULT_TIMEOUT,
    cap: int = DEFAULT_CAP,
    stdin: bytes | None = None,
    writable: list[str] | None = None,
    allow_disposable: bool = False,
) -> ProbeResult:
    """Run argv inside the sandbox. Never call this on a staged final command.

    `writable` is only used by training/eval verifiers on throwaway fixture copies.
    """
    cwd = str(Path(cwd).resolve())
    with tempfile.TemporaryDirectory(prefix="tth-sbx-") as tmp:
        cmd = sandbox_prefix(cwd, tmp, writable, allow_disposable) + list(argv)
        env = {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": os.path.expanduser("~"),
            "TMPDIR": tmp,
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "LC_ALL": "C.UTF-8" if sys.platform != "darwin" else "en_US.UTF-8",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "GIT_OPTIONAL_LOCKS": "0",
            "TERM": "dumb",
        }
        start = time.monotonic()
        proc = subprocess.Popen(  # noqa: S603 - sandboxed probe, never the staged command
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        out: list[bytes] = []
        err: list[bytes] = []
        out_trunc, err_trunc = [False], [False]
        threads = [
            threading.Thread(target=_read_capped, args=(proc.stdout, cap, out, out_trunc)),
            threading.Thread(target=_read_capped, args=(proc.stderr, 2048, err, err_trunc)),
        ]
        for t in threads:
            t.daemon = True
            t.start()
        if stdin is not None and proc.stdin:
            try:
                proc.stdin.write(stdin)
                proc.stdin.close()
            except BrokenPipeError:
                pass
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for t in threads:
            t.join(timeout=1)
        return ProbeResult(
            exit_code=None if timed_out else proc.returncode,
            stdout=b"".join(out).decode("utf-8", "replace"),
            stderr=b"".join(err).decode("utf-8", "replace"),
            truncated=out_trunc[0],
            timed_out=timed_out,
            seconds=round(time.monotonic() - start, 3),
        )


def run_shell(command: str, cwd: str | os.PathLike[str], **kw) -> ProbeResult:
    """Run a classified read-only command string through /bin/sh inside the sandbox."""
    return run(["/bin/sh", "-c", command], cwd, **kw)
