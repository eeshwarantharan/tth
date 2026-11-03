"""Every result says exactly what produced it. The report refuses a result that doesn't.

v0.1's results recorded no commit, adapter or config, and some turned out to come from older adapters than the
ones on disk. That can't happen quietly any more.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REQUIRED = ("arm", "git_sha", "dirty", "policy", "budget", "suites", "platform", "finished")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def adapter_hash(adapter: str | None) -> str | None:
    if not adapter:
        return None
    path = Path(adapter)
    files = sorted(p for p in (path.rglob("*.safetensors") if path.is_dir() else [path]))
    return hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()[:16] if files else None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build(arm: str, label: str, policy: str, budget: int, suites: dict[str, Path], **extra: Any) -> dict[str, Any]:
    return {
        "arm": arm,
        "label": label,
        "git_sha": _git("rev-parse", "HEAD"),
        "dirty": bool(
            _git(
                "status", "--porcelain", "--", "packages", "evals/run.py", "evals/shell", "evals/chess", "evals/redteam"
            )
        ),
        "policy": policy,
        "budget": budget,
        "suites": {name: sha256(path) for name, path in suites.items() if path.exists()},
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "tth": _package_version("tth"),
        "mlx_lm": _package_version("mlx-lm"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def save(results: Path, manifest: dict[str, Any]) -> None:
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (results / f"{manifest['arm']}__manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def load(results: Path, arm: str) -> dict[str, Any]:
    """The manifest for an arm. Raises if it's missing or incomplete, so nothing untraceable gets reported."""
    path = results / f"{arm}__manifest.json"
    if not path.exists():
        raise SystemExit(f"Refusing to report {arm}: {path} is missing, so nothing says what produced these results.")
    manifest = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in manifest]
    if missing:
        raise SystemExit(f"Refusing to report {arm}: its manifest is missing {', '.join(missing)}.")
    return manifest
