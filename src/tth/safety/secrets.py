"""Secret paths that tth never probes, dry-runs or sends anywhere.

This is a layer on top of the sandbox profile, which also denies reads of the
same locations (see tth.sandbox).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Relative to $HOME.
SECRET_HOME_DIRS = [
    ".ssh", ".aws", ".gnupg", ".config/gcloud", ".azure", ".kube", ".docker",
    ".password-store", ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".config/gh",
    "Library/Keychains", "Library/Cookies", "Library/Safari",
    "Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
    "Library/Application Support/BraveSoftware", "Library/Application Support/Microsoft Edge",
    ".mozilla", ".config/google-chrome", ".config/chromium", ".local/share/keyrings",
    ".config/tth",
]  # fmt: skip

_SECRET_NAME = re.compile(
    r"(^|/)(\.env(\..*)?|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|.*\.pem|.*\.key|.*\.p12|.*\.pfx|"
    r"credentials(\.json)?|\.htpasswd|.*\.kdbx|.*keychain.*)$",
    re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(\.ssh|\.aws|\.gnupg|\.env\b|Keychains|\.netrc|\.kube|\.docker/config|id_rsa|id_ed25519|"
    r"\.git-credentials|\.password-store|Cookies|\.config/gcloud|\.config/gh|\.config/tth)",
)


def home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def secret_dirs() -> list[Path]:
    h = home()
    return [h / d for d in SECRET_HOME_DIRS]


def is_secret_path(path: str | os.PathLike[str], cwd: str | os.PathLike[str] | None = None) -> bool:
    raw = os.path.expanduser(os.path.expandvars(str(path)))
    p = Path(raw)
    if not p.is_absolute():
        p = Path(cwd or os.getcwd()) / p
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    for cand in {p, resolved}:
        if _SECRET_NAME.search(str(cand)):
            return True
        for d in secret_dirs():
            if cand == d or d in cand.parents:
                return True
    return False


def mentions_secret(text: str) -> bool:
    """Cheap textual check for arguments that may expand to secret paths."""
    return bool(_SECRET_TEXT.search(text))
