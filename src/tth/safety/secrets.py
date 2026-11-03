"""Secret paths that tth never probes, dry-runs or sends anywhere.

The single source of truth: the sandbox, the classifier and the tools all use these lists. This is a layer on
top of the sandbox profile, which also denies reads of the same locations (see tth.sandbox).
"""

from __future__ import annotations

import fnmatch
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
    ".config/tth", ".bash_history", ".zsh_history", ".sh_history", ".python_history",
    ".psql_history", ".mysql_history", ".node_repl_history", ".sqlite_history",
]  # fmt: skip

# File names that are secret wherever they appear.
_NAME_RE_BODY = (
    r"\.env[^/]*|id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?|[^/]*\.(?:pem|key|p12|pfx|kdbx|jks|keystore)|"
    r"credentials(?:\.json)?|secrets?\.(?:json|ya?ml|toml)|\.secrets|\.htpasswd|[^/]*keychain[^/]*|"
    r"\.[a-z_]*history"
)
_SECRET_NAME = re.compile(rf"(^|/)(?:{_NAME_RE_BODY})$", re.IGNORECASE)

# Same locations, matched inside a command argument such as --file=~/.ssh/config.
_DIR_RE_BODY = "|".join(re.escape(d) for d in SECRET_HOME_DIRS)
_SECRET_DIR = re.compile(rf"(?:^|[/=:,\s])(?:{_DIR_RE_BODY})(?:/|$|[\s,:])", re.IGNORECASE)
_SECRET_WORD = re.compile(rf"(?:^|[/=:,\s])(?:{_NAME_RE_BODY})(?:$|[\s,:])", re.IGNORECASE)

_GLOB_CHARS = re.compile(r"[*?\[]")
# Representative names a glob is tested against. Shell globs skip dotfiles unless the pattern starts with a dot.
_GLOB_TARGETS = sorted(
    {d.split("/")[0] for d in SECRET_HOME_DIRS}
    | {".env", ".envrc", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "x.pem", "x.key", "x.p12", "x.pfx", "credentials"}
)  # fmt: skip


def home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def secret_dirs() -> list[Path]:
    h = home()
    return [h / d for d in SECRET_HOME_DIRS]


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


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


def _glob_can_match_secret(text: str) -> bool:
    for part in re.split(r"[/=:,\s]", text):
        if part == "*" or not _GLOB_CHARS.search(part):
            continue
        for target in _GLOB_TARGETS:
            if target.startswith(".") and not part.startswith("."):
                continue
            if fnmatch.fnmatch(target.lower(), part.lower()):
                return True
    return False


def mentions_secret(text: str, may_glob: bool = True) -> bool:
    """Cheap textual check for arguments that may expand to secret paths.

    Set may_glob to False for quoted words: the shell doesn't expand them, so `.*` is a regex there, not a glob.
    """
    if _SECRET_DIR.search(text) or _SECRET_WORD.search(text):
        return True
    return may_glob and _glob_can_match_secret(text)
