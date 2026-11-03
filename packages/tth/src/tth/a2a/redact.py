"""Redact secrets before anything leaves the machine."""

from __future__ import annotations

import getpass
import math
import os
import re
from pathlib import Path

_SENSITIVE_KEY_RE = (
    r"[A-Za-z0-9_\-.]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_\-]?KEY|ACCESS[_\-]?KEY|PRIVATE[_\-]?KEY|CREDENTIALS?|"
    r"(?<![A-Za-z])PASS(?![A-Za-z])|_PWD(?![A-Za-z])|(?<![A-Za-z])AUTH(?:ORIZATION)?(?![A-Za-z]))[A-Za-z0-9_\-.]*"
)
_ASSIGNED_VALUE_RE = r"""\\"(?:[^"\\]|\\[^"])*\\"|"[^"]*"|'[^']*'|[^\s,;}]+"""

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private-key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S),
    ),
    ("aws-key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    (
        "api-key",
        re.compile(
            r"\b(sk-[A-Za-z0-9_\-]{16,}|[sr]k_(?:live|test)_[A-Za-z0-9]{16,}|nvapi-[A-Za-z0-9_\-]{16,}|"
            r"xox[abpr]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,}|glpat-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|"
            r"npm_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_\-]{16,})"
        ),
    ),  # fmt: skip
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    ("url-credentials", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    ("auth-header", re.compile(r"(?i)\b(?:proxy-)?(?:authorization|cookie|set-cookie|x-api-key)\s*[:=]\s*[^\n'\"]+")),
    ("bearer", re.compile(r"(?i)\bbearer\s+(?=\S*\d|\S{16,})[^\s'\"]+")),
    ("assignment", re.compile(rf"(?i)(\\?[\"']?)({_SENSITIVE_KEY_RE})(\\?[\"']?)\s*([=:])\s*({_ASSIGNED_VALUE_RE})")),
    # 32 and 64 hex are key or digest shaped. 40 is left alone because git hashes are useful and not secret.
    ("hex-secret", re.compile(r"\b(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{32})\b")),
]
_TOKENISH = re.compile(r"\b[A-Za-z0-9_\-+/=]{32,}\b")
_SECRET_ENV_NAME = re.compile(rf"(?i)^{_SENSITIVE_KEY_RE}$")
_MIN_ENV_VALUE = 8  # shorter values are too likely to appear in ordinary text
_SMALL_NUMBER = re.compile(r"^[\"'\\]*\d{1,9}[\"'\\]*$")  # `tokens: 1500` is a count, not a secret


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def _redact_env_values(text: str, hits: list[str]) -> str:
    for name, value in os.environ.items():
        if len(value) >= _MIN_ENV_VALUE and _SECRET_ENV_NAME.match(name) and value in text:
            hits.append("env-value")
            text = text.replace(value, "[REDACTED:env-value]")
    return text


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted text, list of redaction kinds applied)."""
    hits: list[str] = []
    text = _redact_env_values(text, hits)

    for kind, pat in _PATTERNS:

        def sub(m: re.Match[str], kind: str = kind) -> str:
            if kind == "assignment":
                if _SMALL_NUMBER.match(m.group(5)) or m.group(5) == "[REDACTED]" or m.group(5).startswith("[REDACTED:"):
                    return m.group(0)
                hits.append(kind)
                return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}[REDACTED]"
            hits.append(kind)
            if kind == "auth-header":
                return m.group(0).split(":")[0].split("=")[0] + ": [REDACTED]"
            return f"[REDACTED:{kind}]"

        text = pat.sub(sub, text)

    def high_entropy(m: re.Match[str]) -> str:
        s = m.group(0)
        if _entropy(s) > 4.0 and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
            hits.append("high-entropy")
            return "[REDACTED:high-entropy]"
        return s

    text = _TOKENISH.sub(high_entropy, text)
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    try:
        user = getpass.getuser()
        if len(user) >= 3:
            text = re.sub(rf"\b{re.escape(user)}\b", "[user]", text)
    except OSError:
        pass
    return text, hits
