"""Redact secrets before anything leaves the machine."""

from __future__ import annotations

import getpass
import math
import re
from pathlib import Path

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private-key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    ),
    ("aws-key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    (
        "api-key",
        re.compile(
            r"\b(sk-[A-Za-z0-9_\-]{16,}|nvapi-[A-Za-z0-9_\-]{16,}|xox[abpr]-[A-Za-z0-9\-]{10,}|"
            r"AIza[0-9A-Za-z_\-]{30,})"
        ),
    ),  # fmt: skip
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    ("url-credentials", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    (
        "assignment",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|"
            r"CREDENTIALS?|AUTH)[A-Z0-9_]*)\s*([=:])\s*(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
        ),
    ),  # fmt: skip
]
_TOKENISH = re.compile(r"\b[A-Za-z0-9_\-+/=]{32,}\b")


def _entropy(s: str) -> float:
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted text, list of redaction kinds applied)."""
    hits: list[str] = []

    for kind, pat in _PATTERNS:

        def sub(m: re.Match[str], kind: str = kind) -> str:
            hits.append(kind)
            if kind == "assignment":
                return f"{m.group(1)}{m.group(2)}[REDACTED]"
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
