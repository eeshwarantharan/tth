"""Rewrite destructive commands into safer staged forms.

- Deletions become a move-to-trash, with a preview command that lists matches.
- mv/cp get no-clobber.
- sed -i keeps a .bak backup.
- Redirects that would overwrite an existing file are refused.
Every rewrite is reported in `notes` so the user sees it; nothing is silent.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tth.safety.classifier import Classification, Tier, classify
from tth.safety.parser import SimpleCommand

_RISKY_TEXT = re.compile(r"\b(rm|unlink|shred|truncate|dd|mv|cp|sudo|chmod|chown|mkfs|sed\s+-i)\b|>")


def trash_argv() -> list[str]:
    return ["trash"] if sys.platform == "darwin" else ["gio", "trash"]


@dataclass
class Rewrite:
    original: str
    staged: str | None  # None means: refuse to stage
    tier: Tier
    preview: str | None = None  # read-only command listing what would be affected
    notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.staged is not None and self.staged != self.original


def _join(words: list[str]) -> str:
    """Words are already shell source text (original quoting kept), so globs still expand."""
    return " ".join(words)


def _rewrite_rm(cmd: SimpleCommand) -> tuple[list[str], list[str]]:
    pairs = list(zip(cmd.argv[1:], cmd.raw[1:], strict=True))
    if "--" in cmd.argv:
        pairs = pairs[cmd.argv.index("--") :]
        targets = [r for _, r in pairs]
    else:
        targets = [r for a, r in pairs if not a.startswith("-") or a == "-"]
    dash = [t for t in targets if t.startswith("-")]
    targets = [f"./{t}" if t in dash else t for t in targets]
    staged = [*trash_argv(), *targets]
    flags = {ch for a in cmd.argv[1:] if a.startswith("-") and not a.startswith("--") for ch in a[1:]}
    if flags & {"r", "R"} or "--recursive" in cmd.argv:
        preview = ["find", *targets, "-print"]
    else:
        preview = ["ls", "-ld", "--", *targets]
    return staged, preview


def _rewrite_find(cmd: SimpleCommand) -> tuple[list[str], list[str]] | None:
    argv, raw = cmd.argv, cmd.raw
    out: list[str] = []
    i = 0
    deleting = False
    while i < len(argv):
        a = argv[i]
        if a == "-delete":
            deleting = True
            i += 1
            continue
        if a in ("-exec", "-execdir"):
            j = i + 1
            while j < len(argv) and argv[j] not in (";", "+"):
                j += 1
            inner = argv[i + 1 : j]
            if inner and inner[0] in ("rm", "unlink"):
                deleting = True
                i = j + 1
                continue
            out.extend(raw[i : j + 1])
            i = j + 1
            continue
        out.append(raw[i])
        i += 1
    if not deleting:
        return None
    preview = [*out, "-print"]
    staged = [*out, "-exec", *trash_argv(), "{}", "+"]
    return staged, preview


def _rewrite_simple(cmd: SimpleCommand) -> tuple[list[str], list[str] | None, str] | None:
    name = cmd.name.rsplit("/", 1)[-1]
    if name in ("rm", "unlink"):
        staged, preview = _rewrite_rm(cmd)
        return staged, preview, "deletion rewritten as move-to-trash"
    if name == "find":
        r = _rewrite_find(cmd)
        if r:
            return r[0], r[1], "find deletion rewritten as move-to-trash"
        return None
    if name in ("mv", "cp"):
        argv = []
        for a, r in zip(cmd.argv, cmd.raw, strict=True):
            if a in ("-f", "--force"):
                continue
            if a.startswith("-") and not a.startswith("--") and len(a) > 1:
                a = a.replace("f", "")
                if a == "-":
                    continue
                argv.append(a)
            else:
                argv.append(r)
        if "-n" not in argv:
            argv = [argv[0], "-n", *argv[1:]]
        return argv, None, f"{name} given no-clobber (-n)"
    if name == "sed":
        argv = []
        skip_empty = False
        for a, r in zip(cmd.argv, cmd.raw, strict=True):
            if a in ("-i", "--in-place"):
                argv.append("-i.bak")
                skip_empty = True
                continue
            if skip_empty and a == "":
                skip_empty = False
                continue
            skip_empty = False
            argv.append("-i.bak" if a.startswith("-i") and not a.startswith("-i.") else r)
        return argv, None, "sed -i keeps a .bak backup"
    if name == "sort":
        return None
    return None


def rewrite(command: str, cwd: str | os.PathLike[str] | None = None) -> Rewrite:
    c: Classification = classify(command)
    rw = Rewrite(original=command, staged=command, tier=c.tier, reasons=c.reasons)
    if c.tier == Tier.DENIED:
        rw.staged = None
        rw.notes.append("refused: " + "; ".join(c.reasons))
        return rw

    # Constructs the parser cannot see inside (loops, functions) are refused
    # outright when they mention anything that could destroy data.
    if c.parsed.issues and _RISKY_TEXT.search(command):
        rw.staged = None
        rw.notes.append("refused: unverifiable shell construct around a write/delete")
        return rw

    # Redirects that clobber an existing file are destructive.
    base = Path(cwd) if cwd else Path.cwd()
    for sc in c.parsed.commands:
        for r in sc.redirects:
            if r.op in (">", "&>") and r.target and r.target not in ("/dev/null",):
                target = Path(os.path.expanduser(r.target))
                if not target.is_absolute():
                    target = base / target
                if target.exists():
                    rw.staged = None
                    rw.notes.append(f"refused: '>' would overwrite existing file {r.target}")
                    return rw

    if not c.destructive:
        return rw

    if len(c.parsed.commands) != 1 and not c.parsed.issues:
        spliced = _splice_no_clobber(command, c)
        if spliced is not None:
            rw.staged = spliced
            rw.tier = classify(spliced).tier
            rw.notes.append("mv/cp given no-clobber (-n)")
            return rw

    if len(c.parsed.commands) != 1 or c.parsed.issues or c.parsed.commands[0].redirects:
        rw.staged = None
        rw.notes.append("refused: destructive step inside a compound command; split it up")
        return rw

    sc = c.parsed.commands[0]
    if sc.dynamic_args:
        rw.staged = None
        rw.notes.append("refused: destructive command with runtime-expanded arguments")
        return rw
    result = _rewrite_simple(sc)
    if result is None:
        rw.staged = None
        rw.notes.append("refused: no safe rewrite for this destructive command")
        return rw
    staged_argv, preview_argv, note = result
    staged = _join(staged_argv)
    rc = classify(staged)
    if rc.tier >= Tier.NEEDS_REVIEW and rc.tier > c.tier:
        rw.staged = None
        rw.notes.append("refused: rewrite did not classify cleanly")
        return rw
    rw.staged = staged
    rw.tier = max(rc.tier, Tier.CHANGES_FILES)
    rw.preview = _join(preview_argv) if preview_argv else None
    rw.notes.append(note)
    return rw


def _splice_no_clobber(command: str, c: Classification) -> str | None:
    """In a compound command whose only destructive steps are mv/cp, insert -n after each."""
    from tth.safety.classifier import _classify_simple

    inserts: list[int] = []
    for sc in c.parsed.commands:
        if not _classify_simple(sc).destructive:
            continue
        name = sc.name.rsplit("/", 1)[-1]
        if name not in ("mv", "cp") or sc.name_end < 0:
            return None
        if any(a in ("-f", "--force") or (a.startswith("-") and not a.startswith("--") and "f" in a)
               for a in sc.argv[1:]):  # fmt: skip
            return None
        inserts.append(sc.name_end)
    if not inserts:
        return None
    src = command.encode()
    for pos in sorted(inserts, reverse=True):
        src = src[:pos] + b" -n" + src[pos:]
    out = src.decode()
    rc = classify(out)
    return out if not rc.destructive and rc.tier <= Tier.CHANGES_FILES else None


def selection_part(command: str) -> str | None:
    """The read-only part of a command, used by dry_run.

    READ_ONLY commands are their own selection. For deletions we return the
    preview listing. Other write commands have no selection part (None).
    """
    c = classify(command)
    if c.tier == Tier.READ_ONLY:
        return command
    if c.destructive and c.tier == Tier.CHANGES_FILES:
        rw = rewrite(command)
        if rw.preview and classify(rw.preview).tier == Tier.READ_ONLY:
            return rw.preview
    return None
