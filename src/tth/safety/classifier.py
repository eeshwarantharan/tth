"""Allowlist command classifier.

Every simple command must match an explicit rule; anything unrecognised is
NEEDS_REVIEW (never READ_ONLY). Known-dangerous constructs are DENIED and are
never staged. Destructive commands are flagged so the rewriter can turn them
into a preview or a move-to-trash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from tth.safety.parser import ParsedCommand, SimpleCommand, parse
from tth.safety.secrets import mentions_secret


class Tier(IntEnum):
    READ_ONLY = 0
    CHANGES_FILES = 1
    NEEDS_REVIEW = 2
    DENIED = 3

    @property
    def label(self) -> str:
        return self.name.lower().replace("_", "-")


@dataclass
class Verdict:
    tier: Tier
    reasons: list[str] = field(default_factory=list)
    destructive: bool = False

    def worse(self, other: Verdict) -> Verdict:
        tier = max(self.tier, other.tier)
        return Verdict(tier, self.reasons + other.reasons, self.destructive or other.destructive)


@dataclass
class Classification:
    command: str
    tier: Tier
    reasons: list[str]
    destructive: bool
    parsed: ParsedCommand

    @property
    def safe(self) -> bool:
        return self.tier == Tier.READ_ONLY


RO, CF, NR, DN = Tier.READ_ONLY, Tier.CHANGES_FILES, Tier.NEEDS_REVIEW, Tier.DENIED

# Plain read-only tools whose flags never write or execute.
READ_ONLY_CMDS = {
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "sort", "uniq",
    "cut", "tr", "du", "df", "stat", "file", "which", "whereis", "pwd", "echo", "printf",
    "date", "basename", "dirname", "realpath", "readlink", "tree", "column", "comm", "diff",
    "cmp", "md5", "md5sum", "shasum", "sha1sum", "sha256sum", "cksum", "nl", "rev", "paste",
    "join", "less", "more", "printenv", "ps", "uname", "whoami", "id", "hostname", "uptime",
    "xxd", "od", "strings", "jq", "true", "false", "test", "[", "seq", "fold", "expand",
    "unexpand", "tac", "cd", "type", "locale", "sw_vers", "arch", "nproc", "lsof", "free",
    "top", "w", "who", "groups", "cal", "zcat", "gzcat", "bzcat", "xzcat", "yes", "fmt",
    "csvlook", "bc", "mdls", "getconf", "sysctl", "vm_stat", "man", "info", "look", "split",
}  # fmt: skip
# `split` writes files: handled below.
READ_ONLY_CMDS.discard("split")

CHANGE_CMDS = {"mkdir", "touch", "ln", "rmdir", "trash", "gzip", "gunzip", "zip", "unzip",
               "split", "tee", "cp", "mv", "chmod", "rename"}  # fmt: skip

DESTRUCTIVE_CMDS = {"rm", "shred", "truncate", "unlink", "srm"}

DENIED_CMDS = {
    "sudo", "su", "doas", "dd", "fdisk", "diskutil", "shutdown", "reboot", "halt", "poweroff",
    "launchctl", "systemctl", "crontab", "passwd", "chpass", "visudo", "csrutil", "nvram",
    "mount", "umount", "kextload", "iptables", "pfctl", "security", "osascript",
}  # fmt: skip

OPAQUE_CMDS = {
    "eval", "exec", "source", ".", "bash", "sh", "zsh", "fish", "dash", "ksh", "python",
    "python3", "perl", "ruby", "node", "deno", "bun", "php", "lua", "xargs", "make",
    "cmake", "npm", "npx", "pnpm", "yarn", "pip", "pip3", "uv", "pipx", "brew", "apt",
    "apt-get", "yum", "dnf", "pacman", "cargo", "go", "gem", "env", "nohup", "nice",
    "timeout", "time", "watch", "command", "builtin", "parallel", "docker",
    "kubectl", "ssh", "scp", "rsync", "curl", "wget", "nc", "ncat", "telnet", "ftp", "open",
    "export", "alias", "unset", "set", "trap", "kill", "killall", "pkill", "chown", "chgrp",
    "defaults",
}  # fmt: skip

SHELL_INTERPRETERS = {"bash", "sh", "zsh", "fish", "dash", "ksh", "python", "python3", "perl",
                      "ruby", "node", "php", "lua"}  # fmt: skip

GIT_READ = {"status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "shortlog",
            "describe", "ls-tree", "cat-file", "reflog", "grep", "remote", "tag", "branch",
            "config", "count-objects", "whatchanged", "show-ref", "stash"}  # fmt: skip
GIT_WRITE = {"add", "commit", "mv", "init", "fetch", "pull", "switch", "checkout", "merge",
             "rebase", "cherry-pick", "revert", "restore", "rm", "push", "clone"}  # fmt: skip
GIT_DESTRUCTIVE = {"clean", "reset", "gc", "prune", "filter-branch", "update-ref"}

FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir"}
FIND_WRITE = {"-fprint", "-fprint0", "-fprintf", "-fls"}

_SAFE_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr"}
_ROOTISH = re.compile(r"^(/|~|~/|\$HOME/?|/\*|~/\*|\*|\.\.?/?|/[A-Za-z]+/?)$")


def _short_flags(argv: list[str]) -> set[str]:
    out: set[str] = set()
    for a in argv[1:]:
        if a == "--":
            break
        if a.startswith("--"):
            out.add(a)
        elif a.startswith("-") and len(a) > 1:
            out.update("-" + ch for ch in a[1:])
    return out


def _classify_git(cmd: SimpleCommand) -> Verdict:
    args = cmd.argv[1:]
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-c" or args[i].startswith("--exec-path") or args[i] == "--config-env":
            return Verdict(NR, ["git config override can execute programs"])
        if args[i] in ("-C", "--git-dir", "--work-tree"):
            i += 1
        i += 1
    if i >= len(args):
        return Verdict(RO, [])
    sub, rest = args[i], args[i + 1 :]
    if any(r.startswith("--output") or r == "--ext-diff" or r.startswith("--exec") for r in rest):
        return Verdict(NR, [f"git {sub} with output/exec option"])
    if sub in GIT_READ:
        if sub == "branch" and any(r in ("-d", "-D", "-m", "-M", "--delete", "-f") for r in rest):
            return Verdict(CF, [f"git branch {' '.join(rest)} changes refs"])
        if sub == "stash" and rest and rest[0] not in ("list", "show"):
            return Verdict(CF, ["git stash changes the working tree"])
        if sub == "config" and not any(r in ("--get", "--list", "-l", "--get-all") for r in rest):
            return Verdict(NR, ["git config write"])
        if sub in ("remote", "tag") and rest and rest[0] not in ("-v", "show", "-l", "--list"):
            return Verdict(CF, [f"git {sub} changes refs"])
        return Verdict(RO, [])
    if sub in GIT_WRITE:
        return Verdict(CF, [f"git {sub} changes the repository"])
    if sub in GIT_DESTRUCTIVE:
        return Verdict(DN, [f"git {sub} can destroy work"], destructive=True)
    return Verdict(NR, [f"unrecognised git subcommand: {sub}"])


def _classify_find(cmd: SimpleCommand) -> Verdict:
    v = Verdict(RO, [])
    argv = cmd.argv
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-delete":
            v = v.worse(Verdict(CF, ["find -delete removes files"], destructive=True))
        elif a in FIND_WRITE:
            v = v.worse(Verdict(CF, [f"find {a} writes a file"]))
        elif a in FIND_EXEC:
            j = i + 1
            while j < len(argv) and argv[j] not in (";", "+"):
                j += 1
            inner = SimpleCommand(argv=argv[i + 1 : j] or ["?"])
            iv = _classify_simple(inner)
            if a in ("-ok", "-okdir"):
                iv = iv.worse(Verdict(NR, ["find -ok is interactive"]))
            v = v.worse(Verdict(iv.tier, [f"find {a}: " + r for r in iv.reasons], iv.destructive))
            i = j
        i += 1
    return v


def _classify_sed(cmd: SimpleCommand) -> Verdict:
    flags = _short_flags(cmd.argv)
    inplace = any(f.startswith("--in-place") for f in flags) or "-i" in flags
    scripts = [a for a in cmd.argv[1:] if not a.startswith("-")]
    script = scripts[0] if scripts else ""
    # `w`/`W` write files, `e` executes; `r`/`R` read (fine). Be conservative.
    if re.search(r"(^|[;}\n/])\s*[wWe]\b|/[gpIi0-9]*[we]\b", script):
        return Verdict(NR, ["sed script writes or executes"])
    if "-f" in flags:
        return Verdict(NR, ["sed script from file"])
    if inplace:
        return Verdict(CF, ["sed -i edits files in place"], destructive=True)
    return Verdict(RO, [])


def _classify_awk(cmd: SimpleCommand) -> Verdict:
    progs = [a for a in cmd.argv[1:] if not a.startswith("-")]
    if "-f" in cmd.argv:
        return Verdict(NR, ["awk program from file"])
    prog = progs[0] if progs else ""
    if re.search(r"system\s*\(|\|\s*getline|getline\s*<|\|\s*\"|print[f]?[^;}]*?(>>?|\|)\s*[\"$\w]",
                 prog):  # fmt: skip
        return Verdict(NR, ["awk program writes, pipes or executes"])
    return Verdict(RO, [])


def _classify_simple(cmd: SimpleCommand) -> Verdict:
    if not cmd.argv:
        if cmd.assignments:
            return Verdict(NR, ["bare variable assignment"])
        return Verdict(NR, ["empty command"])
    if 0 in cmd.dynamic_args:
        return Verdict(NR, ["command name is computed at runtime"])
    name = cmd.name
    base = name.rsplit("/", 1)[-1]
    if "/" in name and not name.startswith(("/bin/", "/usr/bin/", "/usr/local/bin/")):
        return Verdict(NR, [f"runs a script or local binary: {name}"])
    name = base
    flags = _short_flags(cmd.argv)
    v: Verdict
    if name in DENIED_CMDS:
        v = Verdict(DN, [f"{name} is never staged"])
    elif name in SHELL_INTERPRETERS or name in OPAQUE_CMDS:
        v = Verdict(NR, [f"{name} is opaque (can run arbitrary code)"])
        if name == "xargs" and any(a in DESTRUCTIVE_CMDS for a in cmd.argv[1:]):
            v = Verdict(DN, ["xargs feeding a destructive command"], destructive=True)
    elif name == "git":
        v = _classify_git(cmd)
    elif name == "find":
        v = _classify_find(cmd)
    elif name == "sed":
        v = _classify_sed(cmd)
    elif name in ("awk", "gawk", "mawk", "nawk"):
        v = _classify_awk(cmd)
    elif name == "gio":
        sub = cmd.argv[1] if len(cmd.argv) > 1 else ""
        v = {"trash": Verdict(CF, ["gio trash moves files to trash"]),
             "list": Verdict(RO, []), "info": Verdict(RO, [])}.get(
            sub, Verdict(NR, [f"unrecognised gio subcommand: {sub}"]))  # fmt: skip
    elif name == "tar":
        if flags & {"-t", "--list"}:
            v = Verdict(RO, [])
        else:
            v = Verdict(CF, ["tar creates or extracts files"])
    elif name in DESTRUCTIVE_CMDS:
        targets = [a for a in cmd.argv[1:] if not a.startswith("-")]
        if "--" in cmd.argv:
            targets = cmd.argv[cmd.argv.index("--") + 1 :]
        if any(_ROOTISH.match(t) for t in targets) or not targets:
            v = Verdict(DN, [f"{name} on a root/home/wildcard path"], destructive=True)
        else:
            v = Verdict(CF, [f"{name} deletes data"], destructive=True)
    elif name in ("mv", "cp"):
        if "-n" in flags or "--no-clobber" in flags:
            v = Verdict(CF, [f"{name} (no-clobber)"])
        else:
            v = Verdict(CF, [f"{name} may overwrite files"], destructive=True)
    elif name == "chmod":
        if flags & {"-R", "--recursive"}:
            v = Verdict(NR, ["recursive chmod"])
        else:
            v = Verdict(CF, ["chmod changes permissions"])
    elif name in CHANGE_CMDS:
        v = Verdict(CF, [f"{name} changes files"])
    elif name == "sort" and ("-o" in flags or any(a.startswith("--output") for a in cmd.argv)):
        v = Verdict(CF, ["sort -o writes a file"], destructive=True)
    elif name in ("less", "more", "man") and any("!" in a for a in cmd.argv[1:]):
        v = Verdict(NR, ["pager shell escape"])
    elif name in READ_ONLY_CMDS:
        v = Verdict(RO, [])
    else:
        v = Verdict(NR, [f"unrecognised command: {name}"])

    for i in cmd.dynamic_args:
        if i > 0 and v.tier != RO:
            v = v.worse(Verdict(NR, ["runtime-expanded argument on a write command"]))
    for r in cmd.redirects:
        if r.op in ("<", "<<", "<<<", "<&") or r.op.startswith("<<"):
            continue
        if r.op in (">&",) and (r.target or "").isdigit():
            continue
        if r.target in _SAFE_REDIRECT_TARGETS or (r.op.endswith("&") and r.target in ("1", "2")):
            continue
        if r.target is None:
            v = v.worse(Verdict(NR, ["redirect to a runtime-computed path"]))
        elif r.op in (">|", "&>|"):
            v = v.worse(Verdict(NR, ["forced clobber redirect"], destructive=True))
        elif r.op in (">>", "&>>"):
            v = v.worse(Verdict(CF, [f"appends to {r.target}"]))
        else:
            v = v.worse(Verdict(CF, [f"writes {r.target}"]))
    for sub in cmd.substitutions:
        sv = _classify_parsed(sub)
        if sv.tier != RO:
            v = v.worse(Verdict(max(sv.tier, NR), ["command substitution is not read-only"]))
    if cmd.background:
        v = v.worse(Verdict(NR, ["runs in the background"]))
    if cmd.assignments:
        v = v.worse(Verdict(NR, ["environment assignment prefix"]))
    words = cmd.argv[1:] + [r.target or "" for r in cmd.redirects]
    if any(mentions_secret(w) for w in words):
        v = v.worse(Verdict(NR, ["touches a secret path"]))
    return v


def _classify_parsed(p: ParsedCommand) -> Verdict:
    v = Verdict(RO, [])
    if p.issues:
        v = v.worse(Verdict(NR, sorted(set(p.issues))))
    for c in p.commands:
        v = v.worse(_classify_simple(c))
    for pipe in p.pipelines:
        for pos, idx in enumerate(pipe):
            if pos > 0 and p.commands[idx].name.rsplit("/", 1)[-1] in SHELL_INTERPRETERS:
                v = v.worse(Verdict(DN, ["pipes data into a shell interpreter"]))
    return v


def classify(command: str) -> Classification:
    command = command.strip()
    if not command:
        p = ParsedCommand(source="", issues=["empty command"])
        return Classification(command, NR, ["empty command"], False, p)
    if len(command) > 4000:
        p = ParsedCommand(source=command, issues=["command too long"])
        return Classification(command, NR, ["command too long"], False, p)
    p = parse(command)
    v = _classify_parsed(p)
    reasons = list(dict.fromkeys(v.reasons))
    return Classification(command, v.tier, reasons, v.destructive, p)
