"""Allowlist command classifier.

Every simple command must match an explicit rule; anything unrecognised is
NEEDS_REVIEW (never READ_ONLY). Known-dangerous constructs are DENIED and are
never staged. Destructive commands are flagged so the rewriter can turn them
into a preview or a move-to-trash.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from enum import IntEnum

from tth.safety.parser import ParsedCommand, SimpleCommand, brace_alternatives, parse
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


SENSITIVE_PATH_REASON = "touches a secret path"
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
    "csvlook", "mdls", "getconf", "sysctl", "vm_stat", "man", "info", "look", "split",
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
    "wipefs", "parted", "sgdisk", "hdparm", "exit", "logout",
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

SHELLS = {"bash", "sh", "zsh", "fish", "dash", "ksh"}
SHELL_INTERPRETERS = SHELLS | {"python", "python3", "perl", "ruby", "node", "php", "lua"}
NETWORK_FETCHERS = {"curl", "wget", "nc", "ncat", "ftp", "telnet", "ssh", "scp", "rsync", "aria2c"}

# Commands that run another command given in their arguments.
WRAPPERS = {"env", "command", "builtin", "nohup", "nice", "time", "timeout", "exec", "stdbuf", "ionice", "setsid"}
# Flags of those wrappers that consume the next argument.
_WRAPPER_VALUE_FLAGS = {
    "env": {"-u", "-C", "-S", "--unset", "--chdir"},
    "nice": {"-n", "--adjustment"},
    "timeout": {"-k", "-s", "--kill-after", "--signal"},
    "stdbuf": {"-i", "-o", "-e"},
    "ionice": {"-c", "-n", "-p"},
    "exec": {"-a"},
}
_INLINE_CODE_RISK = re.compile(
    r"rmtree|unlink|remove|\brm\b|rmdir|truncate|system\s*\(|\bexec|popen|spawn|subprocess|child_process|"
    r"\bdelete\b|shutil|os\.|fs\.|File\.|open\s*\(.*[\"'][wax]"
)

# Read-only tools whose flags write files or run programs. Each entry: flags that make the command not read-only.
_UNSAFE_FLAGS: dict[str, tuple[str, ...]] = {
    "rg": ("--pre", "--hostname-bin"),
    "sort": ("--compress-program",),
    "less": ("-o", "-O", "--log-file", "--LOG-FILE"),
    "more": ("-o", "-O", "--log-file", "--LOG-FILE"),
    "info": ("-o", "--output"),
    "file": ("-C", "--compile"),
    "xxd": ("-r", "-revert"),
    "date": ("-s", "--set"),
    "sysctl": ("-w", "--write"),
    "ps": ("-E",),
}
# Read-only tools that treat a second positional argument as an output file. Flags listed here consume a value.
_OUTPUT_POSITIONAL: dict[str, set[str]] = {
    "uniq": {"-f", "-s", "-w", "--skip-fields", "--skip-chars", "--check-chars"},
    "xxd": {"-l", "-s", "-c", "-g", "-o", "-n", "-len", "-seek", "-cols", "-groupsize", "-offset", "-name"},
}
# Where flags can never write or execute, so a runtime-expanded argument can't turn into something dangerous.
_DYNAMIC_ARGS_SAFE = {
    "ls", "cat", "head", "tail", "wc", "echo", "printf", "stat", "basename", "dirname", "realpath", "readlink",
    "du", "df", "nl", "rev", "fold", "cut", "tr", "grep", "egrep", "fgrep", "comm", "diff", "cmp", "md5", "md5sum",
    "shasum", "sha1sum", "sha256sum", "cksum", "paste", "join", "column", "strings", "od", "tac", "expand",
    "unexpand", "test", "[", "seq", "true", "false", "cd", "which", "whereis", "type", "uname", "id", "jq",
}  # fmt: skip
_TAR_EXEC_FLAGS = ("--to-command", "--use-compress-program", "--checkpoint-action", "--rsh-command", "--rmt-command",
                   "--info-script", "--new-volume-script", "-I")  # fmt: skip
_OPAQUE_DESTRUCTIVE: dict[str, set[str]] = {
    "docker": {"rm", "rmi", "prune", "kill"},
    "kubectl": {"delete", "drain"},
    "podman": {"rm", "rmi", "prune", "kill"},
}

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
_SYSTEM_ROOT = re.compile(r"^(/|~|~/|\$HOME/?|/\*|~/\*|/[A-Za-z]+/?)$")
_TRUSTED_BIN_DIRS = {"/bin", "/usr/bin", "/usr/local/bin"}


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


def _has_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    """True for `--flag`, `--flag=x`, and the attached short form `-fvalue`."""
    for a in argv[1:]:
        for f in flags:
            if a == f or a.startswith(f + "="):
                return True
            if len(f) == 2 and a.startswith(f) and not a.startswith("--"):
                return True
    return False


def _positionals(argv: list[str], value_flags: set[str]) -> list[str]:
    out, i = [], 1
    while i < len(argv):
        a = argv[i]
        if a in value_flags:
            i += 2
            continue
        if not a.startswith("-") or a == "-":
            out.append(a)
        i += 1
    return out


def _classify_git(cmd: SimpleCommand) -> Verdict:
    args = cmd.argv[1:]
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i].startswith(("-c", "--config-env", "--exec-path", "--git-dir", "--work-tree")):
            return Verdict(NR, ["git config or repository override can execute programs"])
        if args[i] in ("-C", "--git-dir", "--work-tree"):
            i += 1
        i += 1
    if i >= len(args):
        return Verdict(RO, [])
    sub, rest = args[i], args[i + 1 :]
    if any(r.startswith("--output") or r == "--ext-diff" or r.startswith("--exec") for r in rest):
        return Verdict(NR, [f"git {sub} with output/exec option"])
    if any(r.startswith(("--upload-pack", "--receive-pack")) for r in rest) or (sub == "clone" and "-u" in rest):
        return Verdict(NR, [f"git {sub} option runs a program"])
    if sub == "stash" and rest and rest[0] in ("drop", "clear"):
        return Verdict(CF, ["git stash drop discards saved work"], destructive=True)
    if sub == "grep" and any(r.startswith(("-O", "--open-files-in-pager")) for r in rest):
        return Verdict(NR, ["git grep -O opens files in a program"])
    if sub == "reflog" and rest and rest[0] in ("expire", "delete"):
        return Verdict(DN, ["git reflog can destroy recovery points"], destructive=True)
    if sub == "branch" and any(r in ("-D", "--delete", "-d") for r in rest) and "-D" in rest:
        return Verdict(CF, ["git branch -D deletes unmerged work"], destructive=True)
    if sub == "rm" and "--cached" not in rest:
        return Verdict(CF, ["git rm deletes files"], destructive=True)
    if sub == "restore" and "--staged" not in rest:
        return Verdict(CF, ["git restore discards changes"], destructive=True)
    if sub == "checkout" and (any(r in (".", "--", "-f", "--force") for r in rest)):
        return Verdict(CF, ["git checkout discards changes"], destructive=True)
    if sub == "push" and any(r in ("-f", "--force", "--force-with-lease", "--mirror", "--delete", "-d") for r in rest):
        return Verdict(CF, ["forced or deleting push rewrites the remote"], destructive=True)
    if sub in GIT_READ:
        if sub == "branch" and any(r in ("-d", "-D", "-m", "-M", "--delete", "-f") for r in rest):
            return Verdict(CF, [f"git branch {' '.join(rest)} changes refs"])
        if sub == "branch" and not _git_branch_is_query(rest):
            return Verdict(CF, ["git branch creates a branch"])
        if sub == "stash" and (not rest or rest[0] not in ("list", "show")):
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


def _git_branch_is_query(rest: list[str]) -> bool:
    if any(
        r in ("-l", "--list", "--contains", "--no-contains", "--merged", "--no-merged", "--points-at") for r in rest
    ):
        return True
    return all(r.startswith("-") for r in rest)


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
    if v.destructive:
        v = v.worse(_find_deletion_scope(argv))
    return v


def _find_deletion_scope(argv: list[str]) -> Verdict:
    """A deletion needs a narrow target: a system path, or a whole tree with no filter, is refused."""
    paths = []
    for a in argv[1:]:
        if a.startswith(("-", "(", "!")):
            break
        paths.append(a)
    if any(_SYSTEM_ROOT.match(p) for p in paths):
        return Verdict(DN, ["find deleting from a system or home path"], destructive=True)
    actions = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-depth", "-print", "-print0", "-L", "-H", "-P"}
    filters = [a for a in argv[1:] if a.startswith("-") and a not in actions]
    if not filters and all(_ROOTISH.match(p) for p in paths):
        return Verdict(DN, ["find deleting a whole tree with no filter"], destructive=True)
    return Verdict(RO, [])


_SED_SIMPLE = set("pdqQnNgGhHxlDPzF=#")
_SED_LABEL_OR_FILE = set("bTtrRvL")
_SED_FLAGS_WITH_VALUE = {"-l", "--line-length"}
_SED_SAFE_FLAGS = set("nErsuz")
_SED_SAFE_LONG = {"--quiet", "--silent", "--regexp-extended", "--separate", "--unbuffered", "--null-data",
                  "--posix", "--sandbox"}  # fmt: skip


def _sed_skip_delimited(s: str, i: int, delim: str) -> int | None:
    """Return the index just past the next unescaped delimiter, or None if there isn't one."""
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == delim:
            return i + 1
        i += 1
    return None


def _sed_skip_address(s: str, i: int) -> int | None:
    if i < len(s) and s[i] == "$":
        return i + 1
    if i < len(s) and s[i].isdigit():
        while i < len(s) and (s[i].isdigit() or s[i] == "~"):
            i += 1
        return i
    if i < len(s) and s[i] in "/\\":
        if s[i] == "\\":
            if i + 1 >= len(s):
                return None
            delim, i = s[i + 1], i + 2
        else:
            delim, i = "/", i + 1
        end = _sed_skip_delimited(s, i, delim)
        if end is None:
            return None
        while end < len(s) and s[end] in "IM":
            end += 1
        return end
    return i


def _sed_script_is_safe(s: str) -> bool:
    """Scan a sed script command by command, and accept only commands known not to write or execute."""
    i = 0
    while i < len(s):
        if s[i] in " \t\n;{}":
            i += 1
            continue
        i2 = _sed_skip_address(s, i)
        if i2 is None:
            return False
        i = i2
        while i < len(s) and s[i] in " \t":
            i += 1
        if i < len(s) and s[i] == ",":
            i2 = _sed_skip_address(s, i + 1)
            if i2 is None:
                return False
            i = i2
        while i < len(s) and s[i] in " \t!":
            i += 1
        if i >= len(s):
            return True
        c = s[i]
        i += 1
        if c in "{}":
            continue
        if c == "#":
            while i < len(s) and s[i] != "\n":
                i += 1
        elif c in _SED_SIMPLE:
            if c in "qQlL":
                while i < len(s) and s[i].isdigit():
                    i += 1
        elif c in _SED_LABEL_OR_FILE:
            while i < len(s) and s[i] not in "\n;":
                i += 1
        elif c in "aic":
            while i < len(s) and s[i] != "\n":
                i += 1
        elif c in "sy":
            if i >= len(s):
                return False
            delim = s[i]
            first = _sed_skip_delimited(s, i + 1, delim)
            if first is None:
                return False
            second = _sed_skip_delimited(s, first, delim)
            if second is None:
                return False
            i = second
            while c == "s" and i < len(s) and s[i] not in " \t\n;}":
                if s[i] in "wWeE":
                    return False
                if s[i] not in "gpiImM0123456789":
                    return False
                i += 1
        else:
            return False  # includes w, W and e, and anything we don't recognise
    return True


def _classify_sed(cmd: SimpleCommand) -> Verdict:
    args = cmd.argv[1:]
    scripts: list[str] = []
    positionals: list[str] = []
    inplace, backup = False, False
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a in _SED_FLAGS_WITH_VALUE:
            i += 1
        elif a == "--expression":
            scripts += args[i : i + 1]
            i += 1
        elif a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
        elif a in ("-f", "--file") or a.startswith(("--file=", "-f")):
            return Verdict(NR, ["sed script from file"])
        elif a == "--in-place" or a.startswith("--in-place="):
            inplace, backup = True, "=" in a
        elif a.startswith("--"):
            if a != "--" and a not in _SED_SAFE_LONG:
                return Verdict(NR, [f"unrecognised sed flag: {a}"])
        elif a.startswith("-") and len(a) > 1:
            cluster = a[1:]
            for pos, ch in enumerate(cluster):
                if ch == "i" or ch == "I":
                    inplace, backup = True, bool(cluster[pos + 1 :])
                    break
                if ch == "e":
                    rest = cluster[pos + 1 :]
                    if rest:
                        scripts.append(rest)
                    else:
                        scripts += args[i : i + 1]
                        i += 1
                    break
                if ch == "f":
                    return Verdict(NR, ["sed script from file"])
                if ch not in _SED_SAFE_FLAGS:
                    return Verdict(NR, [f"unrecognised sed flag: -{ch}"])
        else:
            positionals.append(a)
    if inplace and not backup and positionals and positionals[0] == "":
        positionals = positionals[1:]  # BSD: sed -i '' script file
    if not scripts and positionals:
        scripts = positionals[:1]
    if not all(_sed_script_is_safe(s) for s in scripts):
        return Verdict(NR, ["sed script writes, executes or isn't recognised"])
    if inplace:
        if backup:
            return Verdict(CF, ["sed -i edits files in place, with a backup"])
        return Verdict(CF, ["sed -i edits files in place"], destructive=True)
    return Verdict(RO, [])


_AWK_RISK = re.compile(
    r"system\s*\(|getline|\|\s*\"|\|&|(?<!\|)\|(?!\|)|print[f]?[^;}]*?(>>?|\|)\s*[\"$\w(]|@load|@include|close\s*\("
)


def _classify_awk(cmd: SimpleCommand) -> Verdict:
    args = cmd.argv[1:]
    progs: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a in ("-F", "-v"):
            i += 1
        elif a.startswith(("-F", "-v")) or a == "--":
            continue
        elif a.startswith("-") and len(a) > 1:
            return Verdict(NR, [f"awk flag {a} may load code"])
        else:
            progs.append(a)
    prog = progs[0] if progs else ""
    if _AWK_RISK.search(prog):
        return Verdict(NR, ["awk program writes, pipes or executes"])
    return Verdict(RO, [])


def _wrapped_argv(name: str, argv: list[str]) -> list[str]:
    """The command a wrapper will run: `env -u X FOO=1 rm x` -> ['rm', 'x']."""
    value_flags = _WRAPPER_VALUE_FLAGS.get(name, set())
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in value_flags:
            i += 2
        elif a.startswith("-") or (name == "env" and "=" in a) or (name == "timeout" and a[:1].isdigit()):
            i += 1
        else:
            break
    return argv[i:]


def _classify_wrapper(name: str, cmd: SimpleCommand) -> Verdict:
    if name == "command" and len(cmd.argv) > 1 and cmd.argv[1] in ("-v", "-V"):
        return Verdict(RO, [])
    inner = _wrapped_argv(name, cmd.argv)
    if inner:
        iv = _classify_simple(SimpleCommand(argv=inner, raw=inner))
        if iv.tier == DN or iv.destructive:
            return Verdict(DN, [f"{name} wrapping a destructive command"], destructive=True)
    return Verdict(NR, [f"{name} is opaque (can run arbitrary code)"])


def _fetches_code(cmd: SimpleCommand) -> bool:
    return any(c.name.rsplit("/", 1)[-1] in NETWORK_FETCHERS for sub in cmd.substitutions for c in sub.commands)


def _inline_code(name: str, cmd: SimpleCommand) -> str | None:
    """The source text of `sh -c '...'`, `python -c '...'`, `perl -e '...'`, `eval ...`."""
    if name == "eval":
        return " ".join(cmd.argv[1:])
    code_flags = (
        "c"
        if name in SHELLS or name.startswith("python")
        else "eE"
        if name in ("perl", "ruby", "node", "php", "lua")
        else "e"
    )
    args = cmd.argv[1:]
    for i, a in enumerate(args):
        if a.startswith("-") and not a.startswith("--") and any(ch in a for ch in code_flags):
            return args[i + 1] if i + 1 < len(args) else ""
    return None


def _classify_interpreter(name: str, cmd: SimpleCommand) -> Verdict:
    if _fetches_code(cmd):
        return Verdict(DN, [f"{name} running code fetched from the network"], destructive=True)
    if name in ("perl", "ruby") and any(
        a.startswith("-") and not a.startswith("--") and "i" in a for a in cmd.argv[1:]
    ):
        return Verdict(NR, [f"{name} -i edits files in place with no backup"], destructive=True)
    if any(r.op.startswith("<<") for r in cmd.redirects):
        return Verdict(DN, [f"{name} running code from a heredoc can't be checked"])
    v = Verdict(NR, [f"{name} is opaque (can run arbitrary code)"])
    code = _inline_code(name, cmd)
    if not code:
        return v
    if name in SHELLS or name == "eval":
        inner = _classify_parsed(parse(code))
        if inner.tier == DN or inner.destructive:
            return Verdict(DN, [f"{name} running a destructive command"], destructive=True)
    elif _INLINE_CODE_RISK.search(code):
        return Verdict(DN, [f"inline {name} code may delete files or run programs"], destructive=True)
    return v


def _classify_read_only_tool(name: str, cmd: SimpleCommand) -> Verdict:
    argv = cmd.argv
    if name in _UNSAFE_FLAGS and _has_flag(argv, _UNSAFE_FLAGS[name]):
        return Verdict(NR, [f"{name} flag can write files or run programs"])
    if name in _OUTPUT_POSITIONAL and len(_positionals(argv, _OUTPUT_POSITIONAL[name])) >= 2:
        return Verdict(CF, [f"{name} writes to its second file argument"])
    positionals = _positionals(argv, set())
    if name == "man" and any(a.startswith("-") and a not in ("-k", "-f", "-w") for a in argv[1:]):
        return Verdict(NR, ["man flags can run a pager or browser"])
    if name == "tree" and _has_flag(argv, ("-o",)):
        return Verdict(CF, ["tree -o writes a file"])
    if name == "date" and any(not p.startswith("+") for p in positionals):
        return Verdict(NR, ["date with an argument can set the clock"])
    if name == "sysctl" and any("=" in p for p in positionals):
        return Verdict(NR, ["sysctl name=value changes the system"])
    if name in ("hostname", "arch") and positionals:
        return Verdict(NR, [f"{name} with an argument changes the system or runs a program"])
    if name == "ps" and any("e" in p for p in positionals):
        return Verdict(NR, ["ps e shows other processes' environment, which can hold secrets"])
    return Verdict(RO, [])


def _argument_names_destructive_command(cmd: SimpleCommand) -> bool:
    """`ssh host rm x`, `trap 'rm x' EXIT`, `git submodule foreach rm -rf .`: the argument is the program."""
    words = (w for a in cmd.argv[1:] for w in a.split()[:1])
    return any(w.rsplit("/", 1)[-1].lower() in DESTRUCTIVE_CMDS | DENIED_CMDS for w in words)


def _runs_a_shell(cmd: SimpleCommand) -> bool:
    name = cmd.name.rsplit("/", 1)[-1].lower()
    if name in SHELL_INTERPRETERS:
        return True
    if name in WRAPPERS or name == "xargs":
        inner = _wrapped_argv(name, cmd.argv)
        return bool(inner) and _runs_a_shell(SimpleCommand(argv=inner, raw=inner))
    return False


def _classify_substitutions(cmd: SimpleCommand) -> Verdict:
    v = Verdict(RO, [])
    for sub in cmd.substitutions:
        sv = _classify_parsed(sub)
        if sv.tier != RO:
            v = v.worse(Verdict(max(sv.tier, NR), ["command substitution is not read-only"], sv.destructive))
    return v


def _classify_simple(cmd: SimpleCommand) -> Verdict:
    if not cmd.argv:
        if cmd.assignments:
            return Verdict(NR, ["bare variable assignment"])
        return Verdict(NR, ["empty command"])
    if 0 in cmd.dynamic_args:
        return Verdict(NR, ["command name is computed at runtime"]).worse(_classify_substitutions(cmd))
    name = cmd.name
    base = name.rsplit("/", 1)[-1]
    if "/" in name and (
        ".." in name.split("/") or posixpath.dirname(posixpath.normpath(name)) not in _TRUSTED_BIN_DIRS
    ):
        return Verdict(NR, [f"runs a script or local binary: {name}"])
    name = base.lower()  # default macOS filesystems are case-insensitive, so RM runs rm
    flags = _short_flags(cmd.argv)
    v: Verdict
    if name in DENIED_CMDS or name.startswith("mkfs"):
        v = Verdict(DN, [f"{name} is never staged"])
    elif (
        name in ("chmod", "chown", "chgrp")
        and flags & {"-R", "--recursive"}
        and any(_SYSTEM_ROOT.match(a) for a in cmd.argv[1:] if not a.startswith("-"))
    ):
        v = Verdict(DN, [f"recursive {name} on a system or home path"], destructive=True)
    elif name in WRAPPERS:
        v = _classify_wrapper(name, cmd)
    elif name in ("eval", "source", ".") or name in SHELL_INTERPRETERS:
        v = _classify_interpreter(name, cmd)
    elif name == "rsync" and any(a.startswith("--delete") or a.startswith("--remove") for a in cmd.argv[1:]):
        v = Verdict(DN, ["rsync --delete removes files"], destructive=True)
    elif name in _OPAQUE_DESTRUCTIVE and _OPAQUE_DESTRUCTIVE[name] & set(cmd.argv[1:]):
        v = Verdict(DN, [f"{name} removes resources"], destructive=True)
    elif name in OPAQUE_CMDS:
        v = Verdict(NR, [f"{name} is opaque (can run arbitrary code)"])
        if name == "xargs" and any(a in DESTRUCTIVE_CMDS or a in DENIED_CMDS for a in cmd.argv[1:]):
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
        if _has_flag(cmd.argv, _TAR_EXEC_FLAGS):
            v = Verdict(NR, ["tar flag can run a program"])
        elif flags & {"-t", "--list"}:
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
    elif (name == "ln" and "-f" in flags) or (name in ("gzip", "gunzip", "unzip") and flags & {"-f", "-o", "--force"}):
        v = Verdict(CF, [f"{name} with this flag overwrites existing files"], destructive=True)
    elif name in CHANGE_CMDS:
        v = Verdict(CF, [f"{name} changes files"])
    elif name == "sort" and ("-o" in flags or any(a.startswith("--output") for a in cmd.argv)):
        v = Verdict(CF, ["sort -o writes a file"], destructive=True)
    elif name in ("less", "more", "man") and any("!" in a for a in cmd.argv[1:]):
        v = Verdict(NR, ["pager shell escape"])
    elif name in READ_ONLY_CMDS:
        v = _classify_read_only_tool(name, cmd)
    else:
        v = Verdict(NR, [f"unrecognised command: {name}"])

    if v.tier == NR and _argument_names_destructive_command(cmd):
        v = Verdict(DN, [f"{name} given a destructive command as an argument"], destructive=True)
    if any(i > 0 for i in cmd.dynamic_args) and (v.tier != RO or name not in _DYNAMIC_ARGS_SAFE):
        v = v.worse(Verdict(NR, ["a runtime-expanded argument could become a flag that writes or executes"]))
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
    v = v.worse(_classify_substitutions(cmd))
    if cmd.background:
        v = v.worse(Verdict(NR, ["runs in the background"]))
    if cmd.assignments:
        v = v.worse(Verdict(NR, ["environment assignment prefix"]))
    raw = cmd.raw or cmd.argv
    args = [(a, not any(q in src for q in "'\"\\")) for a, src in zip(cmd.argv[1:], raw[1:], strict=False)]
    args += [(r.target or "", True) for r in cmd.redirects]
    if any(mentions_secret(w, may_glob=unquoted) for a, unquoted in args for w in brace_alternatives(a)):
        v = v.worse(Verdict(NR, [SENSITIVE_PATH_REASON]))
    return v


def _is_safe_bc(p: ParsedCommand, idx: int) -> bool:
    """`... | bc` sums numbers. bc can reach a shell through `!` on some systems, so that's the only form allowed."""
    cmd = p.commands[idx]
    if cmd.name != "bc" or len(cmd.argv) > 2 or cmd.redirects or "!" in p.source:
        return False
    return any(idx in pipe[1:] for pipe in p.pipelines)


def _classify_parsed(p: ParsedCommand) -> Verdict:
    v = Verdict(RO, [])
    if p.issues:
        v = v.worse(Verdict(NR, sorted(set(p.issues))))
    for i, c in enumerate(p.commands):
        if not _is_safe_bc(p, i):
            v = v.worse(_classify_simple(c))
    for pipe in p.pipelines:
        for pos, idx in enumerate(pipe):
            if (pos > 0 or idx in p.piped_into) and _runs_a_shell(p.commands[idx]):
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
