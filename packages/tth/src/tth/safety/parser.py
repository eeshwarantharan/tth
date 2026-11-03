"""Parse shell command strings into a flat, conservative structure.

The parser walks the tree-sitter-bash AST and only accepts node types it knows.
Anything else is recorded as an *issue*; the classifier treats issues as
"not provably safe" (default deny).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import tree_sitter_bash as tsb
from tree_sitter import Language, Node, Parser

# Structural nodes we understand. Everything else becomes an issue.
_STRUCTURAL = {
    "program",
    "list",
    "pipeline",
    "redirected_statement",
    "subshell",
    "compound_statement",
    "negated_command",
}
_SEPARATORS = {"|", "|&", "&&", "||", ";", ";;", "\n", "(", ")", "{", "}", "!", "comment"}
_WORDLIKE = {
    "word",
    "string",
    "raw_string",
    "ansi_c_string",
    "concatenation",
    "simple_expansion",
    "expansion",
    "number",
    "command_substitution",
    "process_substitution",
    "arithmetic_expansion",
}
_SUBSTITUTIONS = {"command_substitution", "process_substitution"}
_HEREDOC_PARTS = {"<<", "<<-", "heredoc_start", "heredoc_end", "&&", "||", ";", "&", "|", "|&"}
_BRACE = re.compile(r"\{[^{}]*(?:,|\.\.)[^{}]*\}")
_BRACE_GROUP = re.compile(r"\{([^{}]*,[^{}]*)\}")
_SINGLE_QUOTED = re.compile(r"'[^']*'")
# `!x` runs something from history at an interactive prompt. Not `$!`, `! -name`, `!=` or `\!`.
_HISTORY = re.compile(r"(?<![\\$])!(?![\s=(\"')]|$)|^\s*\^")
# zsh features the bash grammar misreads: =cmd, =(...), ${(flags)x} and glob qualifiers such as *(e:...:).
_ZSH_ONLY = re.compile(r"(?:^|\s)=[\w(]|\$\{\([^)]*\)|[*?\]]\([^)]*[:+][^)]*\)")
_STATEMENTS = {"command", "pipeline", "redirected_statement", "list", "subshell", "compound_statement"}
_ANSI_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b",
                 "f": "\f", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?"}  # fmt: skip


@dataclass
class Redirect:
    op: str
    target: str | None  # None when the target is not a literal word
    fd: str | None = None


@dataclass
class SimpleCommand:
    argv: list[str]
    raw: list[str] = field(default_factory=list)  # original source text of each argv word
    name_end: int = -1  # byte offset just after the command name in the source
    dynamic_args: set[int] = field(default_factory=set)  # argv indexes with expansions
    redirects: list[Redirect] = field(default_factory=list)
    assignments: list[str] = field(default_factory=list)
    substitutions: list[ParsedCommand] = field(default_factory=list)
    background: bool = False

    @property
    def name(self) -> str:
        return self.argv[0] if self.argv else ""


@dataclass
class ParsedCommand:
    source: str
    commands: list[SimpleCommand] = field(default_factory=list)
    pipelines: list[list[int]] = field(default_factory=list)  # indexes into commands
    issues: list[str] = field(default_factory=list)
    piped_into: set[int] = field(default_factory=set)  # commands fed by a pipe outside `pipelines`


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tsb.language()))


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _unescape(word: str) -> str:
    out, i = [], 0
    while i < len(word):
        if word[i] == "\\" and i + 1 < len(word):
            if word[i + 1] != "\n":  # backslash-newline joins two lines and adds nothing
                out.append(word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return "".join(out)


def has_brace_expansion(raw: str) -> bool:
    """True for an unquoted {a,b} or {1..3}, which the shell expands into several words."""
    return not any(q in raw for q in "'\"") and bool(_BRACE.search(raw))


def brace_alternatives(word: str, limit: int = 64) -> list[str]:
    """Every word `a{b,c}d` expands to, so each one can be checked."""
    words, done = [word], []
    while words and len(done) + len(words) <= limit:
        w = words.pop()
        m = _BRACE_GROUP.search(w)
        if not m:
            done.append(w)
            continue
        words += [w[: m.start()] + opt + w[m.end() :] for opt in m.group(1).split(",")]
    return done + words


def _decode_ansi_c(body: str) -> str:
    """Decode the escapes of a $'...' string, so the classifier sees what the shell will run."""

    def sub(m: re.Match[str]) -> str:
        esc = m.group(1)
        if esc[0] in "xuU" and len(esc) > 1:
            return chr(int(esc[1:], 16))
        if esc[0].isdigit():
            return chr(int(esc, 8) & 0xFF)
        return _ANSI_ESCAPES.get(esc, esc)

    return re.sub(r"\\(x[0-9A-Fa-f]{1,2}|u[0-9A-Fa-f]{1,4}|U[0-9A-Fa-f]{1,8}|[0-7]{1,3}|.)", sub, body, flags=re.S)


class _Walker:
    def __init__(self, src: bytes, parsed: ParsedCommand) -> None:
        self.src = src
        self.p = parsed

    def word(self, node: Node, cmd: SimpleCommand | None) -> tuple[str, bool]:
        """Return (literal value, is_dynamic)."""
        t = node.type
        if t == "word" or t == "number":
            return _unescape(_text(node, self.src)), False
        if t == "raw_string":
            return _text(node, self.src)[1:-1], False
        if t == "ansi_c_string":
            body = _text(node, self.src)[2:-1]
            # Escapes are decoded for the checks, and marked dynamic because they exist to obscure a word.
            return _decode_ansi_c(body), "\\" in body
        if t == "string":
            parts, dyn = [], False
            for c in node.children:
                if c.type == '"':
                    continue
                if c.type == "string_content":
                    parts.append(_text(c, self.src).replace('\\"', '"'))
                else:
                    v, d = self.word(c, cmd)
                    parts.append(v)
                    dyn = dyn or d
            return "".join(parts), dyn
        if t == "concatenation":
            parts, dyn = [], False
            for c in node.children:
                v, d = self.word(c, cmd)
                parts.append(v)
                dyn = dyn or d
            return "".join(parts), dyn
        if t in ("simple_expansion", "expansion", "arithmetic_expansion"):
            self.nested_substitutions(node, cmd)
            return _text(node, self.src), True
        if t in ("command_substitution", "process_substitution"):
            inner = ParsedCommand(source=_text(node, self.src))
            sub = _Walker(self.src, inner)
            for c in node.children:
                if c.type not in ("$(", ")", "`", "<(", ">(", "$("):
                    sub.statement(c)
            if t == "process_substitution" and _text(node, self.src).startswith(">("):
                inner.issues.append("output process substitution")
            self.p.issues.extend(inner.issues)
            if cmd is not None:
                cmd.substitutions.append(inner)
            else:
                self.p.issues.append("substitution outside a command")
            return _text(node, self.src), True
        self.p.issues.append(f"unsupported word node: {t}")
        return _text(node, self.src), True

    def nested_substitutions(self, node: Node, cmd: SimpleCommand | None) -> None:
        """Substitutions can sit anywhere inside an expansion, a string or a heredoc body."""
        for c in node.children:
            if c.type in _SUBSTITUTIONS:
                self.word(c, cmd)
            else:
                self.nested_substitutions(c, cmd)

    def nested_commands(self, node: Node) -> None:
        """Commands inside a construct we don't model still get classified."""
        for c in node.children:
            if c.type in _STATEMENTS:
                self.statement(c)
            else:
                self.nested_commands(c)

    def heredoc(self, node: Node, cmd: SimpleCommand) -> None:
        """A heredoc redirect holds its body, and anything after `<<EOF` on the same line."""
        cmd.redirects.append(Redirect(op="<<", target=None))
        for c in node.children:
            if c.type in _HEREDOC_PARTS:
                continue
            if c.type == "heredoc_body":
                self.nested_substitutions(c, cmd)
                continue
            got = self.statement(c)
            if c.type == "pipeline" and got:
                self.p.piped_into.add(got[0])  # `cat <<EOF | sh`: the heredoc is the left side of the pipe

    def redirect(self, node: Node, cmd: SimpleCommand) -> None:
        if node.type == "heredoc_redirect":
            self.heredoc(node, cmd)
            return
        if node.type == "herestring_redirect":
            cmd.redirects.append(Redirect(op="<<", target=None))
            self.nested_substitutions(node, cmd)
            return
        if node.type != "file_redirect":
            self.p.issues.append(f"unsupported redirect: {node.type}")
            return
        op, target, fd = "", None, None
        for c in node.children:
            if c.type == "file_descriptor":
                fd = _text(c, self.src)
            elif c.type in _WORDLIKE:
                v, dyn = self.word(c, cmd)
                target = None if dyn else v
            else:
                op += _text(c, self.src)
        cmd.redirects.append(Redirect(op=op, target=target, fd=fd))

    def command(self, node: Node) -> SimpleCommand:
        cmd = SimpleCommand(argv=[])
        for c in node.children:
            if c.type == "variable_assignment":
                cmd.assignments.append(_text(c, self.src))
                self.nested_substitutions(c, cmd)
            elif c.type == "command_name":
                inner = c.children[0] if c.children else c
                v, dyn = self.word(inner, cmd)
                if dyn:
                    cmd.dynamic_args.add(0)
                cmd.argv.append(v)
                cmd.raw.append(_text(inner, self.src))
                cmd.name_end = c.end_byte
            elif c.type in _WORDLIKE:
                v, dyn = self.word(c, cmd)
                raw = _text(c, self.src)
                if dyn or has_brace_expansion(raw):
                    cmd.dynamic_args.add(len(cmd.argv))
                cmd.argv.append(v)
                cmd.raw.append(raw)
            elif c.type in ("file_redirect", "heredoc_redirect", "herestring_redirect"):
                self.redirect(c, cmd)
            else:
                self.p.issues.append(f"unsupported command part: {c.type}")
        return cmd

    def statement(self, node: Node, redirects: list[Node] | None = None) -> list[int]:
        """Walk a statement; return indexes of simple commands it produced."""
        t = node.type
        if t == "command":
            cmd = self.command(node)
            for r in redirects or []:
                self.redirect(r, cmd)
            self.p.commands.append(cmd)
            return [len(self.p.commands) - 1]
        if t == "redirected_statement":
            body, reds = None, []
            for c in node.children:
                if c.type in ("file_redirect", "heredoc_redirect", "herestring_redirect"):
                    reds.append(c)
                elif body is None:
                    body = c
            if body is None:
                self.p.issues.append("redirect without command")
                return []
            idx = self.statement(body, (redirects or []) + reds)
            if body.type != "command" and reds:
                # Redirect applied to a compound statement; attach to every command.
                for i in idx:
                    for r in reds:
                        self.redirect(r, self.p.commands[i])
            return idx
        if t == "pipeline":
            idx: list[int] = []
            for c in node.children:
                if c.type in _SEPARATORS:
                    continue
                idx.extend(self.statement(c))
            self.p.pipelines.append(idx)
            return idx
        if t in _STRUCTURAL:
            idx = []
            kids = node.children
            for i, c in enumerate(kids):
                if c.type == "comment":
                    # An interactive zsh runs what follows `#` unless interactive_comments is set.
                    self.p.issues.append("comment: zsh would run this text")
                    continue
                if c.type in _SEPARATORS:
                    continue
                if c.type == "&":
                    continue
                got = self.statement(c)
                if i + 1 < len(kids) and kids[i + 1].type == "&":
                    for j in got:
                        self.p.commands[j].background = True
                idx.extend(got)
            return idx
        if t in _SEPARATORS or t == "&":
            return []
        if t == "ERROR" or node.is_missing:
            self.p.issues.append("syntax error")
            self.nested_commands(node)
            return []
        self.p.issues.append(f"unsupported construct: {t}")
        self.nested_commands(node)
        return []


def parse(command: str) -> ParsedCommand:
    parsed = ParsedCommand(source=command)
    if "\\\n" in command:
        # The shell joins these lines into one word, and the grammar doesn't. Classify what the shell will run.
        parsed.issues.append("line continuation")
        command = command.replace("\\\n", "")
    src = command.encode()
    tree = _parser().parse(src)
    root = tree.root_node
    if root.has_error:
        parsed.issues.append("syntax error")
    if _HISTORY.search(_SINGLE_QUOTED.sub("", command)):
        parsed.issues.append("history expansion")
    if _ZSH_ONLY.search(command):
        parsed.issues.append("zsh-only syntax")
    _Walker(src, parsed).statement(root)
    if not parsed.commands and not parsed.issues:
        parsed.issues.append("empty command")
    return parsed
