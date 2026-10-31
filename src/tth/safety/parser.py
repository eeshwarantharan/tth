"""Parse shell command strings into a flat, conservative structure.

The parser walks the tree-sitter-bash AST and only accepts node types it knows.
Anything else is recorded as an *issue*; the classifier treats issues as
"not provably safe" (default deny).
"""

from __future__ import annotations

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


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tsb.language()))


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _unescape(word: str) -> str:
    out, i = [], 0
    while i < len(word):
        if word[i] == "\\" and i + 1 < len(word):
            out.append(word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return "".join(out)


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
            return _text(node, self.src)[2:-1], False
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

    def redirect(self, node: Node, cmd: SimpleCommand) -> None:
        if node.type in ("heredoc_redirect", "herestring_redirect"):
            cmd.redirects.append(Redirect(op="<<", target=None))
            for c in node.children:
                if c.type in ("command_substitution", "simple_expansion", "expansion"):
                    self.word(c, cmd)
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
                for g in c.children:
                    if g.type in ("command_substitution", "process_substitution"):
                        self.word(g, cmd)
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
                if dyn:
                    cmd.dynamic_args.add(len(cmd.argv))
                cmd.argv.append(v)
                cmd.raw.append(_text(c, self.src))
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
            return []
        self.p.issues.append(f"unsupported construct: {t}")
        return []


def parse(command: str) -> ParsedCommand:
    parsed = ParsedCommand(source=command)
    src = command.encode()
    tree = _parser().parse(src)
    root = tree.root_node
    if root.has_error:
        parsed.issues.append("syntax error")
    _Walker(src, parsed).statement(root)
    if not parsed.commands and not parsed.issues:
        parsed.issues.append("empty command")
    return parsed
