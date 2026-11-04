"""Data pack: the model says what it wants, and deterministic code writes the command.

Writing awk or jq is where small models fail most. Their quoting, column positions and header handling all have to
be right, and in the bake-off every model scored zero on these tasks. So the model fills in slots, and this module
reads the file's real header and compiles a command it can test. Names are matched without regard to case, because
models write "North" for a value stored as "north".

The compiled commands are ordinary read-only shell, shown to the user like any other.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass

OPS = ("sum", "avg", "count", "min", "max", "list")
COMPARISONS = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?$")
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_ .-]*$")


class CompileError(ValueError):
    """The request can't be compiled. The message tells the model how to fix it."""


@dataclass(frozen=True)
class Query:
    path: str
    op: str
    column: str = ""  # a column, or an expression such as "units * price"
    where_column: str = ""
    where_op: str = "eq"
    where_value: str = ""


def _check_op(q: Query) -> None:
    if q.op not in OPS:
        raise CompileError(f"op must be one of: {', '.join(OPS)}")
    if q.op != "count" and not q.column:
        raise CompileError(f"{q.op} needs a column")
    if q.where_column and q.where_op not in COMPARISONS:
        raise CompileError(f"where_op must be one of: {', '.join(COMPARISONS)}")
    if q.where_column and not q.where_value:
        raise CompileError("where_column needs a where_value")
    if any(ord(c) < 32 or ord(c) == 127 for c in q.where_value):
        raise CompileError("where_value can't contain control characters")


def _resolve(name: str, columns: list[str]) -> str:
    """The real column for a name the model gave, ignoring case."""
    for column in columns:
        if column.lower() == name.strip().lower():
            return column
    raise CompileError(f"there's no column '{name}'. The columns are: {', '.join(columns)}")


def _parts(expression: str, columns: list[str]) -> list[str | tuple[str]]:
    """Split "units * price" into column tuples and literal operators, so each language can write them its own way."""
    parts: list[str | tuple[str]] = []
    pos = 0
    for m in _IDENTIFIER.finditer(expression):
        parts.append(expression[pos : m.start()])
        parts.append((_resolve(m.group(0), columns),))
        pos = m.end()
    parts.append(expression[pos:])
    leftover = "".join(p for p in parts if isinstance(p, str))
    if not re.fullmatch(r"[\s\d.+\-*/()]*", leftover):
        raise CompileError(f"column can be a name or simple arithmetic, not '{expression}'")
    return parts


# ---- CSV, through awk -----------------------------------------------------------------------------------


def _awk_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').lower() + '"'


def compile_csv(q: Query, header: str) -> str:
    _check_op(q)
    columns = [c.strip() for c in header.strip().split(",")]
    field = {c: f"${i + 1}" for i, c in enumerate(columns)}

    def expression(text: str) -> str:
        return "".join(p if isinstance(p, str) else field[p[0]] for p in _parts(text, columns))

    conditions = ["NR>1"]
    if q.where_column:
        target = field[_resolve(q.where_column, columns)]
        comparison = COMPARISONS[q.where_op]
        if _NUMBER.match(q.where_value):
            conditions.append(f"{target}+0 {comparison} {q.where_value}")
        else:
            conditions.append(f"tolower({target}) {comparison} {_awk_string(q.where_value)}")
    match = " && ".join(conditions)
    value = expression(q.column) if q.column else ""
    bodies = {
        "sum": (f"s+=({value})", "print s+0"),
        "avg": (f"s+=({value}); n++", 'if (n) print s/n; else print "no matching rows"'),
        "count": ("n++", "print n+0"),
        "min": (f"v=({value}); if (!seen || v<m) {{m=v; seen=1}}", "print m"),
        "max": (f"v=({value}); if (!seen || v>m) {{m=v; seen=1}}", "print m"),
    }
    if q.op == "list":
        return f"awk -F, {shlex.quote(f'{match} {{print {value}}}')} {shlex.quote(q.path)}"
    step, finish = bodies[q.op]
    return f"awk -F, {shlex.quote(f'{match} {{{step}}} END {{{finish}}}')} {shlex.quote(q.path)}"


# ---- JSON and JSONL, through jq -------------------------------------------------------------------------


def _jq_field(column: str) -> str:
    if not _SAFE_NAME.match(column):
        raise CompileError(f"'{column}' isn't a usable field name")
    return "." + column if _IDENTIFIER.fullmatch(column) else f'.["{column}"]'


def compile_json(q: Query, keys: list[str], lines: bool) -> str:
    _check_op(q)

    def expression(text: str) -> str:
        return "".join(p if isinstance(p, str) else _jq_field(p[0]) for p in _parts(text, keys))

    selects = ""
    if q.where_column:
        target = _jq_field(_resolve(q.where_column, keys))
        comparison = COMPARISONS[q.where_op]
        if _NUMBER.match(q.where_value):
            selects = f" | map(select({target} {comparison} {q.where_value}))"
        else:
            literal = json.dumps(q.where_value.lower())
            selects = f" | map(select(({target} | tostring | ascii_downcase) {comparison} {literal}))"
    value = expression(q.column) if q.column else ""
    pipelines = {
        "sum": f"map({value}) | add // 0",
        "avg": f"map({value}) | if length > 0 then add / length else empty end",
        "count": "length",
        "min": f"map({value}) | min",
        "max": f"map({value}) | max",
        "list": f".[] | {value}",
    }
    program = f".{selects} | {pipelines[q.op]}"
    flags = "-rs" if lines else "-r"
    return f"jq {flags} {shlex.quote(program)} {shlex.quote(q.path)}"


def keys_of(record: object, path: str) -> list[str]:
    if not isinstance(record, dict) or not record:
        raise CompileError(f"{path} doesn't hold objects with named fields")
    return list(record)
