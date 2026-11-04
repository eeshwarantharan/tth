"""Compiler tools. Every compiled command is really run in the sandbox and checked against Python's answer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tth.mcp.client import ToolSession, connect
from tth.packs.data import CompileError, Query, compile_csv, compile_json
from tth.safety.classifier import Tier, classify

SALES = [
    ("2025-01-01", "north", "widget", 10, 2.5),
    ("2025-01-02", "south", "widget", 20, 2.5),
    ("2025-01-03", "north", "sprocket", 5, 10.0),
    ("2025-01-04", "east", "sprocket", 7, 10.0),
]
METRICS = [
    {"task": "a", "status": "failed", "tokens": 100, "cost": 0.5, "model": "large"},
    {"task": "b", "status": "ok", "tokens": 50, "cost": 0.25, "model": "small"},
    {"task": "c", "status": "failed", "tokens": 7, "cost": 0.125, "model": "small"},
]
USERS = [{"name": "kofi", "age": 40, "city": "Chennai"}, {"name": "hana", "age": 25, "city": "Oslo"},
         {"name": "omar", "age": 61, "city": "Chennai"}]  # fmt: skip


@pytest.fixture
def session(tmp_path: Path):
    lines = ["date,region,product,units,price"] + [",".join(map(str, row)) for row in SALES]
    (tmp_path / "sales.csv").write_text("\n".join(lines) + "\n")
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in METRICS) + "\n")
    (tmp_path / "users.json").write_text(json.dumps(USERS))
    with connect(tmp_path, shell="zsh") as s:
        yield s


def ask(session: ToolSession, **args: object) -> dict:
    r = session.call("aggregate", args)
    assert r.ok, r.data
    return r.data


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"file": "sales.csv", "op": "sum", "column": "units", "where_column": "region", "where_value": "north"}, "15"),
        ({"file": "sales.csv", "op": "sum", "column": "units * price", "where_column": "product", "where_value": "sprocket"}, "120"),
        ({"file": "sales.csv", "op": "avg", "column": "units", "where_column": "region", "where_value": "north"}, "7.5"),
        ({"file": "sales.csv", "op": "count", "where_column": "region", "where_value": "north"}, "2"),
        ({"file": "sales.csv", "op": "count"}, "4"),
        ({"file": "sales.csv", "op": "max", "column": "units"}, "20"),
        ({"file": "sales.csv", "op": "min", "column": "units"}, "5"),
        ({"file": "sales.csv", "op": "sum", "column": "UNITS", "where_column": "Region", "where_value": "NORTH"}, "15"),
        ({"file": "sales.csv", "op": "count", "where_column": "units", "where_op": "gt", "where_value": "7"}, "2"),
        ({"file": "metrics.jsonl", "op": "sum", "column": "tokens", "where_column": "status", "where_value": "failed"}, "107"),
        ({"file": "metrics.jsonl", "op": "sum", "column": "cost", "where_column": "status", "where_value": "FAILED"}, "0.625"),
        ({"file": "metrics.jsonl", "op": "count", "where_column": "status", "where_value": "failed"}, "2"),
        ({"file": "metrics.jsonl", "op": "avg", "column": "tokens", "where_column": "model", "where_value": "small"}, "28.5"),
        ({"file": "users.json", "op": "avg", "column": "age"}, "42"),
        ({"file": "users.json", "op": "count", "where_column": "city", "where_value": "chennai"}, "2"),
    ],
)  # fmt: skip
def test_compiled_commands_give_the_right_answer(session: ToolSession, args: dict, expected: str) -> None:
    data = ask(session, **args)
    assert data["result"] == [expected], data


def test_listing_rows_that_match(session: ToolSession) -> None:
    data = ask(
        session, file="users.json", op="list", column="name", where_column="age", where_op="gt", where_value="30"
    )
    assert data["result"] == ["kofi", "omar"]


def test_no_matching_rows_is_handled(session: ToolSession) -> None:
    data = ask(session, file="sales.csv", op="sum", column="units", where_column="region", where_value="mars")
    assert data["result"] == ["0"]


def test_a_wrong_column_tells_the_model_what_exists(session: ToolSession) -> None:
    r = session.call("aggregate", {"file": "sales.csv", "op": "sum", "column": "revenue"})
    assert not r.ok
    assert "date, region, product, units, price" in r.data["error"]


def test_a_wrong_key_tells_the_model_what_exists(session: ToolSession) -> None:
    r = session.call("aggregate", {"file": "metrics.jsonl", "op": "sum", "column": "bytes"})
    assert not r.ok
    assert "tokens" in r.data["error"]


def test_a_missing_file_is_a_clean_error(session: ToolSession) -> None:
    r = session.call("aggregate", {"file": "nope.csv", "op": "count"})
    assert not r.ok


def test_secret_files_are_refused(session: ToolSession, tmp_path: Path) -> None:
    (tmp_path / ".env.csv").write_text("a,b\n1,2\n")
    r = session.call("aggregate", {"file": ".env.csv", "op": "count"})
    assert not r.ok
    assert "secret" in r.data["error"]


@pytest.mark.parametrize(
    "command",
    [
        compile_csv(Query("sales.csv", "sum", "units", "region", "eq", "north"), "date,region,units"),
        compile_json(Query("m.jsonl", "sum", "tokens", "status", "eq", "failed"), ["tokens", "status"], lines=True),
    ],
)
def test_compiled_commands_are_read_only(command: str) -> None:
    assert classify(command).tier == Tier.READ_ONLY, command


INJECTIONS = [
    "x' ; echo PWNED ; '",
    'x"; system("echo PWNED"); "',
    'x" {system("echo PWNED")} "',
    "$(echo PWNED)",
    "`echo PWNED`",
    "x' | sh #",
    '"; "PWNED" #',
    '" or true) | "PWNED" #',
    "\\",
    '\\"; system("echo PWNED"); "',
]


@pytest.mark.parametrize("value", INJECTIONS)
@pytest.mark.parametrize(("file", "column"), [("sales.csv", "region"), ("metrics.jsonl", "status")])
def test_a_hostile_value_never_runs(session: ToolSession, file: str, column: str, value: str) -> None:
    """Slot values come from the model. The sandbox would stop `rm`, so this payload prints a marker instead."""
    r = session.call("aggregate", {"file": file, "op": "count", "where_column": column, "where_value": value})
    assert "PWNED" not in json.dumps(r.data.get("result", [])), r.data
    if r.ok:
        assert r.data["result"] == ["0"], r.data


@pytest.mark.parametrize("value", INJECTIONS)
def test_a_hostile_value_is_never_classified_as_denied(value: str) -> None:
    for command in (
        compile_csv(Query("s.csv", "count", "", "region", "eq", value), "region,units"),
        compile_json(Query("s.json", "count", "", "region", "eq", value), ["region"], lines=False),
    ):
        assert classify(command).tier in (Tier.READ_ONLY, Tier.NEEDS_REVIEW), command


@pytest.mark.parametrize("value", ["a\nb", "a\x1b[2Jb", "a\x00b"])
def test_control_characters_in_a_value_are_refused(value: str) -> None:
    with pytest.raises(CompileError, match="control characters"):
        compile_csv(Query("s.csv", "count", "", "region", "eq", value), "region,units")


@pytest.mark.parametrize("column", ["units; system(1)", "units | sh", "$(whoami)"])
def test_hostile_columns_are_refused(column: str) -> None:
    with pytest.raises(CompileError):
        compile_csv(Query("s.csv", "sum", column), "units,price")
