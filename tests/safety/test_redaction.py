"""Secrets must not reach an expert payload. The November 2026 audit reproduced each of these leaks."""

from __future__ import annotations

import json

import pytest
from tth.a2a.payload import MAX_RESULT_CHARS, build_payload
from tth.a2a.redact import redact

SECRET = "hunter2"

LEAKS = [
    '{"password": "hunter2"}',
    "{'password': 'hunter2'}",
    '{"api_key":"hunter2"}',
    '{"apiKey": "hunter2"}',
    '{"client_secret": "hunter2"}',
    '{"auth_token": "hunter2"}',
    '{\\"password\\": \\"hunter2\\"}',
    "DB_PASS=hunter2",
    "MYSQL_PWD=hunter2",
    "PGPASSWORD=hunter2",
    "PASS=hunter2",
    "export API_KEY=hunter2",
    "password: hunter2",
    "Authorization: Bearer hunter2",
    "authorization: basic hunter2",
    "curl -H 'Authorization: Bearer hunter2' x",
    "Cookie: session=hunter2",
    "postgres://admin:hunter2@db/x",
]

# Assembled at runtime, so that no token-shaped literal is ever committed. Secret scanners rightly block those.
_BODY = "abcdefghijklmnopqrstuvwxyz0123456789"
TOKENS = [
    "ghp" + "_" + _BODY,
    "glpat" + "-" + _BODY[:20],
    "sk" + "-ant-api03-" + _BODY[:26],
    "sk" + "_live_" + _BODY[:24],
    "hf" + "_" + _BODY[:32],
    "npm" + "_" + _BODY,
    "pypi" + "-" + _BODY[:32],
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "0123456789abcdef" * 2,
    "0123456789abcdef" * 4,
]

NOT_SECRETS = [
    "Author: Jane Doe",
    "authors = 3",
    "BYPASS=1",
    "compass: north",
    "tokens: 1500",
    '{"tokens": 17781, "status": "failed"}',
    "commit 3af7f93726f83d873bc88ccff5c18e965e132f11",
    "session_count=4",
]


@pytest.mark.parametrize("text", LEAKS)
def test_secret_values_are_removed(text: str) -> None:
    out, hits = redact(text)
    assert SECRET not in out, out
    assert hits


@pytest.mark.parametrize("token", TOKENS)
def test_tokens_are_removed(token: str) -> None:
    out, hits = redact(f"value {token} end")
    assert token not in out, out
    assert hits


@pytest.mark.parametrize("text", NOT_SECRETS)
def test_ordinary_text_survives(text: str) -> None:
    assert redact(text)[0] == text


def test_private_key_is_removed_even_without_its_end_marker() -> None:
    body = "A" * 40
    out, hits = redact(f"x\n-----BEGIN OPENSSH PRIVATE KEY-----\n{body}\n")
    assert body not in out and hits == ["private-key"]


def test_secret_env_values_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_TOKEN", "plainvalue123")
    out, hits = redact("the value is plainvalue123 here")
    assert "plainvalue123" not in out
    assert "env-value" in hits


def test_short_env_values_do_not_mangle_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "abc")
    assert "abc" in redact("abc def")[0]


def _messages(content: str) -> list[dict]:
    call = {"id": "1", "function": {"name": "probe", "arguments": {"path": "f"}}}
    return [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "1", "name": "probe", "content": content},
    ]


def test_payload_redacts_a_long_key_before_truncating() -> None:
    key = "-----BEGIN RSA PRIVATE KEY-----\n" + "B" * (MAX_RESULT_CHARS * 2) + "\n-----END RSA PRIVATE KEY-----"
    p = build_payload("req", _messages("config\n" + key), "reason", "macOS", "zsh")
    assert "BBBB" not in p.text
    assert "private-key" in p.redactions


def test_payload_redacts_json_tool_results() -> None:
    content = json.dumps({"sample": ['{"password": "hunter2"}', "DB_PASS=hunter2"]})
    p = build_payload("req", _messages(content), "reason", "macOS", "zsh")
    assert SECRET not in p.text


def test_request_text_is_redacted() -> None:
    p = build_payload("log in with password=hunter2", [], "reason", "macOS", "zsh")
    assert SECRET not in p.text
