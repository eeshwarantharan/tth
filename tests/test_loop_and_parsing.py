from __future__ import annotations

from tth.runtime.toolcalls import parse_qwen, to_openai_messages


def test_parse_qwen_tool_call() -> None:
    t = parse_qwen('<tool_call>\n{"name": "probe", "arguments": {"path": "a", "kind": "head"}}\n</tool_call>')
    assert t.tool_calls[0].name == "probe" and t.tool_calls[0].args == {"path": "a", "kind": "head"}


def test_parse_malformed() -> None:
    t = parse_qwen('<tool_call>{"name": "probe", "arguments": {"path": </tool_call>')
    assert t.tool_calls[0].malformed


def test_plain_text_has_no_calls() -> None:
    assert parse_qwen("<tools><function name='stage'/></tools>").tool_calls == []


def test_openai_args_are_strings() -> None:
    m = to_openai_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "x", "arguments": {"a": 1}},
                    }
                ],
            }
        ]
    )
    assert m[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_bare_json_call_with_wrong_wrapper() -> None:
    # Seen on Kaggle: correct body, <|im_start|> instead of <tool_call>.
    t = parse_qwen('<|im_start|>\n{"name": "inspect_board", "arguments": {"style": "json"}}\n<|im_end|>')
    assert t.tool_calls[0].name == "inspect_board" and t.tool_calls[0].args == {"style": "json"}


def test_prose_is_not_a_call() -> None:
    assert parse_qwen('I would run {"cmd": "ls"} here').tool_calls == []


def test_xml_function_call() -> None:
    text = (
        "<tools>\n"
        '  <function name="stage">\n'
        "    <arguments>\n"
        '      <command>find . -name "*.py" -type f | wc -l</command>\n'
        "      <explanation>Count Python files</explanation>\n"
        "    </arguments>\n"
        "  </function>\n"
        "</tools>"
    )
    t = parse_qwen(text)
    assert len(t.tool_calls) == 1
    assert t.tool_calls[0].name == "stage"
    assert t.tool_calls[0].args == {
        "command": 'find . -name "*.py" -type f | wc -l',
        "explanation": "Count Python files",
    }
