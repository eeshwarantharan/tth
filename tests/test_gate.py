"""The verification gate between the model and the prompt. Each rule here comes from a measured failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tth.agent.loop import run_episode
from tth.agent.product import ProductEnv
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth.inference.toolcalls import parse_qwen


def call(name: str, **args: object) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


class Scripted:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.turns = 0

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        self.turns += 1
        return parse_qwen(self.texts.pop(0))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    return tmp_path


def run(project: Path, *texts: str, budget: int = 6) -> tuple[ProductEnv, Scripted, list[dict]]:
    env = ProductEnv("do it", project, "zsh")
    policy = Scripted(*texts)
    ep = run_episode(env, policy, budget=budget)
    env.close()
    results = [m for m in ep.messages if m["role"] == "tool"]
    return env, policy, results


def staged(env: ProductEnv) -> str | None:
    return env.staged["command"] if env.staged else None


# ---- Staging a read-only command tests it first ----------------------------------------------------


def test_staging_an_untested_read_only_command_tests_it_first(project: Path) -> None:
    env, policy, _ = run(project, call("stage", command="wc -l a.txt"))
    assert staged(env) == "wc -l a.txt"
    assert env.verified
    assert policy.turns == 1  # no extra model turn was spent


def test_a_failing_command_is_not_staged_and_the_model_hears_why(project: Path) -> None:
    env, _, results = run(project, call("stage", command="cat missing.txt"), call("ask_user", question="which file?"))
    assert env.decision == "ask"
    assert "missing.txt" in results[0]["content"]


@pytest.mark.parametrize("command", ["find . -name zzz", "grep zzz a.txt"])
def test_a_command_with_no_output_is_not_called_verified(project: Path, command: str) -> None:
    """v0.1 labelled a find that matched nothing as verified. grep says "no match" by exiting 1."""
    env, _, results = run(project, call("stage", command=command), call("stage", command=command))
    assert staged(env) == command
    assert not env.verified
    assert "no output" in results[0]["content"]


def test_a_blank_line_is_not_an_answer(project: Path) -> None:
    """Found in a real run: an awk that matched nothing printed an empty sum, and was labelled verified."""
    command = "awk '$1 == \"zzz\" {s += 1} END {print s}' a.txt"
    env, _, results = run(project, call("stage", command=command), call("stage", command=command))
    assert staged(env) == command
    assert not env.verified
    assert "no output" in results[0]["content"]


def test_grep_count_of_zero_is_a_real_answer(project: Path) -> None:
    env, _, _ = run(project, call("stage", command="grep -c zzz a.txt"))
    assert staged(env) == "grep -c zzz a.txt"
    assert env.verified


def test_a_placeholder_is_never_swapped_for_another_command(project: Path) -> None:
    """v0.1 replaced `rm -rf $count` with whatever had last passed, so `ls` could appear unasked."""
    env, _, _ = run(
        project,
        call("dry_run", command="ls"),
        call("stage", command="rm -rf $count"),
        call("ask_user", question="what should I remove?"),
    )
    assert env.decision == "ask"
    assert staged(env) is None


# ---- Changing files needs a look first --------------------------------------------------------------


def test_changing_files_without_looking_is_refused_once(project: Path) -> None:
    env, _, results = run(
        project, call("stage", command="mkdir out"), call("inspect_context"), call("stage", command="mkdir out")
    )
    assert "look" in results[0]["content"].lower()
    assert staged(env) == "mkdir out"
    assert not env.verified


def test_changing_files_after_looking_goes_straight_through(project: Path) -> None:
    env, policy, _ = run(project, call("inspect_context"), call("stage", command="mkdir out"))
    assert staged(env) == "mkdir out"
    assert policy.turns == 2


# ---- Loops end ---------------------------------------------------------------------------------------


def test_repeating_a_passing_dry_run_commits_it(project: Path) -> None:
    """The top measured failure: trained models dry-run forever and never commit."""
    env, policy, _ = run(project, call("dry_run", command="wc -l a.txt"), call("dry_run", command="wc -l a.txt"))
    assert staged(env) == "wc -l a.txt"
    assert env.verified
    assert policy.turns == 2


def test_repeating_a_failing_command_stops_the_loop(project: Path) -> None:
    env, policy, _ = run(project, call("dry_run", command="cat missing"), call("dry_run", command="cat missing"))
    assert env.decision == "escalate"
    assert staged(env) is None
    assert policy.turns == 2


def test_repeating_a_changes_files_command_stages_that_command(project: Path) -> None:
    """v0.1 staged an earlier, unrelated command here."""
    env, _, _ = run(
        project,
        call("dry_run", command="ls"),
        call("dry_run", command="mkdir out"),
        call("dry_run", command="mkdir out"),
    )
    assert staged(env) == "mkdir out"
    assert not env.verified


# ---- The model's words count ------------------------------------------------------------------------


def test_saying_it_cannot_do_this_stages_nothing(project: Path) -> None:
    """v0.1 staged the last dry-run when the model said it couldn't do the task."""
    env, _, _ = run(project, call("dry_run", command="ls"), "I cannot do this.", "I really can't.")
    assert staged(env) is None


def test_going_silent_after_a_passing_test_commits_that_command(project: Path) -> None:
    env, _, _ = run(project, call("dry_run", command="wc -l a.txt"), "", "")
    assert staged(env) == "wc -l a.txt"
    assert env.verified


def test_running_out_of_attempts_with_a_tested_command_offers_it(project: Path) -> None:
    calls = [call("dry_run", command=c) for c in ("ls", "wc -l a.txt", "cat a.txt")]
    env, _, _ = run(project, *calls, budget=3)
    assert staged(env) == "cat a.txt"
    assert env.verified


def test_running_out_of_attempts_with_nothing_tested_stages_nothing(project: Path) -> None:
    env, _, _ = run(project, call("inspect_context"), call("inspect_context"), budget=2)
    assert staged(env) is None


# ---- Refusals still hold -----------------------------------------------------------------------------


def test_the_gate_never_stages_what_the_safety_layer_refuses(project: Path) -> None:
    env, _, results = run(
        project, call("inspect_context"), call("stage", command="rm -rf ~"), call("ask_user", question="?")
    )
    assert staged(env) is None
    assert "refused" in results[1]["content"].lower() or "notes" in results[1]["content"]


def test_gate_commits_are_labelled_in_the_trace(project: Path, tmp_path: Path) -> None:
    from tth.trace.schema import TraceLogger, read_trace

    path = tmp_path / "trace.jsonl"
    env = ProductEnv("do it", project, "zsh")
    run_episode(
        env, Scripted(call("dry_run", command="ls"), call("dry_run", command="ls")), budget=6, logger=TraceLogger(path)
    )
    env.close()
    assert any(e.meta.get("source") == "gate" for e in read_trace(path))


# ---- Things a small model really did, found in the bake-off red team ----------------------------------


@pytest.mark.parametrize(
    ("request_text", "command"),
    [
        ("print the contents of the notes", "print the contents of the notes"),
        ("format the usb drive", "format the usb drive"),
        ("chmod 777 everything recursively from root", "chmod 777 everything recursively from root"),
        ("force-move the notes", "force-move the notes"),
    ],
)
def test_echoing_the_request_is_not_a_command(project: Path, request_text: str, command: str) -> None:
    env = ProductEnv(request_text, project, "zsh")
    env.reset()
    env.step(ToolCall(name="inspect_context", args={}))
    out = env.step(ToolCall(name="stage", args={"command": command}))
    env.close()
    assert env.staged is None
    assert not out.done


@pytest.mark.parametrize("command", ["show me my files", "format the usb drive", "force-move a b", "kubeectl get pods"])
def test_a_program_that_does_not_exist_here_is_refused(project: Path, command: str) -> None:
    env = ProductEnv("do it", project, "zsh")
    env.reset()
    env.step(ToolCall(name="inspect_context", args={}))
    out = env.step(ToolCall(name="stage", args={"command": command}))
    env.close()
    assert env.staged is None
    assert "isn't on this machine" in json.dumps(out.result)


@pytest.mark.parametrize("command", ["cd src", "echo hi", "type ls", "ls | wc -l", "mkdir out"])
def test_builtins_and_real_programs_are_fine(project: Path, command: str) -> None:
    env = ProductEnv("do it", project, "zsh")
    env.reset()
    env.step(ToolCall(name="inspect_context", args={}))
    out = env.step(ToolCall(name="stage", args={"command": command}))
    env.close()
    assert "isn't on this machine" not in json.dumps(out.result)


# ---- Good answers written as prose, as Granite does --------------------------------------------------


def test_a_fenced_command_in_prose_is_tested_and_staged(project: Path) -> None:
    text = "**Command to run**\n\n```bash\nwc -l a.txt\n```\n\n**Explanation:** counts lines."
    env, policy, _ = run(project, text)
    assert staged(env) == "wc -l a.txt"
    assert env.verified
    assert policy.turns == 1


def test_an_inline_command_in_prose_is_tested_and_staged(project: Path) -> None:
    env, _, _ = run(project, "It has 2 lines. (You can verify with `wc -l a.txt`.)")
    assert staged(env) == "wc -l a.txt"
    assert env.verified


def test_a_dangerous_command_in_prose_is_still_refused(project: Path) -> None:
    env, _, _ = run(project, "Run this:\n```bash\nrm -rf ~\n```", "Really:\n```bash\nrm -rf ~\n```")
    assert staged(env) is None


def test_a_broken_command_in_prose_is_not_staged(project: Path) -> None:
    env, _, _ = run(project, "Try `cat missing.txt`", "Try `cat missing.txt`")
    assert staged(env) is None


def test_prose_with_several_candidates_is_not_guessed_at(project: Path) -> None:
    env, _, _ = run(project, "Either `ls` or `wc -l a.txt` works.", "Either `ls` or `wc -l a.txt` works.")
    assert staged(env) is None


def test_an_inline_word_is_not_a_command(project: Path) -> None:
    env, _, _ = run(project, "The file `a.txt` has 2 lines.", "The file `a.txt` has 2 lines.")
    assert staged(env) is None


def test_saying_it_cannot_do_this_still_stages_nothing(project: Path) -> None:
    env, _, _ = run(project, call("dry_run", command="ls"), "I can't do that.", "No.")
    assert staged(env) is None


def test_a_prose_answer_gets_a_useful_nudge(project: Path) -> None:
    _, _, _ = run(project, "The answer is 7.", call("ask_user", question="?"))
    env = ProductEnv("x", project, "zsh")
    env.reset()
    out = env.on_text("The answer is 7.")
    env.close()
    assert "stage" in str(out.result)
    assert "command" in str(out.result)
