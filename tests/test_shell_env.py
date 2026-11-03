"""The shell gym: it runs the product's own session, and its verifier is strict."""

from __future__ import annotations

import pytest

from tth.core.types import ToolCall
from tth_gym.envs.shell.env import ShellEnv, ShellState, _number_answer, reward
from tth_gym.envs.shell.tasks import ShellTask

COUNT_PY = ShellTask(id="t", seed=1, family="count_py", request="count .py files in src", decision="stage",
                     check={"type": "number", "value": 5, "tol": 0})  # fmt: skip


@pytest.fixture
def env():
    e = ShellEnv(COUNT_PY, seed=1, p_timeout=0)
    e.reset()
    yield e
    e.close()


def test_the_gym_uses_the_products_gate(env: ShellEnv) -> None:
    """Staging without testing gets tested by the gate, exactly as in the product."""
    out = env.step(ToolCall(name="stage", args={"command": "find src -name '*.py' | wc -l"}))
    assert out.done
    assert env.st.decision == "stage"
    assert env.st.verified


def test_the_gym_stops_loops_the_way_the_product_does(env: ShellEnv) -> None:
    command = {"command": "find src -name '*.py' | wc -l"}
    assert not env.step(ToolCall(name="dry_run", args=command)).done
    assert env.step(ToolCall(name="dry_run", args=command)).done
    assert env.st.decision == "stage"


def test_tools_come_over_mcp(env: ShellEnv) -> None:
    assert env.session is not None
    assert {t.name for t in env.session.tools.tools()} >= {"inspect_context", "probe", "dry_run"}


def test_the_right_answer_scores_correct(env: ShellEnv) -> None:
    env.step(ToolCall(name="stage", args={"command": "find src -name '*.py' | wc -l"}))
    assert reward(env).correct


@pytest.mark.parametrize(
    ("lines", "want", "correct"),
    [
        (["5"], 5, True),
        (["   5"], 5, True),
        (["5 src/a.py"], 5, True),
        (["       5 app.log"], 5, True),
        (["total: 5"], 5, True),
        (["5.0"], 5, True),
        (["4"], 5, False),
        (["5", "6", "7"], 5, False),  # v0.1 passed this, because any number in up to 3 lines counted
        (["34", "5"], 5, False),
        (["3 5 7"], 5, False),
        ([], 5, False),
    ],
)
def test_a_number_answer_is_strict(lines: list[str], want: int, correct: bool) -> None:
    assert _number_answer(lines, {"value": want, "tol": 0}, ShellState()) is correct
