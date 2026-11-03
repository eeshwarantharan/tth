"""Gym environments behind MCP must behave exactly as they do without it."""

from __future__ import annotations

import json

import pytest

from tth.agent.loop import run_episode
from tth.core.types import ModelTurn, ToolCall, ToolDef
from tth_gym.core.mcp_env import McpEnvironment
from tth_gym.envs.chess.env import ChessEnv, EnvConfig
from tth_gym.envs.chess.env import reward as chess_reward
from tth_gym.envs.chess.tasks import ChessTask
from tth_gym.envs.table.env import TableEnv, generate

FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHESS = ChessTask(
    id="t", kind="check", fen=FEN, question="Is the side to move in check?", answer_type="bool", answer="no"
)


class Exerciser:
    """Calls every tool it's offered, with the arguments its schema names, then some bad calls. Deterministic."""

    def __init__(self, turns: int = 14) -> None:
        self.turns = turns
        self.i = 0

    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        self.i += 1
        if self.i > self.turns:
            return ModelTurn(content="", tool_calls=[])
        if self.i % 7 == 0:
            return ModelTurn(content="", tool_calls=[ToolCall(name="not_a_tool", args={"x": 1})])
        if self.i % 5 == 0:
            return ModelTurn(content="", tool_calls=[ToolCall(name="", args={}, malformed="unparseable")])
        terminal = ("submit", "answer", "impossible", "commit", "report")
        usable = [t for t in tools if not any(w in t.name for w in terminal)] or tools
        tool = usable[self.i % len(usable)]
        props = tool.parameters.get("properties", {})
        return ModelTurn(content="", tool_calls=[ToolCall(name=tool.name, args={k: "e2e4" for k in props})])


def _chess(seed: int) -> ChessEnv:
    return ChessEnv(CHESS, seed=seed, config=EnvConfig())


def _table(seed: int) -> TableEnv:
    return TableEnv(generate(1, seed=seed)[0], seed=seed)


def _strip_ids(messages: list[dict]) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False)


@pytest.mark.parametrize("make", [_chess, _table], ids=["chess", "table"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_every_message_is_identical_over_mcp(make, seed: int) -> None:
    """Randomized schemas, injected errors, stale state, malformed replies: all must survive the protocol unchanged."""
    direct = run_episode(make(seed), Exerciser(), budget=12, episode_id="same")
    wrapped = McpEnvironment(make(seed))
    served = run_episode(wrapped, Exerciser(), budget=12, episode_id="same")
    wrapped.close()
    assert _strip_ids(served.messages) == _strip_ids(direct.messages)
    assert served.decision == direct.decision


@pytest.mark.parametrize("make", [_chess, _table], ids=["chess", "table"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_same_seed_gives_the_same_episode(make, seed: int) -> None:
    """Experiments have to be repeatable. Tokens used to come from `secrets`, so episodes differed run to run."""
    first = run_episode(make(seed), Exerciser(), budget=12, episode_id="same")
    second = run_episode(make(seed), Exerciser(), budget=12, episode_id="same")
    assert _strip_ids(first.messages) == _strip_ids(second.messages)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rewards_are_identical_over_mcp(seed: int) -> None:
    plain = _chess(seed)
    run_episode(plain, Exerciser(), budget=12)
    wrapped = McpEnvironment(_chess(seed))
    run_episode(wrapped, Exerciser(), budget=12)
    wrapped.close()
    assert chess_reward(CHESS, wrapped.st).total == chess_reward(CHESS, plain.st).total


def test_randomized_schemas_are_discovered_over_the_protocol() -> None:
    names = set()
    for seed in range(8):
        env = McpEnvironment(_chess(seed))
        start = env.reset()
        names.add(tuple(sorted(t.name for t in start.tools)))
        assert env.session is not None
        assert {t.name for t in env.session.tools()} == {t.name for t in start.tools}
        env.close()
    assert len(names) > 1, "schemas should differ between episodes"


def test_the_session_closes_when_the_episode_ends() -> None:
    env = McpEnvironment(_chess(1))
    run_episode(env, Exerciser(turns=0), budget=2)
    assert env.session is None
