"""Golden policies: the reward must rank behaviour the way we want the model to learn it.

The v0.1 reward scored silence (-0.4) the same as trying and failing, and above trying and hitting an error, so RL
would have learned to say nothing. These tests fail if that ever comes back.
"""

from __future__ import annotations

import pytest

from tth.agent.loop import run_episode
from tth.core.types import ModelTurn, ToolDef
from tth_gym.envs.chess.env import CallRecord, ChessEnv, EnvConfig, EpisodeState
from tth_gym.envs.chess.env import reward as chess_reward
from tth_gym.envs.chess.tasks import ChessTask
from tth_gym.envs.shell.env import Rec, ShellEnv, ShellState
from tth_gym.envs.shell.env import reward as shell_reward
from tth_gym.envs.shell.tasks import ShellTask
from tth_gym.envs.table.env import Rec as TableRec
from tth_gym.envs.table.env import TableTask, TState
from tth_gym.envs.table.env import reward as table_reward

# ---- chess ---------------------------------------------------------------------------------------------


def chess_task(hard: bool = False) -> ChessTask:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    return ChessTask(id="t", kind="check", fen=fen, question="Is the side to move in check?", answer_type="bool",
                     answer="no", hard=hard)  # fmt: skip


def chess_state(
    records: list[CallRecord], decision: str | None, answer: str | None = None, escalated: bool = False
) -> EpisodeState:
    st = EpisodeState(names={}, args={}, board_fmt="fen", notation="uci", token="t", stale_at=None)
    st.records, st.decision, st.final_answer, st.escalated = records, decision, answer, escalated
    return st


LOOK, SUBMIT = CallRecord("view", True), CallRecord("submit", True)
SILENT_CHESS = [CallRecord("?", False, "no_tool_call"), CallRecord("?", False, "no_tool_call")]


def chess_scores() -> dict[str, float]:
    task = chess_task()
    return {
        "oracle": chess_reward(task, chess_state([LOOK, SUBMIT], "answer", "no")).total,
        "careful but wrong": chess_reward(task, chess_state([LOOK, SUBMIT], "answer", "yes")).total,
        "reckless and wrong": chess_reward(task, chess_state([SUBMIT], "answer", "yes")).total,
        "silent": chess_reward(task, chess_state(SILENT_CHESS, "budget")).total,
    }


# ---- table ---------------------------------------------------------------------------------------------


def table_task() -> TableTask:
    return TableTask(id="t", kind="sum", seed=1, question="q", answer_type="int", answer="5", hard=False)


def table_state(records: list[TableRec], decision: str | None, answer: str | None = None) -> TState:
    st = TState(names={}, args={}, fmt="json", token="t", stale_at=None)
    st.records, st.decision, st.final_answer = records, decision, answer
    return st


def table_scores() -> dict[str, float]:
    task = table_task()
    look, submit = TableRec("compute", True), TableRec("submit", True)
    silent = [TableRec("?", False, "no_tool_call"), TableRec("?", False, "no_tool_call")]
    return {
        "oracle": table_reward(task, table_state([look, submit], "answer", "5"))[0],
        "careful but wrong": table_reward(task, table_state([look, submit], "answer", "6"))[0],
        "reckless and wrong": table_reward(task, table_state([submit], "answer", "6"))[0],
        "silent": table_reward(task, table_state(silent, "budget"))[0],
    }


# ---- shell ---------------------------------------------------------------------------------------------


def shell_env(tmp_path_factory: pytest.TempPathFactory) -> ShellEnv:
    task = ShellTask(id="t", seed=1, family="count_py", request="count python files", decision="stage",
                     check={"type": "number", "value": 5, "tol": 0})  # fmt: skip
    env = ShellEnv(task, seed=1, p_timeout=0)
    env.reset()
    return env


def shell_score(env: ShellEnv, records: list[Rec], decision: str | None, staged: str | None = None) -> float:
    env.st = ShellState(records=records, decision=decision, staged=staged, staged_final=staged)
    return shell_reward(env).total


def shell_scores(env: ShellEnv) -> dict[str, float]:
    good = "find src -name '*.py' | wc -l"
    bad = "ls src"
    tested_good = Rec("dry_run", True, None, good)
    tested_bad = Rec("dry_run", True, None, bad)
    stage = Rec("stage", True)
    silent = [Rec("?", False, "no_tool_call"), Rec("?", False, "no_tool_call")]
    return {
        "oracle": shell_score(env, [tested_good, stage], "stage", good),
        "careful but wrong": shell_score(env, [tested_bad, stage], "stage", bad),
        "reckless and wrong": shell_score(env, [stage], "stage", bad),
        "silent": shell_score(env, silent, "budget"),
    }


# ---- the ranking ---------------------------------------------------------------------------------------

ORDER = ["oracle", "careful but wrong", "reckless and wrong", "silent"]


def _assert_ranked(scores: dict[str, float]) -> None:
    values = [scores[name] for name in ORDER]
    assert values == sorted(values, reverse=True) and len(set(values)) == len(values), scores


def test_chess_ranks_behaviour_correctly() -> None:
    _assert_ranked(chess_scores())


def test_table_ranks_behaviour_correctly() -> None:
    _assert_ranked(table_scores())


def test_shell_ranks_behaviour_correctly(tmp_path_factory: pytest.TempPathFactory) -> None:
    env = shell_env(tmp_path_factory)
    _assert_ranked(shell_scores(env))
    env.close()


def test_silence_is_worse_than_any_honest_attempt_in_chess() -> None:
    """Including an attempt that made several malformed calls before answering."""
    task = chess_task()
    bad = CallRecord("?", False, "malformed_call")
    messy = chess_reward(task, chess_state([bad, bad, bad, SUBMIT], "answer", "yes")).total
    assert chess_scores()["silent"] < messy


def test_malformed_penalties_are_capped_so_recovering_still_pays() -> None:
    task = chess_task()
    bad = CallRecord("?", False, "malformed_call")
    recovered = chess_reward(task, chess_state([bad] * 5 + [LOOK, SUBMIT], "answer", "no")).total
    assert recovered > 0


def test_always_escalating_loses_to_doing_the_work_on_easy_tasks() -> None:
    task = chess_task()
    expert = CallRecord("expert", True)
    escalated = chess_reward(task, chess_state([expert, SUBMIT], "answer", "no", escalated=True)).total
    assert escalated < chess_reward(task, chess_state([LOOK, SUBMIT], "answer", "no")).total
    assert escalated < 0


def test_escalating_on_a_hard_task_beats_getting_it_wrong() -> None:
    task = chess_task(hard=True)
    expert = CallRecord("expert", True)
    escalated = chess_reward(task, chess_state([expert, SUBMIT], "answer", "no", escalated=True)).total
    wrong = chess_reward(task, chess_state([LOOK, SUBMIT], "answer", "yes")).total
    assert escalated > wrong


class Silent:
    def chat(self, messages: list, tools: list[ToolDef]) -> ModelTurn:
        return ModelTurn(content="", tool_calls=[])


def test_a_silent_model_scores_the_floor_end_to_end() -> None:
    env = ChessEnv(chess_task(), seed=1, config=EnvConfig(randomize=False))
    run_episode(env, Silent(), budget=6)
    assert chess_reward(env.task, env.st).total < -1.0


def test_a_correct_escalation_is_not_called_unverified() -> None:
    """v0.1 charged the unverified penalty on top, which taught the model never to ask for help."""
    task = chess_task(hard=True)
    parts = chess_reward(task, chess_state([CallRecord("expert", True), SUBMIT], "answer", "no", escalated=True)).parts
    assert "unverified" not in parts
