from __future__ import annotations

import chess

from train.envs.chess.env import ChessEnv, EnvConfig, reward
from train.envs.chess.tasks import ChessTask
from tth.agent.loop import run_episode
from tth.agent.protocol import ModelTurn, ToolCall

MATE_FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"  # Ra8#
TASK = ChessTask("x", "mate1", MATE_FEN, "Find mate", "move", "a1a8", accept_moves=["a1a8"])
CLEAN = EnvConfig(p_timeout=0, p_malformed=0, p_stale_episode=0, p_distractor=0, randomize=False)


class Script:
    def __init__(self, env: ChessEnv, plan: list[tuple[str, dict]]) -> None:
        self.env, self.plan, self.i = env, plan, 0

    def chat(self, messages, tools):
        canon, args = self.plan[self.i]
        self.i += 1
        if canon == "TEXT":
            return ModelTurn("hello", [])
        st = self.env.st
        name = st.names[canon]
        a = {st.args[canon][k]: (st.token if v == "$TOKEN" else v) for k, v in args.items()}
        return ModelTurn("", [ToolCall(name, a)])


def _run(plan, task=TASK, cfg=CLEAN):
    env = ChessEnv(task, seed=1, config=cfg)
    run_episode(env, Script(env, plan))
    return env, reward(env.task, env.st)


def test_verified_commit_gets_full_reward():
    env, r = _run(
        [
            ("view", {}),
            ("simulate", {"move": "a1a8", "state": "$TOKEN"}),
            ("submit", {"answer": "a1a8", "state": "$TOKEN"}),
        ]
    )
    assert r.correct and r.total == 1.0


def test_commit_without_verifying_is_penalised():
    env, r = _run([("view", {}), ("submit", {"answer": "a1a8", "state": "$TOKEN"})])
    assert r.correct and r.parts["unverified"] == -0.4


def test_wrong_token_rejected():
    env, r = _run(
        [
            ("view", {}),
            ("simulate", {"move": "a1a8", "state": "nope"}),
            ("simulate", {"move": "a1a8", "state": "$TOKEN"}),
            ("submit", {"answer": "a1a8", "state": "$TOKEN"}),
        ]
    )
    assert r.correct


def test_text_turns_are_malformed_and_end_episode():
    env, r = _run([("TEXT", {}), ("TEXT", {})])
    assert not r.correct and r.parts["malformed"] == -0.4 and env.st.decision == "budget"


def test_impossible_task():
    t = ChessTask(
        "i",
        "missing_piece",
        chess.STARTING_FEN,
        "Move the knight on e4",
        "impossible",
        "impossible",
    )
    env, r = _run([("view", {}), ("impossible", {"reason": "no knight on e4"})], task=t)
    assert r.correct and r.total == 1.0


def test_unnecessary_escalation_costs():
    env, r = _run(
        [
            ("view", {}),
            ("expert", {"question": "?"}),
            ("simulate", {"move": "a1a8", "state": "$TOKEN"}),
            ("submit", {"answer": "a1a8", "state": "$TOKEN"}),
        ]
    )
    assert r.parts["escalation"] == -0.3


def test_schema_randomization_varies():
    names = {tuple(sorted(ChessEnv(TASK, seed=s).reset().tools[0].parameters["properties"])) for s in range(20)}
    tool_sets = {tuple(sorted(t.name for t in ChessEnv(TASK, seed=s).reset().tools)) for s in range(20)}
    assert len(tool_sets) > 10 and len(names) > 3


def test_stale_state_then_acting_is_penalised():
    cfg = EnvConfig(p_timeout=0, p_malformed=0, p_stale_episode=1.0, p_distractor=0, randomize=True)
    seed = next(
        s for s in range(200) if (lambda e: (e.reset(), e.st.stale_at)[1])(ChessEnv(TASK, seed=s, config=cfg)) == 1
    )
    env = ChessEnv(TASK, seed=seed, config=cfg)
    plan = [
        ("view", {}),
        ("moves", {}),
        ("simulate", {"move": "a1a8", "state": "$TOKEN"}),
        ("view", {}),
        ("simulate", {"move": "a1a8", "state": "$TOKEN"}),
        ("submit", {"answer": "a1a8", "state": "$TOKEN"}),
    ]
    run_episode(env, Script(env, plan))
    rb = reward(env.task, env.st)
    assert rb.parts.get("stale") == -0.5 and rb.correct
