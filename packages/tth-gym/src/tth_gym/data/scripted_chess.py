"""Scripted chess teacher: demonstrations generated from python-chess / Stockfish ground truth.

The policy acts only through the episode's (randomized) tools, so every transcript shows the evidence
for its answer: it views the board, queries or simulates, recovers from injected errors (retry after a
timeout or malformed response, re-view after stale state), escalates to the expert on hard tasks and
verifies the expert's move before committing. Episodes are scored by the same verifier as everything
else and only correct, zero-penalty ones are kept (train/data/filter.py).

  uv run python -m tth_gym.data.scripted_chess 600 data/chess/scripted_raw.jsonl
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import chess
from tth.agent.loop import run_episode
from tth.core.types import ModelTurn, ToolCall, ToolDef

from tth_gym.envs.chess.env import ChessEnv, StockfishExpert, episode_summary
from tth_gym.envs.chess.tasks import ChessTask, generate

TEACHER = "scripted-oracle (python-chess + Stockfish)"


class ScriptedChessPolicy:
    def __init__(self, env: ChessEnv, rng: random.Random) -> None:
        self.env = env
        self.rng = rng
        self.plan: list[tuple[str, dict[str, Any]]] = []
        self.last: tuple[str, dict[str, Any]] | None = None
        self.expert_text: str | None = None
        self.started = False

    # ---- helpers ------------------------------------------------------------------
    def _mv(self, uci: str) -> str:
        b = self.env.board
        return b.san(chess.Move.from_uci(uci)) if self.env.st.notation == "san" else uci

    def _call(self, canon: str, args: dict[str, Any]) -> ModelTurn:
        st = self.env.st
        named = {st.args[canon][k]: v for k, v in args.items()}
        if canon in ("simulate", "submit"):
            named[st.args[canon]["state"]] = st.token
        self.last = (canon, args)
        return ModelTurn("", [ToolCall(st.names[canon], named)])

    def _initial_plan(self) -> list[tuple[str, dict[str, Any]]]:
        t: ChessTask = self.env.task
        b = self.env.board
        view = ("view", {"format": self.env.st.board_fmt})
        if t.kind == "legal":
            m = re.search(r"move (\w+)", t.question)
            assert m
            return [view, ("simulate", {"move": m.group(1)}), ("submit", {"answer": t.answer})]
        if t.kind == "check":
            king = b.king(b.turn)
            assert king is not None
            return [view, ("piece", {"square": chess.square_name(king)}), ("submit", {"answer": t.answer})]
        if t.kind == "captures":
            return [view, ("moves", {"captures_only": True}), ("submit", {"answer": t.answer})]
        if t.kind == "pinned":
            sq = re.search(r"on ([a-h][1-8])", t.question)
            assert sq
            return [view, ("piece", {"square": sq.group(1)}), ("submit", {"answer": t.answer})]
        if t.kind == "missing_piece":
            sq = re.search(r"on ([a-h][1-8])", t.question)
            assert sq
            return [view, ("piece", {"square": sq.group(1)}),
                    ("impossible", {"reason": f"There is no piece on {sq.group(1)}, so no move can be made with it."})]  # fmt: skip
        if t.kind == "mate1":
            mate = t.accept_moves[0]
            plan = [view, ("moves", {})]
            checks = [m.uci() for m in b.legal_moves if b.gives_check(m) and m.uci() not in t.accept_moves]
            if checks and self.rng.random() < 0.5:
                plan.append(("simulate", {"move": self._mv(self.rng.choice(checks))}))
            return plan + [("simulate", {"move": self._mv(mate)}), ("submit", {"answer": self._mv(mate)})]
        if t.kind in ("mate1_none", "best_move", "engine_eval"):
            q = {"mate1_none": "Is there a move that checkmates immediately in this position? If so, which?",
                 "best_move": "What is the best move for the side to move?",
                 "engine_eval": "What is the engine evaluation of this position?"}[t.kind]  # fmt: skip
            return [view, ("expert", {"question": q})]  # the rest depends on the expert's answer
        raise ValueError(t.kind)

    def _after_expert(self) -> list[tuple[str, dict[str, Any]]]:
        t, text = self.env.task, self.expert_text or ""
        if t.kind == "mate1_none":
            return [("impossible", {"reason": "The expert confirmed there is no mate in one."})]
        if t.kind == "engine_eval":
            m = re.search(r"([+-]?\d+\.\d+)", text)
            ans = "yes" if m and float(m.group(1)) > 2.0 else "no"
            return [("submit", {"answer": ans})]
        m = re.search(r"Best move: (\S+)", text)
        move = m.group(1) if m else t.answer
        return [("simulate", {"move": move}), ("submit", {"answer": move})]

    # ---- policy -------------------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]], tools: list[ToolDef]) -> ModelTurn:
        if not self.started:
            self.started = True
            self.plan = self._initial_plan()
        else:
            last = messages[-1]
            result = last.get("content", "")
            try:
                data = json.loads(result) if isinstance(result, str) else result
            except json.JSONDecodeError:
                # Plain-text results (ASCII boards) are fine; broken JSON is a malformed response: retry.
                data = None if str(result).lstrip().startswith("{") else {"text": result}
            err = data.get("error", "") if isinstance(data, dict) else ("" if data is not None else "malformed")
            if err and self.last:
                if "STALE" in err or "state token" in err:
                    self.plan.insert(0, self.last)
                    self.plan.insert(0, ("view", {"format": self.env.st.board_fmt}))
                else:  # timeout / malformed: retry
                    self.plan.insert(0, self.last)
            elif self.last and self.last[0] == "expert" and isinstance(data, dict):
                self.expert_text = str(data.get("expert", ""))
                self.plan = self._after_expert()
        canon, args = self.plan.pop(0)
        return self._call(canon, args)


def main(n: int, out: str, seed: int = 3_000) -> None:
    tasks = list(generate(n, seed=seed, prefix="s"))
    tasks += list(generate(max(1, n // 8), seed=seed + 1, mix={"engine_eval": 1}, prefix="sh"))
    expert = StockfishExpert()
    rng = random.Random(seed)
    rows = []
    try:
        for i, t in enumerate(tasks):
            env = ChessEnv(t, seed=seed * 7 + i, expert=expert)
            pol = ScriptedChessPolicy(env, rng)
            ep = run_episode(env, pol)
            rows.append({"id": ep.id, "messages": ep.messages, "tools": ep.tools, "summary": episode_summary(env),
                         "meta": {**ep.meta, "teacher": TEACHER}})  # fmt: skip
    finally:
        expert.close()
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = sum(r["summary"]["correct"] and r["summary"]["reward"] > 0 for r in rows)
    print(json.dumps({"episodes": len(rows), "correct_positive": ok}))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
