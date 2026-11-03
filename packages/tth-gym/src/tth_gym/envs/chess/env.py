"""Hidden-board chess environment with randomized tool schemas and error injection.

The board is never in the prompt. Tool names, argument names/order, descriptions
and output formats change every episode, so the model must read schemas.
"""

from __future__ import annotations

import json
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.engine
from tth.core.types import EpisodeStart, StepOutcome, ToolCall, ToolDef

from tth_gym.envs.chess.tasks import ChessTask

# Canonical tool -> pool of names. Canonical arg -> pool of names.
NAME_POOL = {
    "view": ["view_board", "get_position", "inspect_board", "show_board", "board_state", "peek",
             "read_position", "observe"],
    "moves": ["legal_moves", "list_moves", "get_legal_moves", "enumerate_moves", "move_options",
              "query_moves"],
    "piece": ["piece_info", "inspect_square", "square_details", "probe_square", "examine",
              "lookup_square"],
    "simulate": ["simulate", "try_move", "dry_run_move", "preview_move", "test_move", "sandbox_move"],
    "submit": ["submit", "commit", "final_answer", "answer", "finalize", "respond"],
    "impossible": ["report_impossible", "cannot_do", "decline_task", "flag_impossible",
                   "task_not_possible"],
    "expert": ["ask_expert", "consult_expert", "escalate", "call_grandmaster", "request_help",
               "expert_agent"],
}  # fmt: skip
ARG_POOL = {
    "format": ["format", "fmt", "encoding", "style", "repr"],
    "square": ["square", "sq", "cell", "at", "location"],
    "captures_only": ["captures_only", "only_captures", "capture_filter", "just_captures"],
    "move": ["move", "mv", "candidate", "action", "play"],
    "state": ["state", "state_token", "version", "token", "etag", "snapshot"],
    "answer": ["answer", "value", "response", "result", "final"],
    "reason": ["reason", "why", "explanation", "details"],
    "question": ["question", "query", "prompt", "ask"],
}  # fmt: skip
DESC_POOL = {
    "view": ["Return the current position and a state token.",
             "Read the hidden board. The response includes a state token required by other tools.",
             "Look at the current game position (also returns the current state token)."],
    "moves": ["List legal moves for the side to move, optionally from one square.",
              "Enumerate the legal moves in the position. Can filter by origin square or captures."],
    "piece": ["Describe the piece on a square: type, color, whether it is pinned, attackers.",
              "Inspect one square of the board."],
    "simulate": ["Try a move on a copy of the board without committing. Needs the current state token.",
                 "Preview the result of a move (legality, check, mate). Requires the state token."],
    "submit": ["Submit your final answer. Ends the task. Needs the current state token.",
               "Commit the final answer (a move, yes/no, or a number). This is irreversible."],
    "impossible": ["Report that the task cannot be done as asked. Ends the task.",
                   "Use when the request is impossible (e.g. it refers to something that does not exist)."],
    "expert": ["Ask a much stronger chess expert agent. Costs budget; use only when you cannot verify it yourself.",
               "Escalate to an expensive expert agent for hard positions."],
}  # fmt: skip
DISTRACTORS = [
    ToolDef("get_clock", "Return the remaining time on both clocks.", {"type": "object", "properties": {}}),
    ToolDef("player_rating", "Look up a player's Elo rating by name.",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
    ToolDef("opening_book", "Return the name of the opening for an ECO code.",
            {"type": "object", "properties": {"eco": {"type": "string"}}, "required": ["eco"]}),
    ToolDef("reset_game", "Reset the game to the starting position. Not allowed during a task.",
            {"type": "object", "properties": {}}),
]  # fmt: skip

SYSTEM = (
    "You are a careful agent working on a chess board you cannot see. Use the tools to look at the "
    "board before answering, verify moves before committing, and re-check after any error. If the "
    "task is beyond what you can verify yourself, consult the expert (it has a cost). If the task is "
    "impossible, report it. Call exactly one tool per turn. Tool names and arguments change between "
    "tasks: read the tool schemas."
)

STATE_TOOLS = {"simulate", "submit"}
OBSERVE_TOOLS = {"view", "moves", "piece", "simulate"}


@dataclass
class EnvConfig:
    p_timeout: float = 0.07
    p_malformed: float = 0.05
    p_stale_episode: float = 0.2
    p_distractor: float = 0.5
    expert_cost: float = 0.3
    randomize: bool = True


@dataclass
class CallRecord:
    canon: str  # canonical tool name, or "?" for unknown/malformed
    ok: bool
    error: str | None = None  # timeout | malformed_response | stale | bad_args | unknown_tool | ...
    move: str | None = None  # UCI move for simulate/submit


@dataclass
class EpisodeState:
    names: dict[str, str]
    args: dict[str, dict[str, str]]
    board_fmt: str
    notation: str
    token: str
    stale_at: int | None
    records: list[CallRecord] = field(default_factory=list)
    stale_pending: bool = False
    escalated: bool = False
    final_answer: str | None = None
    decision: str | None = None
    verified_moves: set[str] = field(default_factory=set)


class ChessEnv:
    domain = "chess"

    def __init__(self, task: ChessTask, seed: int = 0, config: EnvConfig | None = None,
                 expert: Any = None) -> None:  # fmt: skip
        self.task = task
        self.cfg = config or EnvConfig()
        self.rng = random.Random(seed)
        self.board = chess.Board(task.fen)
        self.expert = expert
        self.st: EpisodeState

    # ---- schema randomization -------------------------------------------------
    def _schema(self) -> tuple[list[ToolDef], EpisodeState]:
        r = self.rng
        rand = self.cfg.randomize
        names = {k: (r.choice(v) if rand else v[0]) for k, v in NAME_POOL.items()}
        a = {k: (r.choice(v) if rand else v[0]) for k, v in ARG_POOL.items()}
        board_fmt = r.choice(["fen", "ascii", "json"]) if rand else "fen"
        notation = r.choice(["uci", "san"]) if rand else "uci"
        desc = {k: (r.choice(v) if rand else v[0]) for k, v in DESC_POOL.items()}

        def obj(props: dict[str, dict], required: list[str]) -> dict:
            keys = list(props)
            if rand:
                r.shuffle(keys)
            return {
                "type": "object",
                "properties": {k: props[k] for k in keys},
                "required": required,
            }

        mv = f"Move in {notation.upper()} notation" + (" (e.g. e2e4)" if notation == "uci" else " (e.g. Nf3)")
        args: dict[str, dict[str, str]] = {
            "view": {"format": a["format"]}, "moves": {"square": a["square"], "captures_only": a["captures_only"]},
            "piece": {"square": a["square"]}, "simulate": {"move": a["move"], "state": a["state"]},
            "submit": {"answer": a["answer"], "state": a["state"]}, "impossible": {"reason": a["reason"]},
            "expert": {"question": a["question"]},
        }  # fmt: skip
        tools = [
            ToolDef(names["view"], desc["view"] + f" Board format: {board_fmt}.",
                    obj({a["format"]: {"type": "string", "enum": [board_fmt], "description": "Output format"}}, [])),
            ToolDef(names["moves"], desc["moves"] + f" Moves are in {notation.upper()}.",
                    obj({a["square"]: {"type": "string", "description": "Optional origin square, e.g. e2"},
                         a["captures_only"]: {"type": "boolean", "description": "Only list captures"}}, [])),
            ToolDef(names["piece"], desc["piece"],
                    obj({a["square"]: {"type": "string", "description": "Square name, e.g. f3"}}, [a["square"]])),
            ToolDef(names["simulate"], desc["simulate"],
                    obj({a["move"]: {"type": "string", "description": mv},
                         a["state"]: {"type": "string", "description": "Current state token"}},
                        [a["move"], a["state"]])),
            ToolDef(names["submit"], desc["submit"],
                    obj({a["answer"]: {"type": "string", "description": f"Final answer: {mv.lower()}, yes/no, or a number"},
                         a["state"]: {"type": "string", "description": "Current state token"}},
                        [a["answer"], a["state"]])),
            ToolDef(names["impossible"], desc["impossible"],
                    obj({a["reason"]: {"type": "string"}}, [a["reason"]])),
            ToolDef(names["expert"], desc["expert"],
                    obj({a["question"]: {"type": "string"}}, [a["question"]])),
        ]  # fmt: skip
        if rand and r.random() < self.cfg.p_distractor:
            tools += r.sample(DISTRACTORS, r.randint(1, 2))
        if rand:
            r.shuffle(tools)
        stale_at = r.randint(1, 3) if rand and r.random() < self.cfg.p_stale_episode else None
        st = EpisodeState(names=names, args=args, board_fmt=board_fmt, notation=notation,
                          token=secrets.token_hex(3), stale_at=stale_at)  # fmt: skip
        return tools, st

    # ---- rendering ------------------------------------------------------------
    def _board_repr(self) -> Any:
        b, side = self.board, "white" if self.board.turn else "black"
        if self.st.board_fmt == "fen":
            return {"fen": b.fen(), "state": self.st.token}
        if self.st.board_fmt == "ascii":
            rows = []
            for rank in range(7, -1, -1):
                row = [(b.piece_at(chess.square(f, rank)) or ".").__str__() for f in range(8)]
                rows.append(f"{rank + 1} " + " ".join(row))
            rows.append("  a b c d e f g h")
            return "\n".join(rows) + f"\nside to move: {side}\nstate: {self.st.token}"
        pieces = {chess.square_name(sq): p.symbol() for sq, p in sorted(b.piece_map().items())}
        return {
            "side_to_move": side,
            "pieces": pieces,
            "castling": b.castling_xfen(),
            "state": self.st.token,
        }

    def _fmt_move(self, m: chess.Move, board: chess.Board | None = None) -> str:
        return (board or self.board).san(m) if self.st.notation == "san" else m.uci()

    def _parse_move(self, s: str) -> chess.Move | None:
        s = (s or "").strip()
        for parse in (self.board.parse_uci, self.board.parse_san):
            try:
                return parse(s)
            except ValueError:
                continue
        return None

    # ---- protocol ---------------------------------------------------------------
    def reset(self) -> EpisodeStart:
        tools, self.st = self._schema()
        self.tools = tools
        self.canon_of = {v: k for k, v in self.st.names.items()}
        return EpisodeStart(system=SYSTEM, user=self.task.question, tools=tools,
                            meta={"task_id": self.task.id, "kind": self.task.kind, "hard": self.task.hard,
                                  "names": self.st.names, "board_fmt": self.st.board_fmt,
                                  "notation": self.st.notation})  # fmt: skip

    def _err(self, rec: CallRecord, msg: str) -> StepOutcome:
        self.st.records.append(rec)
        return StepOutcome(result={"error": msg}, ok=False)

    def step(self, call: ToolCall) -> StepOutcome:
        st = self.st
        if call.malformed:
            return self._err(
                CallRecord("?", False, "malformed_call"),
                f"could not parse tool call: {call.malformed}",
            )
        canon = self.canon_of.get(call.name)
        if canon is None:
            if any(d.name == call.name for d in DISTRACTORS):
                return self._err(CallRecord("distractor", False, "distractor"), "this tool is not relevant here")
            return self._err(CallRecord("?", False, "unknown_tool"), f"unknown tool '{call.name}'")
        expected = st.args[canon]
        args: dict[str, Any] = {}
        for k, v in call.args.items():
            ck = next((c for c, n in expected.items() if n == k), None)
            if ck is None:
                return self._err(
                    CallRecord(canon, False, "bad_args"),
                    f"unknown argument '{k}'. Expected: {sorted(expected.values())}",
                )
            args[ck] = v

        n = len(st.records)
        # Injected errors (never on terminal tools, so the episode can recover).
        if canon not in ("submit", "impossible") and self.cfg.randomize:
            if st.stale_at is not None and n >= st.stale_at and not st.stale_pending and canon != "view":
                st.stale_at = None
                st.stale_pending = True
                st.token = secrets.token_hex(3)
                return self._err(CallRecord(canon, False, "stale"),
                                 "STALE_STATE: the position was refreshed and your state token is no longer valid. "
                                 "View the board again before continuing.")  # fmt: skip
            roll = self.rng.random()
            if roll < self.cfg.p_timeout:
                return self._err(
                    CallRecord(canon, False, "timeout"),
                    "timeout: the board service did not respond; retry",
                )
            if roll < self.cfg.p_timeout + self.cfg.p_malformed:
                st.records.append(CallRecord(canon, False, "malformed_response"))
                return StepOutcome(result='{"partial": tr', ok=False)

        if canon in STATE_TOOLS:
            if st.stale_pending:
                return self._err(
                    CallRecord(canon, False, "acted_on_stale"),
                    "STALE_STATE: view the board again to get a fresh state token.",
                )
            if str(args.get("state", "")) != st.token:
                return self._err(
                    CallRecord(canon, False, "bad_token"),
                    "invalid state token; view the board to get the current one",
                )
        return getattr(self, f"_t_{canon}")(args)

    def _t_view(self, args: dict) -> StepOutcome:
        self.st.stale_pending = False
        self.st.records.append(CallRecord("view", True))
        return StepOutcome(result=self._board_repr())

    def _t_moves(self, args: dict) -> StepOutcome:
        moves = list(self.board.legal_moves)
        sq = args.get("square")
        if sq:
            try:
                s = chess.parse_square(str(sq).strip().lower())
            except ValueError:
                return self._err(CallRecord("moves", False, "bad_args"), f"invalid square '{sq}'")
            moves = [m for m in moves if m.from_square == s]
        if args.get("captures_only") in (True, "true", "True", 1, "1"):
            moves = [m for m in moves if self.board.is_capture(m)]
        self.st.records.append(CallRecord("moves", True))
        return StepOutcome(result={"count": len(moves), "moves": [self._fmt_move(m) for m in moves]})

    def _t_piece(self, args: dict) -> StepOutcome:
        try:
            s = chess.parse_square(str(args.get("square", "")).strip().lower())
        except ValueError:
            return self._err(CallRecord("piece", False, "bad_args"), f"invalid square '{args.get('square')}'")
        p = self.board.piece_at(s)
        self.st.records.append(CallRecord("piece", True))
        if p is None:
            return StepOutcome(result={"square": chess.square_name(s), "piece": None})
        pinned = p.piece_type != chess.KING and self.board.is_pinned(p.color, s)
        return StepOutcome(result={
            "square": chess.square_name(s), "piece": chess.piece_name(p.piece_type),
            "color": "white" if p.color else "black", "pinned_to_king": pinned,
            "attacked_by_enemy": len(self.board.attackers(not p.color, s)),
        })  # fmt: skip

    def _t_simulate(self, args: dict) -> StepOutcome:
        m = self._parse_move(str(args.get("move", "")))
        if m is None or not self.board.is_legal(m):
            self.st.records.append(CallRecord("simulate", True, move=None))
            return StepOutcome(result={"legal": False, "move": args.get("move")})
        b = self.board.copy()
        cap = b.piece_at(m.to_square)
        b.push(m)
        self.st.verified_moves.add(m.uci())
        self.st.records.append(CallRecord("simulate", True, move=m.uci()))
        return StepOutcome(result={"legal": True, "move": self._fmt_move(m), "gives_check": b.is_check(),
                                   "checkmate": b.is_checkmate(),
                                   "captures": chess.piece_name(cap.piece_type) if cap else None,
                                   "fen_after": b.fen()})  # fmt: skip

    def _t_submit(self, args: dict) -> StepOutcome:
        ans = str(args.get("answer", "")).strip()
        rec = CallRecord("submit", True)
        if self.task.answer_type == "move" or self.task.kind in ("mate1_none",):
            m = self._parse_move(ans)
            rec.move = m.uci() if m else None
        self.st.records.append(rec)
        self.st.final_answer = ans
        self.st.decision = "answer"
        return StepOutcome(result={"submitted": ans}, done=True, decision="answer")

    def _t_impossible(self, args: dict) -> StepOutcome:
        self.st.records.append(CallRecord("impossible", True))
        self.st.final_answer = "impossible"
        self.st.decision = "impossible"
        return StepOutcome(result={"acknowledged": True}, done=True, decision="impossible")

    def _t_expert(self, args: dict) -> StepOutcome:
        self.st.escalated = True
        self.st.records.append(CallRecord("expert", True))
        answer = self.expert.answer(self.task, self.board, self.st.notation) if self.expert else "expert unavailable"
        return StepOutcome(result={"expert": answer})

    def on_text(self, text: str) -> StepOutcome:
        self.st.records.append(CallRecord("?", False, "no_tool_call"))
        return StepOutcome(result="Reply with exactly one tool call.", ok=False)

    def finish(self, reason: str) -> StepOutcome:
        self.st.decision = self.st.decision or reason
        return StepOutcome(result={"ended": reason}, done=True, decision=self.st.decision)


# ---- reward (pure function of task + records) -----------------------------------
@dataclass
class RewardBreakdown:
    total: float
    correct: bool
    parts: dict[str, float]


def is_correct(task: ChessTask, st: EpisodeState) -> bool:
    if task.answer_type == "impossible":
        return st.decision == "impossible"
    if st.decision != "answer" or st.final_answer is None:
        return False
    ans = st.final_answer.strip().lower()
    if task.answer_type == "bool":
        norm = {"yes": "yes", "true": "yes", "y": "yes", "no": "no", "false": "no", "n": "no"}
        return norm.get(ans.split()[0].strip(".,!") if ans else "", None) == task.answer
    if task.answer_type == "int":
        digits = "".join(ch for ch in ans if ch.isdigit())
        return digits != "" and int(digits) == int(task.answer)
    if task.answer_type == "move":
        sub = next((r for r in reversed(st.records) if r.canon == "submit"), None)
        return bool(sub and sub.move in task.accept_moves)
    return False


def reward(task: ChessTask, st: EpisodeState, expert_cost: float = 0.3) -> RewardBreakdown:
    parts: dict[str, float] = {}
    correct = is_correct(task, st)
    if st.escalated:
        parts["escalation"] = (0.6 if correct else 0.0) - expert_cost if task.hard else -expert_cost
    else:
        parts["answer"] = 1.0 if correct else 0.0

    recs = st.records
    parts["malformed"] = -0.2 * sum(1 for r in recs if r.error in ("malformed_call", "unknown_tool",
                                                                   "bad_args", "no_tool_call"))  # fmt: skip
    if any(r.error in ("acted_on_stale",) for r in recs):
        parts["stale"] = -0.5
    final_idx = next((i for i in range(len(recs) - 1, -1, -1) if recs[i].canon in ("submit", "impossible")), None)
    if final_idx is not None:
        before = recs[:final_idx]
        observed = any(r.ok and r.canon in OBSERVE_TOOLS for r in before)
        verified = observed
        if task.answer_type == "move" and recs[final_idx].canon == "submit":
            verified = recs[final_idx].move is not None and recs[final_idx].move in st.verified_moves
        if task.kind == "engine_eval":
            verified = st.escalated  # no local tool can verify an engine evaluation
        if not verified:
            parts["unverified"] = -0.4
        # Claiming success after a tool error: the last call failed and the answer
        # was never backed by a successful call of that same tool.
        if (
            before
            and not before[-1].ok
            and before[-1].error in ("timeout", "malformed_response", "stale", "acted_on_stale", "bad_token")
        ):
            failed = before[-1].canon
            if not any(r.ok and r.canon == failed for r in before[:-1]):
                parts["claim_after_error"] = -1.0
    total = round(sum(parts.values()), 4)
    return RewardBreakdown(total=total, correct=correct, parts=parts)


class StockfishExpert:
    """Expert agent used in chess training: Stockfish for moves, python-chess for facts."""

    def __init__(self) -> None:
        from tth_gym.envs.chess.tasks import open_engine

        self.engine = open_engine()

    def answer(self, task: ChessTask, board: chess.Board, notation: str) -> str:
        def fmt(m: chess.Move) -> str:
            return board.san(m) if notation == "san" else m.uci()

        if task.kind in ("mate1", "mate1_none"):
            from tth_gym.envs.chess.tasks import mating_moves

            mates = mating_moves(board)
            return f"Mating move: {fmt(chess.Move.from_uci(mates[0]))}" if mates else "There is no mate in one."
        if task.answer_type == "impossible":
            return "The question refers to something that is not on the board."
        if task.kind == "engine_eval" and self.engine:
            info = self.engine.analyse(board, chess.engine.Limit(depth=14))
            cp = info["score"].white().score(mate_score=10000)
            return f"Engine evaluation: {cp / 100:+.2f} pawns (positive favours White)."
        if task.kind == "best_move" and self.engine:
            info = self.engine.analyse(board, chess.engine.Limit(depth=14))
            return f"Best move: {fmt(info['pv'][0])}"
        return f"Answer: {task.answer}"

    def close(self) -> None:
        if self.engine:
            self.engine.quit()


def episode_summary(env: ChessEnv) -> dict[str, Any]:
    rb = reward(env.task, env.st, env.cfg.expert_cost)
    return {"task_id": env.task.id, "kind": env.task.kind, "hard": env.task.hard, "correct": rb.correct,
            "reward": rb.total, "parts": rb.parts, "escalated": env.st.escalated, "decision": env.st.decision,
            "calls": len(env.st.records)}  # fmt: skip


def dumps(x: Any) -> str:
    return json.dumps(x)
