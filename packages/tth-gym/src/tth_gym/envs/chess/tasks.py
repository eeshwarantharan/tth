"""Short, verifiable chess tasks. Ground truth comes from python-chess / Stockfish."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.engine

PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}  # fmt: skip

KINDS = [
    "legal",
    "check",
    "captures",
    "pinned",
    "mate1",
    "mate1_none",
    "missing_piece",
    "best_move",
]


@dataclass
class ChessTask:
    id: str
    kind: str
    fen: str
    question: str
    answer_type: str  # bool | int | move | impossible
    answer: str  # canonical: "yes"/"no", "3", "e2e4" (one accepted move), "impossible"
    accept_moves: list[str] = field(default_factory=list)  # all accepted UCI moves
    hard: bool = False  # beyond local ability: escalation is the right call
    best_cp: int | None = None  # for best_move: engine score of the best move

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(line: str) -> ChessTask:
        return ChessTask(**json.loads(line))


def random_position(rng: random.Random, min_ply: int = 8, max_ply: int = 50) -> chess.Board:
    while True:
        b = chess.Board()
        for _ in range(rng.randint(min_ply, max_ply)):
            moves = list(b.legal_moves)
            if not moves:
                break
            # Bias towards captures and checks so positions get tactical.
            weights = [3 if b.is_capture(m) or b.gives_check(m) else 1 for m in moves]
            b.push(rng.choices(moves, weights)[0])
        if not b.is_game_over() and len(b.piece_map()) >= 8:
            return chess.Board(b.fen())


def mating_moves(b: chess.Board) -> list[str]:
    out = []
    for m in b.legal_moves:
        b.push(m)
        if b.is_checkmate():
            out.append(m.uci())
        b.pop()
    return out


def is_pinned(b: chess.Board, sq: int) -> bool:
    p = b.piece_at(sq)
    return bool(p) and p.piece_type != chess.KING and b.is_pinned(p.color, sq)  # type: ignore[union-attr]


def _side(b: chess.Board) -> str:
    return "White" if b.turn == chess.WHITE else "Black"


def _yn(x: bool) -> str:
    return "yes" if x else "no"


def _illegal_candidate(b: chess.Board, rng: random.Random) -> chess.Move | None:
    pseudo = [m for m in b.pseudo_legal_moves if not b.is_legal(m)]
    if pseudo:
        return rng.choice(pseudo)
    own = [sq for sq, p in b.piece_map().items() if p.color == b.turn]
    for _ in range(50):
        m = chess.Move(rng.choice(own), rng.randrange(64))
        if m.from_square != m.to_square and not b.is_legal(m):
            return m
    return None


def make_task(
    kind: str, rng: random.Random, idx: str, engine: chess.engine.SimpleEngine | None = None
) -> ChessTask | None:  # noqa: E501
    b = random_position(rng)
    fen = b.fen()
    if kind == "legal":
        if rng.random() < 0.5:
            m, legal = rng.choice(list(b.legal_moves)), True
        else:
            cand = _illegal_candidate(b, rng)
            if cand is None:
                return None
            m, legal = cand, False
        return ChessTask(idx, kind, fen, f"Is the move {m.uci()} (UCI notation) legal in the current position?",
                         "bool", _yn(legal))  # fmt: skip
    if kind == "check":
        want = rng.random() < 0.5
        for _ in range(200):
            if b.is_check() == want:
                break
            b = random_position(rng)
        return ChessTask(idx, kind, b.fen(), "Is the side to move currently in check?", "bool", _yn(b.is_check()))
    if kind == "captures":
        n = sum(1 for m in b.legal_moves if b.is_capture(m))
        return ChessTask(idx, kind, fen, "How many legal capturing moves does the side to move have? Answer with a number.",
                         "int", str(n))  # fmt: skip
    if kind == "pinned":
        want = rng.random() < 0.5
        for _ in range(400):
            own = [sq for sq, p in b.piece_map().items() if p.color == b.turn and p.piece_type != chess.KING]
            pinned = [sq for sq in own if is_pinned(b, sq)]
            pool = pinned if want else [sq for sq in own if sq not in pinned]
            if pool:
                sq = rng.choice(pool)
                p = b.piece_at(sq)
                assert p
                return ChessTask(idx, kind, b.fen(),
                                 f"Is the {PIECE_NAMES[p.piece_type]} on {chess.square_name(sq)} pinned to its own king?",
                                 "bool", _yn(is_pinned(b, sq)))  # fmt: skip
            b = random_position(rng)
        return None
    if kind in ("mate1", "mate1_none"):
        want = kind == "mate1"
        for _ in range(3000):
            mates = mating_moves(b)
            if bool(mates) == want:
                q = f"{_side(b)} to move. Find a move that delivers checkmate immediately and submit it."
                if want:
                    return ChessTask(idx, kind, b.fen(), q, "move", mates[0], accept_moves=mates)
                # Proving no mate exists needs every move checked: beyond a 6-call budget.
                return ChessTask(idx, kind, b.fen(), q, "impossible", "impossible", hard=True)
            b = random_position(rng, 10, 70)
        return None
    if kind == "missing_piece":
        empty = [sq for sq in chess.SQUARES if b.piece_at(sq) is None]
        sq = rng.choice(empty)
        name = rng.choice(["knight", "bishop", "rook", "queen"])
        return ChessTask(idx, kind, fen, f"Submit a legal move for the {name} on {chess.square_name(sq)}.",
                         "impossible", "impossible")  # fmt: skip
    if kind == "best_move":
        if engine is None:
            return None
        for _ in range(60):
            if not mating_moves(b):
                info = engine.analyse(b, chess.engine.Limit(depth=12), multipv=3)
                if len(info) >= 2:
                    s1 = info[0]["score"].pov(b.turn).score(mate_score=10000)
                    s2 = info[1]["score"].pov(b.turn).score(mate_score=10000)
                    if s1 is not None and s2 is not None and s1 - s2 >= 250 and -100 <= s1 <= 1500:
                        best = info[0]["pv"][0].uci()
                        return ChessTask(idx, kind, b.fen(),
                                         f"{_side(b)} to move. There is one clearly best move (a tactic). Find it and submit it.",
                                         "move", best, accept_moves=[best], hard=True, best_cp=s1)  # fmt: skip
            b = random_position(rng)
        return None
    if kind == "engine_eval":
        if engine is None:
            return None
        for _ in range(60):
            info = engine.analyse(b, chess.engine.Limit(depth=14))
            cp = info["score"].white().score(mate_score=10000)
            if cp is not None and abs(abs(cp) - 200) >= 80:  # keep clear-cut cases
                return ChessTask(idx, kind, b.fen(),
                                 "According to a strong chess engine, is White better by more than two pawns "
                                 "(+2.00) in this position? Answer yes or no.", "bool", _yn(cp > 200),
                                 hard=True, best_cp=cp)  # fmt: skip
            b = random_position(rng)
        return None
    raise ValueError(kind)


def open_engine() -> chess.engine.SimpleEngine | None:
    path = shutil.which("stockfish")
    return chess.engine.SimpleEngine.popen_uci(path) if path else None


DEFAULT_MIX = {"legal": 3, "check": 2, "captures": 2, "pinned": 2, "mate1": 2, "mate1_none": 1,
               "missing_piece": 1, "best_move": 2}  # fmt: skip


def generate(n: int, seed: int, mix: dict[str, int] | None = None, prefix: str = "c") -> Iterator[ChessTask]:
    mix = mix or DEFAULT_MIX
    rng = random.Random(seed)
    kinds = [k for k, w in mix.items() for _ in range(w)]
    engine = open_engine() if ("best_move" in mix or "engine_eval" in mix) else None
    try:
        made = 0
        while made < n:
            kind = kinds[made % len(kinds)]
            t = make_task(kind, rng, f"{prefix}{seed}-{made}", engine)
            if t:
                made += 1
                yield t
    finally:
        if engine:
            engine.quit()


def save(tasks: list[ChessTask], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(t.to_json() + "\n" for t in tasks))


def load(path: str | Path) -> list[ChessTask]:
    return [ChessTask.from_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
