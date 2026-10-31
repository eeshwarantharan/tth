"""Render demo/tth.gif from REAL recorded eval traces (no hand-made frames of model behaviour).

Scenes: (1) chess: board + live tool log, with a rejected/failed call then recovery; (2) chess
escalation; (3) shell: inspect -> probe -> dry_run -> stage with tier badge; (4) shell escalation.
The episode ids used are written to demo/manifest.json so anyone can check them against
evals/results/traces/.  Run after `make eval`:  uv run python -m demo.render [--arm C_chess_shell]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import chess
from PIL import Image, ImageDraw, ImageFont

RES = Path("evals/results")
W, H, SQ = 1100, 480, 52
BG, FG, DIM, GREEN, RED, YELLOW = (
    (18, 18, 24),
    (230, 230, 230),
    (130, 130, 140),
    (80, 200, 120),
    (230, 90, 90),
    (230, 190, 80),
)
LIGHT, DARK, HL = (238, 216, 180), (181, 136, 99), (120, 180, 255)
GLYPH = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}


def font(size: int, symbols: bool = False) -> Any:
    for p in (["/System/Library/Fonts/Apple Symbols.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] if symbols else []) + [
        "/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:  # fmt: skip
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F, FS, FB, FP = font(15), font(12), font(22), font(40, symbols=True)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def events(arm: str, suite: str, episode: str) -> list[dict]:
    return [e for e in jl(RES / "traces" / f"{arm}__{suite}.jsonl") if e["episode"] == episode]


def pick(rows: list[dict], pred) -> dict | None:
    return next((r for r in rows if pred(r)), None)


def log_line(e: dict) -> tuple[str, tuple]:
    c = e.get("call") or {}
    arg = next(iter((c.get("args") or {}).values()), "") if c else ""
    res = e.get("result")
    if isinstance(res, dict) and "error" in res:
        tail = "error: " + str(res["error"])[:40]
    elif isinstance(res, dict):
        tail = ", ".join(f"{k}={str(v)[:18]}" for k, v in list(res.items())[:2])
    else:
        tail = str(res).replace("\n", " ")[:42]
    return f"{'✓' if e['ok'] else '✗'} {c.get('tool', '?'):<15} {str(arg)[:28]:<28} → {tail}", GREEN if e["ok"] else RED


def draw_log(d: ImageDraw.ImageDraw, x: int, title: str, lines: list[tuple[str, tuple]]) -> None:
    d.text((x, 20), title, font=FB, fill=FG)
    for i, (t, col) in enumerate(lines[-14:]):
        d.text((x, 64 + i * 26), t[:72], font=F, fill=col)


def draw_board(d: ImageDraw.ImageDraw, b: chess.Board, hl: set[int]) -> None:
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), 7 - chess.square_rank(sq)
        x, y = 30 + f * SQ, 40 + r * SQ
        col = HL if sq in hl else (LIGHT if (f + r) % 2 == 0 else DARK)
        d.rectangle([x, y, x + SQ, y + SQ], fill=col)
        p = b.piece_at(sq)
        if p:
            d.text((x + 8, y + 2), GLYPH[p.symbol()], font=FP, fill=(0, 0, 0))


def squares_of(arg: str, b: chess.Board) -> set[int]:
    for parse in (b.parse_uci, b.parse_san):
        try:
            m = parse(arg)
            return {m.from_square, m.to_square}
        except ValueError:
            continue
    try:
        return {chess.parse_square(arg)}
    except ValueError:
        return set()


def chess_scene(arm: str, ep: dict, tasks: dict[str, dict], title: str) -> list[Image.Image]:
    b = chess.Board(tasks[ep["task_id"]]["fen"])
    frames, lines = [], [(f"task: {tasks[ep['task_id']]['question'][:64]}", DIM)]
    for e in events(arm, "chess", ep["episode"]):
        if e.get("call"):
            lines.append(log_line(e))
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        args = (e.get("call") or {}).get("args") or {}
        draw_board(d, b, set().union(*[squares_of(str(v), b) for v in args.values()]) if args else set())
        draw_log(d, 480, title, lines)
        frames.append(img)
    return frames


def shell_scene(arm: str, ep: dict, title: str, request: str) -> list[Image.Image]:
    frames, lines = [], [(f"$ {request[:66]}", FG)]
    for e in events(arm, "shell", ep["episode"]):
        if e.get("call"):
            lines.append(log_line(e))
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        draw_log(d, 30, title, lines)
        res = e.get("result") if isinstance(e.get("result"), dict) else {}
        if e.get("decision") == "stage" and res.get("command"):
            tier = res.get("tier", "")
            col = {"read-only": GREEN, "changes-files": YELLOW}.get(tier, RED)
            d.rectangle([30, H - 70, W - 30, H - 25], outline=col, width=2)
            d.text((44, H - 60), f"[{tier}]  {res['command'][:80]}", font=F, fill=col)
        elif e.get("decision") == "escalate":
            d.text(
                (44, H - 60), "can't verify locally → escalate (payload shown, user must consent)", font=F, fill=YELLOW
            )
        frames.append(img)
    return frames


ARCH = [
    ("1 · TEACHER makes demonstrations", ["Nemotron 3.5 (API) + scripted oracle play chess / table / shell tasks",
     "through the SAME tools the student will use: view → simulate → submit, probe → dry_run → stage"]),
    ("2 · VERIFIER keeps only correct episodes", ["python-chess · Stockfish · sandboxed execution of the staged command",
     "✓ correct + zero penalties → training data      ✗ everything else → discarded"]),
    ("3 · STUDENT learns (Qwen2.5-Coder 1.5B + LoRA)", ["SFT on verified episodes, then GRPO: try each task 4×,",
     "verifier scores each try, better-than-average tries are reinforced", "then a short shell SFT — the same for arms B, C, D"]),
    ("4 · PRODUCT on your Mac", ["student runs locally: inspect → probe → dry_run (sandbox) → stage on your prompt",
     "never executes · can't verify? → escalate to EXPERT (Nemotron via A2A) after your consent"]),
    ("5 · Is it chess? Four arms, same shell eval", ["A bare model · B shell only · C chess → shell · D non-chess table → shell",
     "C > B and C > D ⇒ chess helps   ·   C ≈ D ⇒ any tool practice helps"]),
]  # fmt: skip


def arch_frames() -> list[Image.Image]:
    frames = []
    for i in range(len(ARCH)):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((30, 14), "how tth is built  (diagram, not model output)", font=FS, fill=DIM)
        for j, (title, body) in enumerate(ARCH[: i + 1]):
            y = 40 + j * 86
            col = FG if j == i else DIM
            d.rectangle([30, y, W - 30, y + 78], outline=(GREEN if j == i else DIM), width=2)
            d.text((44, y + 6), title, font=FB if j == i else F, fill=col)
            for k, line in enumerate(body[:2]):
                d.text((44, y + 36 + k * 19), line, font=FS, fill=col)
        frames += [img, img]
    return frames


def reward_frame() -> Image.Image | None:
    logs = [(n, Path(f"adapters/{n}/grpo_log.json")) for n in ("C0_chess_grpo", "D_table_grpo")]
    logs = [(n, json.loads(p.read_text())) for n, p in logs if p.exists()]
    if not logs:
        return None
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 20), "GRPO: verifier reward per round (real training log)", font=FB, fill=FG)
    for li, (name, rounds) in enumerate(logs):
        y = 90 + li * 160
        d.text((30, y), name, font=F, fill=DIM)
        for r in rounds:
            v = max(-1.0, min(1.0, r["mean_reward"]))
            x0 = 260 + r["round"] * 220
            d.rectangle([x0, y + 110 - int(100 * max(v, 0)), x0 + 120, y + 110], fill=GREEN)
            d.text(
                (x0, y + 118),
                f"round {r['round']}: reward {r['mean_reward']:+.2f}, acc {r['accuracy']:.0%}",
                font=FS,
                fill=FG,
            )
    return img


def results_frame() -> Image.Image | None:
    p = RES / "results.json"
    if not p.exists():
        return None
    res = json.loads(p.read_text())
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 20), "held-out shell accuracy by arm (95% CI)  — evals/results/RESULTS.md", font=FB, fill=FG)
    for i, (arm, m) in enumerate((a, r["shell"]) for a, r in res.items() if "shell" in r):
        y = 80 + i * 60
        d.text((30, y + 8), arm, font=F, fill=FG)
        w = int(600 * m["accuracy"])
        d.rectangle([260, y, 260 + w, y + 36], fill=GREEN if arm.startswith("C") else (90, 140, 220))
        lo, hi = m["ci"]
        d.line([260 + int(600 * lo), y + 18, 260 + int(600 * hi), y + 18], fill=FG, width=3)
        d.text((880, y + 8), f"{m['accuracy']:.0%}", font=F, fill=FG)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C_chess_shell")
    ap.add_argument("--chess-arm", default="C0_chess")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in jl(Path("evals/chess/tasks.jsonl"))}
    shell_tasks = {t["id"]: t for t in jl(Path("evals/shell/tasks.jsonl"))}
    crows, srows = jl(RES / f"{args.chess_arm}__chess.jsonl"), jl(RES / f"{args.arm}__shell.jsonl")
    if not crows or not srows:
        raise SystemExit("no eval results for these arms yet: run `make eval` first")

    def has_fail(arm: str, suite: str, ep: dict) -> bool:
        return any(not e["ok"] and e.get("call") for e in events(arm, suite, ep["episode"]))

    picks = {
        "chess_recover": pick(crows, lambda r: r["correct"] and not r["hard"] and has_fail(args.chess_arm, "chess", r))
        or pick(crows, lambda r: r["correct"] and not r["hard"]),
        "chess_escalate": pick(crows, lambda r: r["correct"] and r["hard"] and r["escalated"]),
        "shell_stage": pick(
            srows, lambda r: r["correct"] and r["decision"] == "stage" and has_fail(args.arm, "shell", r)
        )
        or pick(srows, lambda r: r["correct"] and r["decision"] == "stage"),
        "shell_escalate": pick(srows, lambda r: r["decision"] == "escalate" and r["expected"] == "escalate"),
    }
    frames: list[Image.Image] = arch_frames()
    rf = reward_frame()
    frames += [rf, rf] if rf else []
    if picks["chess_recover"]:
        frames += chess_scene(
            args.chess_arm,
            picks["chess_recover"],
            tasks,
            f"student · arm {args.chess_arm} · the board: look, simulate, commit",
        )
    if picks["chess_escalate"]:
        frames += chess_scene(
            args.chess_arm,
            picks["chess_escalate"],
            tasks,
            f"student · arm {args.chess_arm} · knowing its limits: escalate",
        )
    if picks["shell_stage"]:
        frames += shell_scene(args.arm, picks["shell_stage"], f"student · arm {args.arm} · same habits in your terminal",
                              shell_tasks[picks["shell_stage"]["task_id"]]["request"])  # fmt: skip
    if picks["shell_escalate"]:
        frames += shell_scene(args.arm, picks["shell_escalate"], f"student · arm {args.arm} · escalation in the terminal",
                              shell_tasks[picks["shell_escalate"]["task_id"]]["request"])  # fmt: skip
    rs = results_frame()
    frames += [rs, rs, rs] if rs else []
    end = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(end).text((W // 2 - 200, H // 2 - 30), "tth — small but good.", font=font(36), fill=FG)
    ImageDraw.Draw(end).text((W // 2 - 150, H // 2 + 30), "uv sync && uv run tth setup", font=F, fill=DIM)
    frames += [end]
    out = Path("demo/tth.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[1300] * (len(frames) - 1) + [3000], loop=0)
    manifest = {k: (v and {"arm": args.chess_arm if k.startswith("chess") else args.arm, "episode": v["episode"],
                           "task_id": v["task_id"]}) for k, v in picks.items()}  # fmt: skip
    Path("demo/manifest.json").write_text(json.dumps(manifest, indent=1))
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-movflags", "faststart", "-pix_fmt",
                        "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "demo/tth.mp4"], check=False)  # fmt: skip
    print(
        f"wrote {out} ({len(frames)} frames) and demo/manifest.json; missing scenes: "
        f"{[k for k, v in picks.items() if not v] or 'none'}"
    )


if __name__ == "__main__":
    main()
