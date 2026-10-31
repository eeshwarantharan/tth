# CLAUDE.md — tth (tiny terminal helper)

> **tth — small but good.** A local terminal agent that looks before it acts and knows when to call for help.

## What we're building

A small on-device model (~1.5B) that turns plain-English requests into shell commands **by inspecting the real machine first** and **staging** the result on the user's prompt. It never runs commands itself. When it can't verify a task, it **escalates** to a stronger agent over A2A, after showing the user exactly what would be sent.

The model learns this discipline in a **chess training environment** (hidden board, verifiable moves, tool calls with randomized schemas). It then transfers the habit to the shell with a short fine-tune. Chess is the training ground; the terminal is the product.

### The pitch, in one line
Most terminal requests are easy, so don't pay a frontier model for them. tth handles easy tasks locally and escalates hard ones.

### Protocol roles
- **MCP**: how the model sees the world (shell probes in the product; board tools in training).
- **A2A**: how the model asks for help (an expert agent: a larger local model or a cloud model).
- **Teacher model**: generates training trajectories (verified before use) and serves as the expert agent.

## Non-negotiable invariants

Never weaken these without explicit approval in the conversation. If a task seems to require it, stop and ask.

1. **tth never executes the final command.** The final command is written to the shell's input buffer. No `subprocess.run`, `os.system` or equivalent on the final command, ever.
2. **Probes are read-only and offline.** Every probe runs inside the sandbox (Linux: `bwrap` with read-only binds and no network; macOS: `sandbox-exec` deny-write and deny-network profile). Probes have a timeout (default 3s) and an output cap (default 8 KB).
3. **Safety is allowlist-based.** The command classifier parses commands into an AST (pipes, subshells, `-exec`, redirections) and **default-denies** anything it doesn't recognize. Opaque constructs (`eval`, `bash -c`, `python -c`, `xargs <cmd>`, scripts, `make`, package-manager runs) are never shown as safe.
4. **Destructive commands are never staged raw.** Deletions become a preview (list matches) or a move-to-trash. `mv`/`cp` get no-clobber. Any rewrite is shown to the user, never applied silently.
5. **Escalation requires consent.** Before any A2A call, show the exact payload. Redact secrets (env values, tokens, keys, `.env` contents, `~/.ssh`). The user can decline; the default is local-only.
6. **Secret paths are never probed.** `~/.ssh`, `~/.aws`, `~/.gnupg`, keychains, `.env*`, browser profiles. This is an extra layer on top of the sandbox.
7. **No fabricated results.** Never invent eval numbers, benchmark scores or demo footage. If an eval wasn't run, say so.

## Repository layout

```
src/tth/
  cli/          # entrypoint, shell integration (zsh buffer prefill; bash via bind -x)
  agent/        # model ↔ tools loop, step budget, escalation policy
  tools/        # MCP server exposing read-only shell probes
  safety/       # parser, classifier (tiers), rewriter, red-team cases
  sandbox/      # bwrap / sandbox-exec wrappers
  escalate/     # A2A client, payload preview, redaction
  runtime/      # llama.cpp (GGUF) and MLX backends
  trace/        # unified trajectory schema + logger (shared by product and training)
train/
  envs/chess/   # chess environment: tasks, verifiers, schema randomizer, error injection
  envs/shell/   # containerized shell tasks with execution-based checks
  data/         # teacher trajectory generation + verification
  sft/  grpo/   # training configs and scripts
experts/        # A2A expert agents used in training and in the product
evals/          # eval harness + task suites
demo/           # trace-driven GIF renderer
tests/
```

## Stack

- Python 3.11+, `uv`, `ruff`, `pyright`, `pytest`
- `python-chess` + Stockfish (training verifiers; never used as a judge model)
- Official MCP Python SDK (`mcp`) and the official A2A Python SDK
- Training: TRL (SFT, GRPO) + PEFT/LoRA; vLLM for rollouts when available
- Base model: Qwen2.5-Coder-1.5B-Instruct (Apache-2.0). Keep the license notice in all releases.
- Inference: llama.cpp (GGUF) and MLX

## Commands

```bash
uv sync                          # install
uv run pytest                    # all tests
uv run pytest tests/safety       # safety suite (must always pass)
uv run ruff check . && uv run ruff format .
uv run pyright
make eval                        # full eval harness
make demo                        # render the demo GIF from recorded traces
```

Before reporting any task as done: tests, lint and type check pass, and the safety suite passes.

## Core design

### Agent loop (product)
1. `inspect_context`: OS, shell, working directory listing, git branch/status.
2. `probe`: head of a file, CSV header, JSON keys, `--help` for a binary.
3. `dry_run`: runs the read/selection part of the command in the sandbox; returns match count, sample output or the error.
4. Decide: **stage** (verified) / **escalate** (can't verify) / **ask** (request is ambiguous).
5. `stage`: classifier + rewriter, then write to the shell buffer with a tier badge (read-only / changes files / needs review).

Step budget: default 6 tool calls before forced escalate-or-ask.

### Training environment (chess)
- The board is never in the prompt; the model must call tools to see it.
- Tools mirror the product loop: inspect → query constraints → simulate → commit, plus an expert agent over A2A at a cost.
- **Schema randomization every episode:** tool names, argument names and order, descriptions, output formats (FEN / ASCII / JSON). The model must read schemas, not memorize them.
- **Error injection:** stale state tokens, timeouts, malformed responses, distractor tools.
- Task types: short verifiable questions and puzzles (legality, pins, captures, mate-in-1, impossible tasks). No full games as reward.

### Rewards
| Event | Reward |
|---|---|
| Correct final answer / committed move | +1.0 |
| Correct escalation (task beyond local ability) | +0.6 minus escalation cost |
| Unnecessary escalation | escalation cost only (default −0.3) |
| Correct "can't be done" on impossible tasks | +1.0 |
| Commit without verifying (even if correct) | −0.4 |
| Acting on stale state | −0.5 |
| Malformed tool call | −0.2 (episode continues) |
| Claiming success after a tool error | −1.0 |

Verifiers (`python-chess`, Stockfish, container execution) assign reward. The teacher model never does.

### Training pipeline
1. Teacher generates trajectories with the same tools → keep only verifier-approved ones.
2. SFT on verified chess trajectories.
3. GRPO in the chess environment.
4. Short shell fine-tune (SFT, optionally GRPO) on containerized shell tasks.
5. Ablations: (a) base model, (b) shell-only training, (c) chess → shell, (d) same-size non-chess synthetic tool data → shell.

### Unified trace format
Product runs and training rollouts write the same JSONL schema, so evals and the demo share one renderer:
```json
{"episode": "id", "domain": "chess|shell", "step": 3,
 "call": {"tool": "dry_run", "args": {}}, "result": {}, "ok": true,
 "decision": null, "reward": null, "ts": 0}
```

## Evaluation

Report all of these, including failures:
- % tasks handled locally; accuracy vs. frontier-only baseline
- Escalation precision / recall
- Cost and latency per task
- Unsafe commands staged (target: 0 on the red-team suite; report the actual number)
- Transfer: BFCL / Tau2-Bench subsets for general tool use
- The ablation table from the training pipeline

Never tune on eval tasks. Held-out splits live in `evals/` and are never imported by `train/`.

## The demo (this is what makes tth land)

One GIF, about 30–45s, rendered from **real recorded traces**. No hand-made frames of model behavior.

**Scene 1 — The board (≈12s).** Split screen: the chessboard on the left, the live tool-call log on the right. The model inspects the board, simulates a move, gets *rejected* (illegal or stale state), re-inspects, simulates again, commits. Each call lights up the matching piece or square.

**Scene 2 — Knowing its limits (≈6s).** A hard position. The log shows "can't verify locally → escalating"; the expert agent's answer arrives and the move is committed.

**Scene 3 — The same habits, in your terminal (≈15s).** A wipe transition to a terminal with the same log panel. Request: *"sum the token costs for failed tasks in metrics.jsonl"*. The log shows inspect → probe file (real key names appear) → dry-run (error, fixed) → dry-run (ok) → command appears on the prompt with a green read-only badge. The log rows line up visually with Scene 1's rows (inspect ↔ inspect, simulate ↔ dry-run, commit ↔ stage).

**Scene 4 — Escalation in the terminal (≈8s).** A hard request → consent prompt showing the redacted payload → user accepts → staged answer.

**End card:** *tth — small but good.* plus the install command.

Implementation:
- `demo/render.py` reads trace JSONL; board frames come from `python-chess` SVG → PNG; the log panel is rendered by the same code for both domains.
- Terminal scenes are recorded with VHS (`.tape` files in `demo/`) running the real CLI against fixture directories.
- Compose with ffmpeg; output `demo/tth.gif` plus an mp4. `make demo` regenerates everything from committed traces and tapes.
- Keep the selected trace IDs in `demo/manifest.json` so anyone can verify the footage is real.

## Milestones

| # | Goal | Done when |
|---|---|---|
| M0 | Safety core + shell probes + CLI staging with the untrained base model | Safety suite passes; baselines recorded |
| M1 | Chess environment with schema randomization + verifiers | Random and base-model baselines recorded |
| M2 | Teacher trajectories → verified → SFT | SFT beats base on held-out chess tasks |
| M3 | GRPO with escalation cost | Escalation precision/recall reported |
| M4 | Shell transfer + escalation over A2A | Full eval table incl. ablations |
| M5 | Demo + release (GGUF, MLX, model card, data card) | `make demo` reproduces the GIF; model card lists limits |

Each milestone must be useful on its own if the project stopped there.

## Working conventions

- Small, focused diffs. One concern per PR.
- Every safety change adds red-team test cases (`tests/safety/cases/*.yaml`), including the failure that motivated it.
- New tools must declare: read-only?, sandboxed?, timeout, output cap.
- Prefer verifiers and deterministic checks over model judgment everywhere.
- Log every rollout and product run in the unified trace format.
- When unsure whether something touches an invariant, ask before writing code.

## Release checklist

- [ ] Model card: base model + license, training data summary, eval table, known failure modes, safety behavior
- [ ] Data card for released trajectories (how they were generated and verified)
- [ ] GGUF + MLX builds with checksums
- [ ] `pip install tth` → `tth setup` works on a clean macOS and Linux machine
- [ ] Demo GIF regenerated from committed traces