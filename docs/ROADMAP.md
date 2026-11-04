# tth roadmap

**Definition:** tth is a terminal agent that runs on a ~1.5B model in under 1 GB of RAM. It inspects your machine, tests the command in a read-only sandbox, and puts it on your prompt without running it. When it can't verify a command, it asks a stronger agent, but only with your consent.

**Research question:** can a large model's verified reasoning make a small model better at terminal tasks that need reasoning, and does chess practice add anything?

The reasoning behind this plan is in the [November 2026 audit](analysis/2026-11-audit.md).

## Goal

These are targets. None has been met yet.

| Area | Done when |
|---|---|
| Size | A ~1.5B model stays under 1,000 MB on the M2, counting weights, working memory and the CLI process |
| Speed | Under 3s for easy tasks, under 10s for hard ones |
| Safety | 0 unsafe commands staged on the red-team suite, and tth never runs the final command |
| Reasoning transfer | Reasoning-trained beats actions-only on a held-out reasoning suite, across 3+ seeds, with a confidence interval that excludes zero |
| Chess | Chess practice beats a non-chess control and more shell training. Published either way |
| Public proof | Published scores on NL2SH-ALFA plus our own reasoning suite |

**Out of scope:** general coding, training on the user's device, firewalling other agents, agent swarms.

## Principles

1. **Everything must improve the small model, measurably.** If it doesn't, it doesn't ship.
2. **Test what ships.** Promotion runs on the final quantized build on the target machine.
3. **Make the model's job smaller.** Use the protocols and deterministic tools to take work off the model.
4. **The safety layer doesn't depend on the model.** It stays deterministic.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Truth and hygiene | Done |
| 1 | Safety hardening | Done |
| Site | Project page on GitHub Pages, updated after every phase | Live |
| 2 | Restructure into `tth` and `tth-gym` | Done |
| 3 | MCP and A2A at the core, grammar, llama.cpp, terminal UI | Done |
| 4 | Gym core, honest baselines, sub-1 GB bake-off, reasoning suite | Next: the sub-1 GB model choice and a reasoning suite remain |
| 5 | Reasoning transfer: teacher rationales, real chess play, preregistered test | Planned |
| 6 | Release | Planned |

## The improvement ladder

Each change is one rung, scored against the rung below it on the same held-out tasks, with paired bootstrap CIs and at least 3 seeds where training is involved. It ships only if it wins.

| Rung | Change | Result (100 held-out shell tasks unless noted) |
|---|---|---|
| L0 | Stock model (one per bake-off candidate) | Measured. Random baseline 5, current model 25, Qwen3-1.7B 26, Granite 4.1-3B 43 |
| L1 | + grammar generated from MCP tool schemas | **Off by default.** 7/20 either way, at 10 times the time |
| L2 | + verify, then commit, by construction | **Kept.** Runs out of attempts fell from 15–16 of 20 to 0–5 for trained adapters |
| L3 | + compact prompt and structured observations | Not run |
| L4 | + compiler tools (the model fills slots, tools write the command) | **Kept for Granite**, with prose reading: 43 → 58. Measured together, so not separated. The previous model went from 25 to 18 (paired interval −17 to +3) |
| L5 | + retrieved lessons (few-shot) | Not run |
| L6 | + SFT from oracle and teacher demos | Not run |
| L7 | + DAgger rounds | Not run |
| L8 | + RL (GRPO) | Not run |
| Q | Quantization matrix on the winner | Not run |

**Stop rule.** If the best student can't beat the best stock model under the size cap after at least 5k verified rows, training investment stops. tth then ships that stock model with the safety layer, and the gym stays as an open benchmark.

## The reasoning experiment

Training so far has only copied tool calls: [4 of 376 shell teacher turns contain any text](../evals/diagnostics/results/reasoning_in_data.json), and chess was mostly scripted single-position questions. So the core idea is untested, not failed.

- **Capture reasoning.** Keep the teacher's thinking, compress it to about 40 tokens per turn (goal, evidence, next step), sample many per task, and keep only what verifiers approve.
- **Play real chess.** The student plays with a short rationale for each move. At a blunder, the teacher explains why and gives a better move, and Stockfish scores it.
- **Test on tasks that need reasoning.** A new held-out suite covers error recovery, wrong assumptions about formats, multi-step deduction, ambiguity and knowing its limits. Tasks must be unsolvable without looking: hidden values are randomized per seed, so a blind guess fails.
- **Compare fairly.** Every arm gets the same training steps and tokens, with at least 3 seeds:

| Arm | Tests |
|---|---|
| Stock model | The baseline |
| Actions only | What we did before, done properly |
| Reasoning, on shell | Does reasoning transfer at all? |
| Chess reasoning, then shell | Does it cross domains? |
| Non-chess reasoning, then shell | Is it chess, or any reasoning practice? |
| More reasoning on shell, in place of the chess | Is chess better than simply more shell? |

NL2SH-ALFA is the external check that basic translation still works. It's mostly one-shot translation, so it can't show a reasoning benefit. The agent loop has to run in a way that matches how it scores.

## What the protocols are for

They don't make the model smarter on their own, and they aren't the security boundary. That's the sandbox, the classifier and the consent step. Their cost is small: about 49 MB and roughly half a second of import time for MCP, which is the next startup cost to cut. A2A loads only on escalation.

**MCP is the single tool contract.** Training and the product use the same tools, so what we train on is what ships. Any other agent can use the read-only probes through `tth mcp`.

**A2A makes the expert swappable.** Granite on this machine, a machine on the network or a cloud model, with no vendor lock-in, and every request for more evidence goes through consent. The gym doesn't use it yet: the chess teacher and DAgger takeover get built over A2A in Phase 5, so training and escalation share one handoff.

## Phases in detail

**Phase 1: Safety.** Add every known bypass as a red-team case first, then fix it. Walk heredocs and expansions, deny dynamic arguments, make allowlists flag-aware, refuse wrapped destructive forms, keep one source of truth for secrets, and redact JSON-aware before truncating.

**Phase 2: Restructure.** Two packages with one-way dependencies enforced in CI, a typed config file, dependency extras and working wheel installs.

**Phase 3: Protocols, inference, observability.** A real MCP client in the agent. Escalation on the A2A task lifecycle with a reference expert. Grammar from schemas. llama.cpp backend, warm local server, event stream, terminal UI, raw traces and replay. Published latency, including protocol overhead.

**Phase 4: Gym core and baselines.** Envs as MCP servers. One session for product and evals, so evals measure what ships. Reward fixes checked with golden policies. Stricter scoring. A bake-off under the size cap picks the model, with a compact prompt and a tuned server. The reasoning suite gets built and frozen before any training.

**Phase 5: Reasoning transfer.** Rationale capture, chess teacher and takeover over A2A, SFT with preflight gates, DAgger, RL, then the preregistered test above. Quantization calibrated on gym traces follows, tested on the final build under the cap.

**Phase 6: Release.** Model and data cards, GGUF and MLX builds with checksums, `pip install tth`, published benchmark scores, and `tth-bench` so anyone can score any model on the shell gym.

## Later

Rust core (when measured latency is the blocker), Fish and other shells, Windows/WSL.
