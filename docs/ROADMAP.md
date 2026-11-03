# tth roadmap

**Goal:** a local small-model agent for the terminal, trained in an RL gym with teacher-student methods. MCP is how it sees. A2A is how it asks for help. The reasoning behind this plan is in the [November 2026 audit](analysis/2026-11-audit.md).

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
| 2 | Restructure into `tth` and `tth-gym` | Next |
| 3 | MCP and A2A at the core, grammar, llama.cpp, terminal UI | Planned |
| 4 | Gym core, honest baselines, model bake-off | Planned |
| 5 | Teacher-student lab, fine-tuning, quantization | Planned |
| 6 | Release | Planned |

## The improvement ladder

Each change is one rung, scored against the rung below it on the same held-out tasks, with paired bootstrap CIs and at least 3 seeds where training is involved. It ships only if it wins.

| Rung | Change |
|---|---|
| L0 | Stock model (one per bake-off candidate) |
| L1 | + grammar generated from MCP tool schemas |
| L2 | + verify, then commit, by construction |
| L3 | + compact prompt and structured observations |
| L4 | + compiler tools (the model fills slots, tools write the command) |
| L5 | + retrieved lessons (few-shot) |
| L6 | + SFT from oracle and teacher demos |
| L7 | + DAgger rounds |
| L8 | + RL (GRPO) |
| Q | Quantization matrix on the winner |

**Stop rule.** If the best student can't beat the stock model with grammar after at least 5k verified rows, training investment stops. tth then ships the best stock model with the safety layer, and the gym stays as an open benchmark.

## How the protocols help the model

**MCP**
- Tool schemas compile into a decoding grammar, so calls are valid by construction.
- `stage` becomes available only after a verifying step, and the choices then narrow toward it. This targets the looping seen in the audit.
- Compiler tools such as `csv_aggregate(file, op, column, where)` replace error-prone `awk`.
- Structured observations and task-shaped tool lists shrink the prompt.
- Domain packs add capability without touching core, and each pack brings its own gym tasks.
- Gym envs are MCP servers, so training and product share one contract.

**A2A**
- A consented expert answer that passes local `dry_run` can become a local lesson (opt-in), so the same request needn't escalate twice.
- The same loop powers training: at a student's first failure, a teacher takes over from that exact state (DAgger).
- Agent Cards route requests to a bigger local model before any hosted one, always with consent.
- Tasks carry what the student tried, and return typed artifacts that can be verified locally.

## Phases in detail

**Phase 1: Safety.** Add every known bypass as a red-team case first, then fix it. Walk heredocs and expansions, deny dynamic arguments, make allowlists flag-aware, refuse wrapped destructive forms, keep one source of truth for secrets, and redact JSON-aware before truncating.

**Phase 2: Restructure.** Two packages with one-way dependencies enforced in CI, a typed config file, dependency extras and working wheel installs.

**Phase 3: Protocols, inference, observability.** A real MCP client in the agent. Escalation on the A2A task lifecycle with a reference expert. Grammar from schemas. llama.cpp backend, warm local server, event stream, terminal UI, raw traces and replay. Published latency, including protocol overhead.

**Phase 4: Gym core and baselines.** Envs as MCP servers. One session for product and evals, so evals measure what ships. Reward fixes checked with golden policies. Stricter scoring and a real-world held-out suite. Baselines and model bake-off.

**Phase 5: Teacher-student lab.** Oracle and LLM teachers as A2A agents, SFT with preflight gates, DAgger, RL, then a quantization matrix including importance matrices calibrated on gym traces. A preregistered, multi-seed test of whether chess helps.

**Phase 6: Release.** Model and data cards, GGUF and MLX builds with checksums, `pip install tth`, and `tth-bench` so anyone can score any model on the shell gym.

## Later

Rust core (when measured latency is the blocker), Fish and other shells, Windows/WSL, on-device fine-tuning from local lessons.
