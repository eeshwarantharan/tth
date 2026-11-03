# Architecture

tth has two packages. **`tth`** is the product people install. **`tth-gym`** is where small models get trained. Dependencies point one way, and CI enforces it with `uv run lint-imports`.

```
evals/  ──►  tth_gym  ──►  tth
```

The product never imports training code or held-out evals. Training never imports held-out evals.

## Inside `tth`

Each layer may only import the layers below it.

```
cli                        the commands
expert | ui | shell        A2A expert server | what the user sees | shell widgets
agent                      the model-to-tools loop, the session, decisions
a2a | inference | mcp      asking for help | running a model | reaching tools
sandbox                    the read-only, offline jail for probes
safety                     parser, classifier, rewriter, secret rules
trace | core               trace format | shared types and config
```

## Where to look

| You want to | Start here |
|---|---|
| Change what counts as safe | `packages/tth/src/tth/safety/classifier.py`, then add a case to `tests/safety/cases/` |
| Change how commands are parsed | `packages/tth/src/tth/safety/parser.py` |
| Add or change a tool | `packages/tth/src/tth/mcp/server.py` and `toolbox.py` |
| Change sandbox rules | `packages/tth/src/tth/sandbox/runner.py` |
| Change the agent loop | `packages/tth/src/tth/agent/loop.py` |
| Add a model backend | `packages/tth/src/tth/inference/` |
| Change what escalation sends | `packages/tth/src/tth/a2a/payload.py` and `redact.py` |
| Add a settings option | `packages/tth/src/tth/core/config.py` |
| Add a gym environment | `packages/tth-gym/src/tth_gym/envs/` |
| Add a shell task family | `packages/tth-gym/src/tth_gym/envs/shell/tasks.py` |
| Change training | `packages/tth-gym/src/tth_gym/train/` |
| Add a held-out suite | `evals/` |
| Change the project site | `site/`, with data from `scripts/build_site_data.py` |

## Settings and files

| What | Where |
|---|---|
| Settings | `~/.config/tth/config.toml`, with `TTH_*` environment variables on top |
| API keys | `~/.config/tth/.env`. A project's own `.env` is never read |
| Model | `~/.local/share/tth/models/`, or `TTH_MODEL` |
| Traces | `~/.local/state/tth/traces.jsonl`, readable only by you |

## Debugging

- `tth setup` checks the sandbox, the model and the backend.
- `tth why` explains the last result. `tth why --raw` also shows exactly what the model wrote.
- `tth replay FILE` draws a recorded episode with no model, using the same code that draws a live run.
- Every run appends to the trace file, in one JSONL format shared by the product, training and evals. Each model turn records its raw text, because the lack of it hid a training bug in v0.1.
- `evals/diagnostics/` holds small, reproducible experiments, such as measuring where training loss sits per token.

## MCP

The agent reaches tools only through `tth/mcp/client.py`, a real MCP client. Tools are discovered with `tools/list` and called with `tools/call`. CI fails if the agent imports tool code directly.

The default transport is in-memory, and stdio is available too. Both speak the same protocol, and every contract test runs over both. In-memory connects about 100 times faster, with about the same per-call cost ([measured](../evals/diagnostics/results/mcp_overhead.json)), which is why it's the default.

`stage`, `escalate` and `ask_user` live in `tth/agent/decisions.py`, in the client. Staging writes to your prompt, so it's never served over MCP.

## What's changing

[Phase 3 of the roadmap](ROADMAP.md) continues with verify-then-commit, a grammar generated from tool schemas, a llama.cpp backend, and escalation on the full A2A task lifecycle. Today A2A carries one text message. The gym still calls `mcp/toolbox.py` directly, until Phase 4 turns its environments into MCP servers.
