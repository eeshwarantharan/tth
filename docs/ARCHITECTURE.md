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
| Change the consent flow | `packages/tth/src/tth/agent/escalation.py` |
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

## A2A

The agent reaches experts only through `tth/agent/escalation.py`, which is where consent lives. Tests fail if anything else sends to an expert.

1. **Discover.** The client fetches the expert's Agent Card. This sends no user data.
2. **Ask.** The user sees the exact redacted text and its real destination. tth's reference expert declares where it forwards to. Only that expert, running locally with no forwarding target, can say the text stays on this machine. The default answer is no, and no flag or variable skips this.
3. **Follow the task.** The expert can return `input-required` to ask for one more read-only result. That runs through the MCP server, which refuses secret paths, and needs fresh consent. Follow-ups are capped.
4. **Check the answer.** Typed answers arrive as an artifact, and plain text from other agents is parsed as a fallback. Either way the command goes through the classifier and a local test.

`tth expert` runs the reference expert with any OpenAI-compatible model. Set `expert_base_url` and `expert_model` in the config. Listening beyond this machine requires `TTH_EXPERT_TOKEN`.

## The gate

`tth/agent/product.py` sits between the model and your prompt, and holds the rules that make "look before you act" structural. Staging a read-only command tests it first. Changing files requires having looked. Repeating a passing test commits it, and repeating a failure stops the loop. Decisions made by a rule are labelled `gate` in the trace.

## Running a model

`tth/inference/` has three backends, chosen by the `backend` setting:

| Backend | Runs | Grammar |
|---|---|---|
| `llamacpp` | A GGUF through a warm local `llama-server`, which stays loaded between requests and only listens on this machine | Available: the MCP tool schemas become the decoding grammar |
| `mlx` | MLX weights in the same process | No |
| `openai` | Any OpenAI-compatible server you run | Depends on the server |

## What's changing

Phase 4 of the [roadmap](ROADMAP.md) turns gym environments into MCP servers, fixes the rewards and runs the model bake-off. Until then the gym calls `mcp/toolbox.py` directly.
