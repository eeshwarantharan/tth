<div align="center">
  <img src="assets/tth_logo.png" alt="tth logo" width="420" />

  <h1>tth, the tiny terminal helper</h1>

  <p><strong>Small but good.</strong> A local small-model agent that looks before it acts, never runs your command, and knows when to ask for help.</p>

  <p><a href="https://eeshwarantharan.github.io/tth/">Project site</a> · <a href="docs/ROADMAP.md">Roadmap</a> · <a href="docs/analysis/2026-11-audit.md">Audit</a></p>

  <p>
    <a href="https://github.com/eeshwarantharan/tth/actions/workflows/ci.yml"><img src="https://github.com/eeshwarantharan/tth/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
    <img src="https://img.shields.io/badge/status-early%20research%20preview-orange.svg" alt="Status: early research preview">
  </p>
</div>

> [!WARNING]
> **tth is an early research preview, and not ready for daily use.** The model is wrong about four times in ten on our test tasks, and the safety layer hasn't been independently reviewed. Always read a staged command before pressing Enter. The [audit](docs/analysis/2026-11-audit.md) has the details.

## What it does

You describe what you want. tth inspects your real files in a read-only sandbox, tests a command, and puts it on your prompt. **You** decide whether to run it.

```bash
t count all python files under src
# tth probes and dry-runs in a sandbox, then your prompt becomes:
find src -name '*.py' | wc -l
```

## Why it's different

- **It never runs the final command.** The command is written to your shell's input buffer. No exceptions.
- **Probes are read-only and offline,** inside `sandbox-exec` on macOS or `bwrap` on Linux.
- **Safety is deterministic.** An AST-based allowlist classifies commands and doesn't rely on the model.
- **Escalation needs consent.** You see exactly what would be sent, secrets redacted, and the default is no.
- **Training happens in a gym with verifiable rewards.** Software verifiers assign rewards, never an LLM.

## Status

| Area | State |
|---|---|
| Shell staging for zsh, plus Ctrl-G on bash | Works |
| Sandbox on macOS | Works and tested |
| Sandbox on Linux | Implemented, not yet tested in CI against real `bwrap` |
| Command classifier and rewriter | Hardened against every [known bypass](docs/analysis/2026-11-audit.md#4-safety-defects), with 451 red-team cases and adversarial fuzzing. Not independently reviewed, so more gaps are likely |
| Secret redaction | Covers JSON, env values, tokens and keys. Pattern-based, so it can miss things |
| Chess, table and shell gym environments | Work |
| MCP | The agent reaches tools only through a real MCP client, and other agents can use `tth mcp`. The gym moves over in Phase 4 |
| A2A | The only path to an expert: Agent Cards, task states, evidence requests with fresh consent, typed answers. Any A2A agent can be an expert |
| Trained model | **None worth shipping.** The default is a stock model. No trained model has beaten one yet |
| llama.cpp backend | The default: a warm local server that stays loaded between requests |
| `pip install tth` | Wheel builds and works from a clean install on macOS. Not published to PyPI yet |

## Results so far

**The headline: a stock model scores 58 of 100 on held-out shell tasks with no training, and nothing we've trained has beaten a stock model yet.**

| Setup, on 100 held-out shell tasks | Correct (95% CI) | Unsafe, of 20 |
|---|---|---|
| **Granite 4.1-3B + compiler tools (default)** | **58** (48–67) | 0 |
| Granite 4.1-3B, stock | 43 (33–52) | 0 |
| Qwen3-1.7B, stock | 26 (2–11) | 0 on final code |
| Qwen2.5-Coder-1.5B, stock | 25 (1–10) | 0 |
| Random baseline | 5 (1–10) | 0 |

- It costs about 2.6 GB of memory and 15 seconds per task, more than we'd like on both.
- These tasks are synthetic, and their families overlap with training templates, so expect lower numbers on real work.
- The first training run scored 0/100 because training never taught the model its tool-call tag. Fine-tuning hasn't had a fair test yet: it had 90 examples and 12 steps.

Sources: [current results](evals/results/RESULTS.md) · [full audit](docs/analysis/2026-11-audit.md) · [v0.1 results](evals/results-v0.1/RESULTS.md)

## Quickstart

On macOS or Linux, with `uv`. It uses a stock model that's right a bit over half the time on our test tasks, so read every command before you run it.

```bash
git clone https://github.com/eeshwarantharan/tth.git && cd tth
uv sync
brew install llama.cpp   # or build llama-server another way, and put it on your PATH

# Download the model (2.1 GB on disk, about 2.6 GB of memory)
uv run hf download ibm-granite/granite-4.1-3b-GGUF granite-4.1-3b-Q4_K_M.gguf --local-dir ~/.local/share/tth/models

uv run tth setup   # checks the sandbox, model and backend

echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> ~/.zshrc
echo 'eval "$(tth init zsh)"' >> ~/.zshrc && exec zsh
```

Then use `t <request>`, or type a request and press Ctrl-G.

On Linux, also install `bubblewrap`, and use `tth init bash`. For less memory, set `backend = "mlx"` in `~/.config/tth/config.toml` and use the smaller Qwen2.5-Coder model, which scores about 25.

To try escalation, copy [`.env.example`](.env.example) to `~/.config/tth/.env` and add a key, or point `expert_base_url` at any OpenAI-compatible model. tth never reads a project's own `.env`.

## How it works

```
 your request
      │
      ▼
 ┌──────────┐  tools   ┌────────────────────────────┐
 │  small   │◄────────►│ inspect · probe · dry_run  │  read-only, offline sandbox
 │  model   │          └────────────────────────────┘
 │  (local) │   help   ┌────────────────────────────┐
 │          │◄────────►│ expert agent, with consent │  secrets redacted
 └────┬─────┘          └────────────────────────────┘
      ▼
 classifier + rewriter ──► command on your prompt, with a tier badge
                           (you press Enter, or you don't)
```

**The training idea:** a gym with hidden state, verifiable rewards, randomized tool schemas and injected errors. A teacher generates demonstrations, verifiers keep only the correct ones, and the student learns from them. It then practices in the gym, with the teacher stepping in when it fails.

**Where this is heading:** MCP becomes the only way the model reaches tools, and A2A the only way it reaches help, in both the product and the gym. See the [roadmap](docs/ROADMAP.md).

## Repository layout

```
packages/tth/       the product: CLI, agent, MCP tools, A2A escalation, safety, sandbox, inference, traces
packages/tth-gym/   the gym: chess, shell and table environments, teacher data, training
evals/              held-out suites, red-team requests, recorded results, diagnostics
data/               verified training trajectories
site/               project page, generated from committed results
docs/               architecture, roadmap, audit, history
tests/              test suite, including the safety red team
```

Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It has a table of where to look for each kind of change.

## Contributing

There's a lot of useful work here at every level. Good places to start:

- **Break the classifier.** Find a command that gets a wrong safety tier and add it to `tests/safety/cases/`.
- **Add gym task families** with execution-based checks.
- **Add shell integrations** such as Fish.

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately, as described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0, see [LICENSE](LICENSE). The base model, Qwen2.5-Coder-1.5B-Instruct, is Apache-2.0 from the Qwen team, see [NOTICE](NOTICE).
