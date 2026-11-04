<div align="center">
  <img src="assets/tth_logo.png" alt="tth logo" width="420" />

  <h1>tth, the tiny terminal helper</h1>

  <p><strong>Small but good.</strong> A terminal agent built to run on a ~1.5B model in under 1 GB of RAM. It looks before it acts, never runs your command, and knows when to ask for help.</p>

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

## The goal

tth is a terminal agent that runs on a ~1.5B model in under 1 GB of RAM. It inspects your machine, tests the command in a read-only sandbox, and puts it on your prompt without running it. When it can't verify a command, it asks a stronger agent, but only with your consent.

**The research question:** can a large model's verified reasoning make a small model better at terminal tasks that need reasoning, and does chess practice add anything? So far, training has only copied tool calls, so [this hasn't been tested](evals/diagnostics/results/reasoning_in_data.json).

These are targets, and **none of them has been met yet**:

| Area | Done when |
|---|---|
| Size | A ~1.5B model stays under 1,000 MB, counting weights, working memory and the CLI process |
| Speed | Under 3s for easy tasks, under 10s for hard ones |
| Safety | 0 unsafe commands staged on the red-team suite, and tth never runs the final command |
| Reasoning transfer | Reasoning-trained beats actions-only on a held-out reasoning suite, across 3+ seeds, with a confidence interval that excludes zero |
| Chess | Chess practice beats a non-chess control and more shell training. Published either way |
| Public proof | Published scores on NL2SH-ALFA plus our own reasoning suite |

Out of scope: general coding, training on your device, firewalling other agents, and agent swarms.

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
| Qwen3-1.7B, stock | 26 (17–34) | 0 on final code |
| Qwen2.5-Coder-1.5B, stock | 25 (17–34) | 0 |
| Random baseline | 5 (1–10) | 0 |

- The default costs about 2.6 GB of memory and a median of 15 seconds per task, so it **misses the size and speed goals**. A sub-1 GB bake-off will replace it.
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

**The training idea:** a gym with hidden state, verifiable rewards, randomized tool schemas and injected errors. A large model works through tasks and writes down short reasoning, verifiers keep only what leads to correct results, and the small model learns from it. In chess, the student plays, and at a blunder the teacher explains why and gives a better move, with Stockfish scoring both.

**What the protocols are for:** MCP is the single tool contract, so what we train on is what ships. A2A makes the expert swappable: Granite on your Mac, a machine on your network, or a cloud model. They aren't what keeps you safe. That's the sandbox, the classifier and the consent step. See the [roadmap](docs/ROADMAP.md).

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

Apache-2.0, see [LICENSE](LICENSE). The models keep their own licenses, listed in [NOTICE](NOTICE).
