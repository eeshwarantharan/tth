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
> **tth is an early research preview, and not ready for daily use.** The model is often wrong, and the safety layer hasn't been independently reviewed. Always read a staged command before pressing Enter. The [audit](docs/analysis/2026-11-audit.md) has the details.

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
| MCP | Server works, and the agent doesn't use it yet (Phase 3) |
| A2A | Basic escalation works, full task lifecycle planned (Phase 3) |
| Trained model | **None worth shipping.** No trained arm has beaten the untrained base |
| llama.cpp backend | Only through an OpenAI-compatible server you run yourself (Phase 3) |
| `pip install tth` | Not published. Install from source |

## Results so far

**The honest headline: training hasn't improved the model yet, and we now know why the first attempt failed.**

The recorded results showed trained arms at 0/100. They weren't failing at reasoning: training made end-of-turn likely without teaching the model's tool-call tag, so it replied with nothing. Forcing the tag, with no retraining, brought those same weights back:

| Arm | Recorded (100 tasks) | Forced format (20 tasks) |
|---|---|---|
| Untrained base | 9/100 | 5/20; 7/20 unforced |
| Shell training only | 0/100 | 4/20 |
| Chess → shell | 0/100 | 4/20 |
| Table → shell | 0/100 | 4/20 |

These 20-task runs are small and their confidence intervals all overlap, so no arm is shown to beat another. Trained arms do verify more often, but they usually loop until they run out of steps. Red-team runs staged 0 unsafe commands out of 20 for every arm, mostly because the models staged almost nothing.

Sources: [recorded results](evals/results/RESULTS.md) · [diagnostics](evals/diagnostics/) · [full audit](docs/analysis/2026-11-audit.md)

## Quickstart

macOS on Apple Silicon. This uses the **untrained base model**, so expect mistakes.

```bash
git clone https://github.com/eeshwarantharan/tth.git && cd tth
uv sync --extra mlx

# Download the 4-bit model (839 MB on disk)
uv run python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit', local_dir='models/qwen2.5-coder-1.5b-4bit')"

uv run tth setup   # checks the sandbox and model

echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> ~/.zshrc
echo 'eval "$(tth init zsh)"' >> ~/.zshrc && exec zsh
```

Then use `t <request>`, or type a request and press Ctrl-G.

On Linux, install `bubblewrap`, use `tth init bash`, and run a model behind an OpenAI-compatible server (for example `llama-server`) with `--backend openai`. To try escalation, copy [`.env.example`](.env.example) to `.env` and add a key.

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
src/tth/     the product: CLI, agent loop, tools, safety, sandbox, escalation, runtimes, traces
train/       gym environments (chess, shell, table), teacher data, SFT and GRPO
evals/       held-out suites, red-team requests, recorded results, diagnostics
experts/     A2A expert agent
site/        project page, generated from committed results
data/        verified training trajectories
docs/        roadmap, audit, history
tests/       test suite, including the safety red team
```

## Contributing

There's a lot of useful work here at every level. Good places to start:

- **Break the classifier.** Find a command that gets a wrong safety tier and add it to `tests/safety/cases/`.
- **Add gym task families** with execution-based checks.
- **Add shell integrations** such as Fish.

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately, as described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0, see [LICENSE](LICENSE). The base model, Qwen2.5-Coder-1.5B-Instruct, is Apache-2.0 from the Qwen team, see [NOTICE](NOTICE).
