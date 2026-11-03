# Contributing to tth

Thanks for helping. tth is an early research preview, so there's plenty of room to have a real impact.

## Setup

```bash
git clone https://github.com/eeshwarantharan/tth.git && cd tth
uv sync --extra mlx     # on Linux: uv sync
uv run pytest
```

Linux also needs `bubblewrap` for the sandbox.

## Before you open a PR

```bash
uv run ruff check . && uv run ruff format .
uv run pyright
uv run pytest
uv run pytest tests/safety    # must always pass
```

## Ground rules

These protect users, so they aren't negotiable. If your change seems to need an exception, open an issue first.

1. **tth never executes the final command.** It's written to the shell's input buffer, nothing more.
2. **Probes are read-only and offline,** always inside the sandbox, with a timeout and an output cap.
3. **Safety is allowlist-based.** Anything the classifier doesn't recognize is denied. Opaque constructs are never shown as safe.
4. **Destructive commands are never staged raw.** They become a preview or a move to trash, and rewrites are always shown.
5. **Escalation requires consent.** Show the exact payload, redact secrets, default to no.
6. **Secret paths are never probed.**
7. **No fabricated results.** If an eval wasn't run, say so. Every published number must trace to a committed file.

## Good first contributions

- **Break the classifier.** Find a command that gets a wrong tier, then add it to `tests/safety/cases/*.yaml` along with the fix. Every safety change must add a red-team case, including the failure that motivated it.
- **Add a gym task family** in `train/envs/shell/tasks.py`, with a check based on real execution.
- **Add a shell integration** for Fish or another shell.

## How we work

- Small, focused diffs, one concern per PR.
- Prefer verifiers and deterministic checks over model judgment.
- Never tune on eval tasks. `train/` must never import held-out data from `evals/`.
- New tools must declare whether they're read-only and sandboxed, plus their timeout and output cap.

## Conduct

Be respectful and assume good faith. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
