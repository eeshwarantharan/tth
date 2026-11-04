# Data card — tth teacher trajectories

## What
Multi-turn tool-use trajectories (system prompt, tool schemas, user request, assistant tool calls,
tool results) in two domains:

| File | Domain | Content |
|---|---|---|
| `data/chess/tasks_train.jsonl` | chess | generated tasks (FEN, question, ground truth) |
| `data/chess/teacher_raw.jsonl` | chess | every teacher episode, with reward breakdown |
| `data/chess/teacher_verified.jsonl` | chess | training subset (see filter) |
| `data/shell/tasks_train.jsonl` | shell | generated tasks (fixture seed, request, check) |
| `data/shell/teacher_raw.jsonl` / `teacher_verified.jsonl` | shell | same as above |

Counts, per-kind/per-family breakdowns and per-teacher counts are in the generated
`data/*/teacher_verified.stats.json` files.

## How it was generated
- **Teachers** (NVIDIA API catalog, thinking enabled, temperature 0.6, one tool call per turn):
  - both domains: `nvidia/nemotron-3.5-super-vl-120b-a12b` for the first episodes, then
    `nvidia/nemotron-3.5-lightning-30b-a3b` after the shared per-key rate limit (~40 requests/min across
    models) kept Super throttled (HTTP 429) for most of the time budget. Every row records its `teacher`;
    per-teacher counts are in the stats files.
  - Only the final tool calls and visible content are stored; the teachers' reasoning text is not.
  - **chess, scripted oracle** (`data/chess/scripted_*.jsonl`, teacher `scripted-oracle (python-chess +
    Stockfish)`): `packages/tth-gym/src/tth_gym/data/scripted_chess.py` acts only through each episode's randomized tools and shows
    the evidence for every answer (view → query/simulate → submit), recovers from injected errors (retry
    after timeout/malformed response, re-view after stale state), escalates to the Stockfish expert on hard
    tasks and simulates the expert's move before committing. Added because the LLM teachers almost never
    escalated, so their verified chess data had no escalation examples, and the API rate limit capped how
    much data they could produce. This departs from CLAUDE.md's "teacher model generates trajectories";
    approved by the project owner. Scripted transcripts are more templated than LLM ones.
- **Environments**: the same code the student is trained and evaluated in (`train/envs/chess`,
  `train/envs/shell`). Chess episodes have randomized tool schemas, board formats and injected errors.
  Shell episodes run the product's read-only, sandboxed toolbox on deterministic fixture directories.

## How it was verified
Rewards come only from deterministic verifiers (python-chess, Stockfish, and sandboxed execution of the
staged shell command compared to ground truth computed in Python). The teacher never grades itself.
`packages/tth-gym/src/tth_gym/data/filter.py` keeps an episode only if it is **correct and has zero penalties** (no malformed calls,
no unverified commit, no acting on stale state, no claiming success after an error).

## Splits and contamination
Held-out eval tasks live in `evals/` and use different seeds (chess positions are de-duplicated against the
training FENs; shell fixtures use disjoint seeds, different request wording and two eval-only task
families). `tth_gym` never imports `evals/`.

## Known limitations
- Synthetic fixtures: real repositories are messier than the generated projects.
- Shell commands were verified on macOS (BSD userland); some may behave differently with GNU tools.
- Two teachers with different strengths were mixed in the shell set.
- The project owner states that NVIDIA allows training on these outputs. That hasn't been independently checked against NVIDIA's API terms, so confirm it before redistributing the trajectories.
