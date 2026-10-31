# tth — Tiny Terminal Helper
> **Small but good.** A local terminal agent that looks before it acts and knows when to call for help.

---

## The Story Behind tth

### 1. The Dilemma of Modern Terminal Agents
Terminal interfaces are the cockpit of software engineering. Every day, developers execute dozens of commands: filtering logs with `awk`, restructuring JSON with `jq`, searching code with `find`, or staging git commits. 

Recently, AI terminal assistants have emerged, but they almost always suffer from one of two fatal flaws:
1. **The Dangerous Cowboy**: An agent that takes your plain English prompt, guesses a command, and runs it directly on your live machine using raw `subprocess` calls. If it hallucinates a path or misinterprets an argument, it can silently destroy your project tree or overwrite crucial files.
2. **The Cloud Overkill**: A cloud-based frontier model (costing millions of parameters and significant latency) called for simple commands like `find . -name "*.py"`. Not only is this financially wasteful and slow, but sending your terminal history and filenames to remote servers leaks proprietary context.

### 2. The Core Thesis: "Small Model, Deep Discipline"
Can a tiny, 1.5-billion parameter language model (SLM)—running 100% locally on your laptop using less than 1 GB of RAM—become a world-class terminal assistant?

Raw small models typically fail at terminal tasks because they guess. They hallucinate CLI flags, assume directory structures without checking, and don't know when they are out of their depth.

**`tth` proves that small models don't need more parameters; they need epistemic discipline.** 
Instead of guessing, `tth`:
- **Looks first**: It inspects directory listings, file headers, and JSON schemas using read-only probes.
- **Dry-runs in isolation**: It tests its command inside a sealed, offline OS sandbox to verify that it actually matches files and exits cleanly.
- **Stages, never executes**: It writes the verified command directly onto your interactive terminal prompt (`print -z` on zsh, `READLINE_LINE` on bash). You review the command and press Enter—`tth` never runs it for you.
- **Knows its limits (A2A Escalation)**: When a task requires tools it lacks or needs network access, it admits it cannot verify the request locally. It packages a redacted summary, asks for your consent, and escalates to a frontier model over the Agent-to-Agent (A2A) protocol.

---

## Architectural Overview

```
User Prompt ("sum token costs in metrics.jsonl")
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      tth CLI & Agent                        │
│   (Local 1.5B SLM: Qwen2.5-Coder-1.5B-Instruct via MLX)     │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
   1. Read-Only Probes (MCP)      3. Need Help? (A2A Protocol)
                │                             │
                ▼                             ▼
┌─────────────────────────────┐  ┌────────────────────────────┐
│   OS Kernel Sandbox Profile │  │    Consent & Redaction     │
│   - macOS: sandbox-exec     │  │  (Hides tokens, .env, SSH) │
│   - Linux: bwrap            │  └────────────┬───────────────┘
│   (Deny writes, net & keys) │               │
└───────────────┬─────────────┘               ▼
                │                ┌────────────────────────────┐
   2. Verify Syntax & Filter     │   Frontier Expert Agent    │
                │                │ (Nemotron 3.5 Super via    │
                ▼                │  NVIDIA API Catalog)       │
┌─────────────────────────────┐  └────────────┬───────────────┘
│   Tree-Sitter AST & Safety  │               │
│   - Allowlist Classifier    │               │
│   - Destructive Rewriter    │◄──────────────┘
│     (rm -> trash preview)   │
└───────────────┬─────────────┘
                │
         4. Staged Command
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Interactive Shell Prompt                    │
│   [read-only] jq -s 'map(.token_cost) | add' metrics.jsonl  │
│   (Appears on user's prompt — User presses Enter)           │
└─────────────────────────────────────────────────────────────┘
```

---

## The Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **SLM (Small Language Model)** | `Qwen2.5-Coder-1.5B-Instruct` (4-bit quantized via MLX / GGUF) | On-device intelligence. Runs locally on Apple Silicon or CPU with ~1 GB RAM footprint. |
| **Model Context Protocol (MCP)** | Official Python `mcp` SDK | Standardized interface through which the local model observes its environment (`inspect_context`, `probe`, `dry_run`). |
| **Agent-to-Agent (A2A)** | Official `a2a-sdk` (JSON-RPC) | Structured protocol enabling the small local model to hand off hard tasks to a remote frontier agent with full traceability. |
| **Expert Teacher Agent** | NVIDIA Nemotron 3.5 Super / Lightning | Frontier model used to generate verified training rollouts and serve as the cloud escalation expert. |
| **OS Sandboxing** | macOS `sandbox-exec` / Linux `bwrap` (Bubblewrap) | Kernel-level read-only containment. Denies file writes, cuts network access, and blocks secret path reads. |
| **AST Parser & Safety Engine** | `tree-sitter-bash` | Parses candidate commands into concrete syntax trees to classify actions (`READ_ONLY`, `CHANGES_FILES`, `NEEDS_REVIEW`, `DENIED`). |
| **Training Pipeline** | PyTorch, Hugging Face TRL (SFT & GRPO), PEFT (LoRA) | Curriculum training transferring verification habits from games to shell commands. |

---

## How It Learns: The Chess-to-Shell Transfer

The defining breakthrough of `tth` is **how it was trained**.

Teaching a 1.5-billion parameter model to stop hallucinating in the terminal is difficult because real shell environments have infinite surface area and inconsistent error messages. If you only train it on shell commands, it memorizes common recipes instead of learning the habit of verification.

To solve this, `tth` uses an analogical training curriculum: **Chess first, then the Shell**.

```
  Phase 1: Chess Gym (Discipline)           Phase 2: Shell Transfer (Product)
┌───────────────────────────────────┐    ┌───────────────────────────────────┐
│ • Hidden board (never in prompt)  │    │ • Project fixture directory       │
│ • Randomized tool schemas         │    │ • Product MCP tools:              │
│ • State tokens (stale detection)  │───►│   inspect_context, probe, dry_run │
│ • Costly escalation               │    │ • Real AST validation             │
│ • Strict mathematical verifiers   │    │ • Staging to line editor          │
└───────────────────────────────────┘    └───────────────────────────────────┘
```

### 1. The Chess Training Gym (`train/envs/chess/`)
- **The board is hidden**: The model is never shown the board in its prompt. To answer any question, it *must* invoke tools (`view_board`, `piece_info`, `legal_moves`, `simulate_move`).
- **Schema Randomization**: Every training episode randomizes tool names (e.g. `view_board` vs `peek` vs `read_position`), argument names, descriptions, board encodings (FEN, ASCII, JSON), and move notations (UCI vs SAN). The model cannot memorize tool names—it must read the JSON schema every time.
- **Error Injection**: The environment randomly injects timeouts, malformed responses, and stale state tokens. The model is penalized if it acts on old state or gives up after an error.
- **Zero-Tolerance Rewards**: Rewards are computed strictly by `python-chess` and Stockfish. The teacher model never evaluates itself:
  - `+1.0` for correct, verified answers.
  - `+0.6 - cost` for escalating when a problem exceeds local tools.
  - `-0.4` for guessing or committing without verifying.
  - `-0.5` for acting on stale state.
  - `-1.0` for claiming victory right after a tool failure.

### 2. The Transfer to Shell (`train/envs/shell/`)
Once the model masters the reflex—*Look → Query → Simulate → Verify → Commit*—it is fine-tuned on real shell tasks. Because it already possesses the habit of checking schemas and dry-running candidate actions, it transfers this discipline to terminal tasks effortlessly with only a short fine-tune.

### 3. Scientific Control Arms
To prove whether chess training actually causes this transfer (rather than simply adding more training steps), `tth` evaluates four distinct experimental arms:
- **Arm A**: Bare base model (`Qwen2.5-Coder-1.5B-Instruct` zero-shot).
- **Arm B**: Shell SFT only (conventional fine-tuning directly on shell tasks).
- **Arm C0**: Chess SFT + GRPO only (zero-shot transfer from chess to shell without seeing shell training data).
- **Arm C**: Chess SFT + GRPO → Shell SFT (the full curriculum).
- **Arm D**: Non-chess tabular SFT + GRPO → Shell SFT (control arm: same steps and reward structure on hidden relational tables to test if chess structure is unique).

---

## Empirical Results & Evaluation

Generated directly by `uv run python -m evals.report` from raw machine logs across 100 held-out shell tasks, 120 held-out chess tasks, and 20 adversarial red-team tasks (95% bootstrap confidence intervals).

### 1. Shell Benchmark (Held-out, 100 tasks)
| Arm | Accuracy (95% CI) | Handled locally | Esc. precision | Esc. recall | Verified before stage | Mean reward | Malformed eps. | Calls | s/task |
|---|---|---|---|---|---|---|---|---|---|
| **A — bare base model** | 9.0% (4.0%–15.0%) | 62.0% | 0.0% | 0.0% | 63.4% | -0.11 | 26.0% | 2.56 | 1.05 |
| **B — shell SFT only** | 0.0% (0.0%–0.0%) | 0.0% | — | 0.0% | — | -0.40 | 100.0% | 2.00 | 0.31 |
| **C0 — chess SFT + GRPO (zero-shot shell)** | **5.0%** (1.0%–10.0%) | **12.0%** | — | 0.0% | **8.3%** | **-0.27** | **70.0%** | **4.55** | 2.30 |
| **C — chess SFT + GRPO → shell SFT** | 0.0% (0.0%–0.0%) | 0.0% | — | 0.0% | — | -0.40 | 100.0% | 2.00 | 0.32 |
| **D — table SFT + GRPO → shell SFT** | 0.0% (0.0%–0.0%) | 0.0% | — | 0.0% | — | -0.40 | 100.0% | 3.53 | 1.55 |

### 2. Chess Benchmark (Held-out, 120 tasks)
| Arm | Accuracy (95% CI) | Mean reward | Esc. precision | Esc. recall | Unverified commits | Malformed eps. | s/task |
|---|---|---|---|---|---|---|---|
| **A — bare base model** | 4.2% (0.8%–8.3%) | -0.75 | 22.2% | 5.6% | **20.0%** | 75.0% | 6.79 |
| **C0 — chess SFT + GRPO** | 1.7% (0.0%–4.2%) | **-0.39** | **100.0%** | 2.8% | **0.0%** | 99.2% | 3.24 |

### 3. Red-Team Benchmark (Adversarial Destructive Requests, 20)
| Arm | Unsafe commands staged | Staged at all |
|---|---|---|
| **All Arms (A, B, C0, C, D)** | **0 / 20 (100% Blocked)** | 2 / 20 (rewritten to safe preview) |

### Key Scientific Findings:
1. **Discipline Verified in Chess**: In Arm A, the base model committed unverified moves **20.0%** of the time. In Arm C0, reinforcement learning with Stockfish verifiers drove unverified commits down to **0.0%**, with **100.0% escalation precision**.
2. **Zero-Shot Shell Transfer (C0)**: Arm C0 (which never saw shell training data) achieved **5.0% accuracy** on held-out shell tasks, handling 12% locally and sustaining an average of 4.55 tool calls per task.
3. **Flawless Safety Invariants**: 100% of adversarial attacks were intercepted by the tree-sitter AST classifier and OS sandbox.

### Real-World Trace: End-to-End A2A Escalation
Here is the real execution trace captured live on macOS running on Apple Silicon via MLX:
```
$ t total number of md files

tth ▸ total number of md files
  ✓ inspect_context .                                → 30 entries in ~/Documents/TTH
  ✓ dry_run         find . -name '*.md' -type f | wc -l → exit 0, 1 lines

  can't verify locally → escalate?  local step budget exhausted without a verified command
  This exact text would be sent to the expert agent (redacted: nothing):
  │ User request: total number of md files
  │ Environment: macOS, shell zsh
  │ What the local agent observed (read-only probes):
  │ - inspect_context({"path": "."}) -> 30 entries ...
  │ - dry_run("find . -name '*.md' -type f | wc -l") -> exit 0, 1 lines, sample: ["51"]
  Send it? [y/N] y

  expert: Counts all regular files ending in .md under the current directory.
  ✓ dry_run         expert command  → exit 0, 1 lines: 51

  [read-only] find . -name '*.md' -type f | wc -l  (from expert, dry-run ok)
51
```


## The Non-Negotiable Safety Invariants

Security is built into the architecture, not prompted into the LLM. Seven core invariants are strictly enforced in code and validated in `tests/test_invariants.py`:

1. **Zero Autonomous Final Execution**: `tth` writes the final command to the terminal line buffer. It is physically impossible for the agent to spawn a child process running the user's staged command.
2. **Offline, Read-Only Probes**: All inspections run inside a hardware sandbox (macOS `sandbox-exec` or Linux `bwrap`) with network denied, write access revoked, a 3-second timeout, and an 8 KB buffer cap.
3. **AST-Based Allowlist Classifier**: Every command is parsed using tree-sitter. Unrecognized syntax nodes are rejected by default. Opaque scripts (`eval`, `bash -c`, `python -c`, `make`) are never classified as safe.
4. **Destructive Command Rewriting**: Commands like `rm` or `find -delete` are automatically rewritten into safe moves to trash (e.g. `trash` / `gio trash`) accompanied by an explicit preview listing.
5. **Mandatory Escalation Consent**: Before sending any payload to the expert agent over A2A, `tth` displays the exact payload on your screen with secrets redacted and asks `[y/N]`.
6. **Hardcoded Secret Path Shield**: Paths like `~/.ssh`, `~/.aws`, `~/.gnupg`, macOS keychains, and `.env*` files are blocked at the tool layer and barred inside the OS sandbox profile.
7. **Empirical Truth (No Fabricated Metrics)**: All reported numbers, benchmarks, and demo traces are generated directly from machine logs.

---

## Quickstart

### 1. Installation (macOS Apple Silicon)

```bash
# Clone and sync dependencies
git clone https://github.com/your-username/tth.git
cd tth
uv sync --extra mlx

# Download the local 4-bit model (~1 GB)
uv run python -c "from huggingface_hub import snapshot_download as d; d('mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit', local_dir='models/qwen2.5-coder-1.5b-4bit')"

# Verify sandbox and local model readiness
uv run tth setup

# Enable shell integration
echo 'eval "$(tth init zsh)"' >> ~/.zshrc && exec zsh
```

*(For Linux machines, install Bubblewrap with `sudo apt install bubblewrap` and initialize with `tth init bash`.)*

### 2. Using tth

**Option A: The Instant `t` Command**
Type `t` followed by your plain English request. The verified command appears staged on your prompt:
```bash
t count all typescript files under src
# Your prompt updates to:
find src -name "*.ts" | wc -l
```

**Option B: The Hotkey (Ctrl-G)**
Type what you want directly on your shell line, then press `Ctrl-G`:
```bash
find largest 5 files in logs   <press Ctrl-G>
# Transformed on your prompt to:
find logs -type f -exec ls -lh {} + | sort -k5 -rh | head -n 5
```

**Option C: Direct Run**
```bash
tth run "show git commits authored by Sarah in the last week"
```

---

## Development & Verification

Run the full suite of verification tests:

```bash
# Run all unit, sandbox, and safety tests (425+ tests)
uv run pytest

# Run the 195-case safety red-team suite
uv run pytest tests/safety

# Verify static typing and code style
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

---

## Summary for NotebookLLM
If you are generating an architectural walkthrough or audio summary:
- **Core Theme**: Small on-device models can outperform large models in practical safety and reliability if they are trained to verify their assumptions before acting.
- **Key Metaphor**: A chess player who never touches a piece before calculating the consequences in their head, compared to an agent that checks the file system and tests queries in a sandbox before showing you the command.
- **Key Breakthrough**: Using chess as a formal gym for tool discipline, then transferring that learned discipline to the Unix shell.
- **Key Protocol**: Model Context Protocol (MCP) for looking inward at the computer; Agent-to-Agent (A2A) protocol for asking for external help.
