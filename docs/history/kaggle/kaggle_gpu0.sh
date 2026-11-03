#!/bin/bash
# Kaggle 2x T4 pipeline (matched, compute-limited settings; see RUNBOOK.md).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. TTH_SANDBOX=disposable-container CUDA_VISIBLE_DEVICES=0
T="python -m train.sft.train_lora --batch-size 2 --accum 4 --epochs 1"
G="python -m train.grpo.grpo_chess --rounds 1 --tasks-per-round 32 --group 4 --lr 2e-5"
cat data/chess/teacher_verified.jsonl > data/chess/sub.jsonl && head -213 data/chess/scripted_verified.jsonl >> data/chess/sub.jsonl  # this GPU only (no race)
$T --data data/chess/sub.jsonl --out adapters/C_chess_sft
$G --domain chess --adapter adapters/C_chess_sft/peft --out adapters/C0_chess_grpo
$T --init-adapter adapters/C0_chess_grpo/peft --data data/shell/teacher_verified.jsonl --out adapters/C_chess_shell
python -m evals.run --arm C0_chess --policy hf --adapter adapters/C0_chess_grpo/peft
python -m evals.run --arm C_chess_shell --policy hf --adapter adapters/C_chess_shell/peft
echo '{"job": "gpu0", "note": "ALL DONE"}'
