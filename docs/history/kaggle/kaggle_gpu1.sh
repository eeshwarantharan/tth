#!/bin/bash
# Kaggle 2x T4 pipeline (matched, compute-limited settings; see RUNBOOK.md).
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. TTH_SANDBOX=disposable-container CUDA_VISIBLE_DEVICES=1
T="python -m train.sft.train_lora --batch-size 2 --accum 4 --epochs 1"
G="python -m train.grpo.grpo_chess --rounds 1 --tasks-per-round 32 --group 4 --lr 2e-5"
head -300 data/table/scripted_verified.jsonl > data/table/sub.jsonl  # this GPU only (no race)
[ -f evals/results/A_base__redteam.jsonl ] || python -m evals.run --arm A_base --policy hf
$T --data data/table/sub.jsonl --out adapters/D_table_sft
$G --domain table --adapter adapters/D_table_sft/peft --out adapters/D_table_grpo
$T --init-adapter adapters/D_table_grpo/peft --data data/shell/teacher_verified.jsonl --out adapters/D_table_shell
$T --data data/shell/teacher_verified.jsonl --out adapters/B_shell
python -m evals.run --arm D_table_shell --policy hf --adapter adapters/D_table_shell/peft
python -m evals.run --arm B_shell --policy hf --adapter adapters/B_shell/peft
echo '{"job": "gpu1", "note": "ALL DONE"}'
