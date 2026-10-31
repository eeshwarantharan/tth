# Runbook — Kaggle (2× T4): train, evaluate, render the video

```
 Mac (done)                     Kaggle notebook (~1.5–2 h, one screen)                  Mac (5 min)
 teacher data ─► tth_colab.zip ─► GPU0: C (chess→shell), B (shell)   ─► evals ─► table ─► unzip results,
                                  GPU1: A eval, D (table→shell)      ─►          + video   try `t <request>`
```

## 0. Build the bundle (Mac)
```bash
make colab          # writes tth_colab.zip (code + verified data; never .env)
```

## 1. Kaggle setup (once)
1. kaggle.com → **Create → New Dataset** → upload `tth_colab.zip` → name it `tth` → Create.
2. **Create → New Notebook** → File → **Import Notebook** → `train/kaggle_tth.ipynb`.
3. Right panel: **Session options → Accelerator: GPU T4 ×2**, **Internet: On**, **Add Input → your `tth` dataset**.

## 2. Run (interactive session)
Run the cells top to bottom:
- **Setup cells:** should list `src, train, evals, demo, data …`, show two Tesla T4s and print 4 data line counts.
- **Launch cell:** starts both GPU pipelines in the background.
- **Monitor cell:** live bars, refreshed every 10 s:
  ```
  ── runs/gpu0.log  C_chess_sft
     [█████████·····················] 31/93 loss=0.41 tok_s=2100 eta_min=12.5
     · GRPO round 0: mean_reward=0.42 accuracy=0.61 train_rows=96
  ```
  `loss` should fall. GRPO `mean_reward` should rise across rounds. `eval …` bars appear at the end.
  `!!` means a job failed: open the log with `!tail -50 runs/gpu0.log` and send it to Claude.
  If the browser disconnects, just re-run the monitor cell (the jobs keep running).
- **Results cell:** prints the results table and shows the video inline.
- **Download:** the Output panel (right side) → `tth_results.zip`.

Expected time: ~1.5–2 h (not yet measured on Kaggle). Kaggle's free GPU quota is ~30 h/week.

## 3. Back on the Mac
```bash
unzip -o ~/Downloads/tth_results.zip -d .     # adapters/*/mlx, evals/results, demo/tth.gif + tth.mp4
open demo/tth.mp4                              # the video
cat evals/results/RESULTS.md                   # paste these tables into README "Results"
```
Read the shell table (accuracy, 95% CI):
- **C > B and C > D**, CIs separated → chess specifically helps.
- **C ≈ D > B** → verified tool practice helps; chess is not special.
- **C ≈ B** → no transfer. Overlapping CIs = no demonstrated effect. Report it honestly either way.

## 4. Try the product with the best adapter
```bash
export TTH_ADAPTER=adapters/C_chess_shell/mlx      # or whichever arm scored best
echo 'eval "$(uv run --project '"$PWD"' tth init zsh)"' >> ~/.zshrc && exec zsh
cd runs/demo_fixture && t sum the cost for failed tasks in metrics.jsonl
```

## 5. Submit
```bash
make test && make lint
git add -A && git status        # .env, models/, runs/, zips are gitignored. Check that before committing.
git commit -m "tth: safety core, chess/table/shell envs, training pipeline, evals, demo"
```

## Notes to state in the README
- Evals run on Kaggle with the fp16 HF model; the product runs the 4-bit MLX model. Numbers may differ slightly.
- Shell evals on Kaggle run in `disposable-container` mode (no bwrap there); the product always sandboxes.
- Teacher commands were verified on macOS; Kaggle's Linux (GNU) tools can differ for a few flags.
- The video's architecture frames are labelled diagrams; every model-behaviour frame comes from real eval
  traces of the trained student (episode ids in `demo/manifest.json`).
- One seed per arm.
