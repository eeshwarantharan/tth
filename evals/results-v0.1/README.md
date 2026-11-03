# v0.1 results (historical)

These are the original results, kept unchanged, except that the folder moved here. Don't treat them as a baseline. The [audit](../../docs/analysis/2026-11-audit.md) explains why:

- They record no commit, adapter hash or config, and the chess rows for three arms came from older adapters than the ones on disk.
- They ran before several prompt and parser changes, on Linux under a prompt that said macOS.
- They used more lenient scoring and a reward that ranked silence too highly.

Current results are in [`evals/results/`](../results/). Every result there has a manifest saying exactly what produced it.
