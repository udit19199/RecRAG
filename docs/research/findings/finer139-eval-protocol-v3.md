# FiNER-139 evaluation protocol v3

Protocol v3 adds statistically grounded comparisons and reproducible sampling
controls on top of v2 extended metrics (partial F1, macro F1, bootstrap CI,
error taxonomy, concept recall).

## Sampling

| Control | Purpose |
|---------|---------|
| `split` | `train` / `validation` / `test` — prefer **test** for final reports |
| `stratified` | Proportional sample by primary gold XBRL concept |
| `seed` / `seeds[]` | Reproducible draws; multiple seeds → mean ± std aggregate |
| `suite_path` | Frozen inline or HF-id suite (`data/finer139_frozen_suite_v1.json`) |

## New metrics (v3)

- **Paired bootstrap delta** — 95% CI for strict micro-F1 difference (B − A) on the same sentences
- **McNemar sentence hits** — discordant counts on per-sentence strict-hit (any TP)
- **Multi-seed aggregate** — strict F1 mean / std / min / max across seeds

## Artifacts

- CI fixture: `tests/fixtures/finer139_mini_corpus.json`
- Frozen suite: `data/finer139_frozen_suite_v1.json`
- Runner emits `evaluation.protocol_version: 3`, `paired_comparisons`, and optionally `multi_seed_aggregate`

## Usage

```python
from experiments.finer139.runner import RunParams, run_benchmark

# Single seed, test split, stratified
run_benchmark(RunParams(sample_size=100, seed=42, split="test", stratified=True))

# Multi-seed robustness (works with frozen suites; variance reflects run stochasticity only)
run_benchmark(RunParams(seeds=[42, 43, 44], methods=["nlp", "ontology"]))

# Offline CI
run_benchmark(RunParams(suite_path="tests/fixtures/finer139_mini_corpus.json", methods=["ontology"]))
```
