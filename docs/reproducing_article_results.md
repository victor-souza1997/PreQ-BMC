# Reproducing Article Results

Use `preqbmc reproduce` to call the existing article experiment runner and `preqbmc aggregate` to call the existing aggregation and plotting scripts.

## Dry Run

```bash
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1 --dry-run
```

## Run and Aggregate

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

## Existing Results

```bash
preqbmc aggregate \
  --input-root output/article_runs \
  --output-root output/article_results \
  --plots
```

## Claim/Table/Figure Map

| Article claim/table/figure | Command | Expected output file |
| --- | --- | --- |
| Quality metrics | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_quality_metrics.csv` |
| Bit-width allocation | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_bitwidths.csv` |
| Success/failure status | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_success_failure.csv` |
| Scalability status | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_scalability.csv` |
| ESBMC status counts | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_esbmc_status_counts.csv` |
| MRR sweep summary | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_mrr.csv` |
| Implementation gap diagnostics | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results` | `output/article_results/table_implementation_gap.csv` |
| Saturation gap figure | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots` | `output/article_results/plots/fig_saturation_formal_vs_refined.png` |
| Formal vs refined accuracy figure | `preqbmc aggregate --input-root output/article_runs --output-root output/article_results --plots` | `output/article_results/plots/fig_accuracy_formal_vs_refined.png` |
| SMT size and depth figures | Generate SMT formulas during article runs, then aggregate and plot | `output/article_results/plots/fig_smt_size_full_vs_block.png`, `output/article_results/plots/fig_smt_depth_full_vs_block.png` |

The exact runtime depends on ESBMC, Gurobi availability, and the selected experiment subset.
