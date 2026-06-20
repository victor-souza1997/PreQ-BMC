# Article Experiments

This experiment layer is for article-ready runs about deployment-aware verification of fixed-point quantized neural networks with structured C harnesses, ESBMC, layer-wise preimage contracts, block-wise hidden-layer verification, and saturation-aware refinement.

It does not replace the existing paper scripts. The runner calls the existing robustness pipeline and writes article-specific run metadata, aggregate tables, LaTeX files, and plots under `output/article_runs` and `output/article_results`.

## Run Everything

```bash
python tool/scripts/run_article_experiments.py \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

Large disabled runs, including the `mnist_2blk_100_100` scalability frontier, are skipped unless `--include-disabled` is supplied.

## Run One Benchmark

```bash
python tool/scripts/run_article_experiments.py \
  --only iris \
  --max-runs 1 \
  --continue-on-error \
  --aggregate \
  --plots
```

MNIST 50_50:

```bash
python tool/scripts/run_article_experiments.py \
  --only mnist_2blk_50_50 \
  --continue-on-error \
  --aggregate \
  --plots
```

## Resume After a Crash

```bash
python tool/scripts/run_article_experiments.py --resume --continue-on-error --aggregate --plots
```

`--resume` skips run directories whose `run_status.json` already has `"status": "success"`. Use `--force` to rerun them.

## Aggregate Existing Outputs

```bash
python tool/scripts/aggregate_article_results.py \
  --input-root output/article_runs \
  --output-root output/article_results
```

The aggregator is tolerant of partial and failed runs. It still creates the required CSV tables, LaTeX tables, plots input files, `failed_runs.csv`, and `article_summary.md`.

## Generate Plots

```bash
python tool/scripts/plot_article_results.py \
  --input-root output/article_results \
  --output-root output/article_results/plots
```

Plots use matplotlib only. If a table has no numeric data yet, the script writes a placeholder PNG so the result directory remains complete.

## SMT Syntactic Complexity

To request SMT formula export during runs:

```bash
python tool/scripts/run_article_experiments.py \
  --only iris \
  --esbmc-generate-smt-formula
```

Then analyze formulas:

```bash
python tool/scripts/analyze_smt_complexity.py \
  --input-root output/article_runs \
  --output output/article_results/smt_complexity.csv
```

The reported SMT metrics are syntactic complexity indicators, including file size, operator counts, and approximate SMT nesting depth. They are not exact solver clause counts.

## MRR Sweeps

Runs with `eps_sweep` are expanded into one run per epsilon. The aggregate script computes `mrr_discrete`, the largest tested input epsilon whose contract verification succeeds.

Override sweep values for MRR-enabled runs:

```bash
python tool/scripts/run_article_experiments.py \
  --only iris \
  --mrr-mode discrete \
  --mrr-eps-values 0.05,0.1,0.2,0.3
```

MRR is enabled by default only for small benchmarks such as Iris, Seeds, and MNIST 1blk_10. Larger MNIST models use fixed epsilon unless explicitly enabled.

## Statuses

`contract_status` is one of `VERIFIED`, `FAILED`, `TIMEOUT`, `MEMOUT`, `UNKNOWN`, or `SKIPPED`.

`no_saturation_status` uses the same status vocabulary. In article runs, formal no-saturation is optional unless `require_formal_no_saturation` is set. A timeout or memory-out in an optional no-saturation check is reported as evidence, not as a contract failure.

`final_status` is:

- `VERIFIED` when the contract, deployment quality, Python/C exact match, and required optional checks all pass.
- `PARTIAL_VERIFIED` when the contract and deployment quality pass but optional no-saturation evidence is skipped or inconclusive.
- `FAILED`, `TIMEOUT`, `MEMOUT`, or `UNKNOWN` when the main contract or deployment-quality checks do not pass.

## Epsilon Terminology

`--eps` is the input perturbation radius:

```text
x_low = clip(sample - input_epsilon)
x_high = clip(sample + input_epsilon)
```

The aliases `--input-epsilon` and `--perturbation-radius` are preferred in article scripts. Outputs report `input_epsilon` and `normalized_input_epsilon = input_epsilon / dataset.input_scale`.

Generated hidden-layer harnesses use `preimage_tolerance` for interval acceptance. This is separate from the input perturbation radius.

## Block-Wise Soundness

For hidden affine layer contracts, each output neuron has an independent affine expression over the same input box and fixed Q/I/F allocation. Splitting the output layer into contiguous blocks verifies the same per-neuron postcondition over the same assumptions, then conjoins the block results. This is a decomposition of the structured harness query, not a weakening of the layer-wise preimage contract.

## Monolithic and Full-Layer Baselines

Full-layer and monolithic C-to-SMT checks are resource-controlled with ESBMC timeout and memory limits. `TIMEOUT`, `MEMOUT`, and `UNKNOWN` are valid scalability evidence and are kept in the aggregate tables instead of being filtered out.

