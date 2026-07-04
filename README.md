# PreQ-BMC

PreQ-BMC is a preimage-guided bounded model checking framework for deployment-aware verification of fixed-point Quantized Neural Network implementations.

The repository contains the research prototype used for the article experiments and the SBSeg 2026 Tool Track / Salao de Ferramentas artifact. Some internal modules still use historical project names for compatibility, but public commands and documentation use the name PreQ-BMC.

## Key Features

- Layer-wise preimage contracts for neural-network robustness properties.
- Fixed-point C harness generation for ESBMC/BMC checks.
- Block-wise verification for dense hidden layers using a shared layer `<Q,I,F>` format.
- Formal no-saturation checks and empirical implementation diagnostics.
- Python and generated-C deployment comparisons.
- Article experiment runners, aggregation scripts, tables, and plots.

## Scope and Limitations

PreQ-BMC currently targets the supported benchmark subset in this repository: fixed-point affine/ReLU MLP-style networks loaded from the provided benchmark weights. The formal pipeline verifies the existing generated ESBMC harnesses and fixed-point contracts; this artifact preparation does not change the mathematical verification semantics.

Full end-to-end MILP-based preimage synthesis requires a valid Gurobi license. For artifact evaluation, we provide cached preimage contracts for small examples so that reviewers can run harness generation, ESBMC verification, and diagnostics without Gurobi.

Deployment C export already exists for runs with the C backend enabled. The pipeline writes it under:

```text
<run-output>/c_export/qnn_model.c
<run-output>/c_export/qnn_model.so
```

ESBMC verification harnesses are distinct artifacts and are written under:

```text
<run-output>/layers/
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/victor-souza1997/PreQ-BMC.git
cd PreQ-BMC
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For the full experiment pipeline, install optional dependencies:

```bash
pip install -e '.[full]'
```

Optional solver and plotting groups are also available:

```bash
pip install -e '.[gurobi]'
pip install -e '.[plots]'
```

Install ESBMC separately and ensure it is on `PATH`:

```bash
esbmc --version
```

Configure Gurobi only if you will run full MILP preimage synthesis.

## Verify the Environment

```bash
preqbmc verify-environment
```

This command reports Python version, ESBMC availability, optional Gurobi availability, and Python package availability. Missing Gurobi is reported as optional for cached demos.

## Quickstart Without Gurobi

Run the cached Iris demo:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

The demo uses the cache in `examples/preimage_cache/` and writes:

```text
output/demo_run/reports/pipeline_summary.json
output/demo_run/reports/experiment_summary.json
output/demo_run/layers/
output/demo_run/c_export/qnn_model.c
```

If ESBMC is not installed, the command stops before running the pipeline and prints installation guidance.

## Full Synthesis With Gurobi

With ESBMC and Gurobi configured, run the same small example end to end:

```bash
preqbmc demo --output output/demo_run_gurobi
```

Or call the existing pipeline directly:

```bash
python tool/scripts/run_robustness_pipeline.py \
  --dataset iris_15x2 \
  --arch 2blk_15_15 \
  --sample-id 27 \
  --eps 0.05 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --output-dir output/iris_full
```

## Reproducing Article Experiments

Dry-run one Iris article experiment:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --dry-run
```

Run and aggregate article experiments:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

Aggregate existing outputs:

```bash
preqbmc aggregate \
  --input-root output/article_runs \
  --output-root output/article_results \
  --plots
```

See [docs/reproducing_article_results.md](docs/reproducing_article_results.md) for table and figure commands.

## Outputs

Important per-run files:

- `reports/pipeline_summary.json`: full run configuration, verification status, block summaries, and artifact paths.
- `reports/experiment_summary.json`: paper-oriented summary with formal and deployment metrics.
- `reports/qnn_vs_keras_metrics.json`: Python fixed-point, Keras, and generated-C deployment comparison.
- `reports/refinement_history.json`: quality-refinement attempts when enabled.
- `reports/table_*.csv`: per-run CSV tables.
- `layers/*.c`: ESBMC verification harnesses.
- `c_export/qnn_model.c`: generated fixed-point deployment C implementation.

Aggregate outputs under `output/article_results/` include `all_experiments.csv`, `table_quality_metrics.csv`, `table_scalability.csv`, `table_mrr.csv`, plot PNGs, and LaTeX table fragments.

## Tests

Run the available unit tests:

```bash
python -m unittest discover tool/tests
```

Some tests require optional packages such as TensorFlow or host tools such as `gcc`. ESBMC-dependent tests are skipped automatically when `esbmc` is not installed.

## Artifact Evaluation

Start with [docs/artifact_evaluation.md](docs/artifact_evaluation.md). It lists minimal software requirements, which commands require Gurobi or ESBMC, and expected runtime for the cached demo.

## Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/installation.md](docs/installation.md)
- [docs/artifact_evaluation.md](docs/artifact_evaluation.md)
- [docs/metrics.md](docs/metrics.md)
- [docs/reproducing_article_results.md](docs/reproducing_article_results.md)
- [docs/gurobi_and_esbmc.md](docs/gurobi_and_esbmc.md)

## Citation

```bibtex
@inproceedings{preqbmc2026,
  title = {PreQ-BMC: Preimage-Guided Bounded Model Checking of Fixed-Point Quantized Neural Network Implementations},
  author = {TODO},
  booktitle = {SBSeg 2026 Tool Track},
  year = {2026},
  doi = {TODO}
}
```

## License

This repository includes an Apache-2.0 `LICENSE` file.

License choice must be confirmed by all authors before final submission.
