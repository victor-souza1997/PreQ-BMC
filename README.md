# PreQ-BMC

PreQ-BMC is a research tool for deployment-aware fixed-point quantization of neural-network classifiers. It synthesizes one fixed-point format `<Q,I,F>` per affine layer and checks whether the generated fixed-point implementation preserves a local robustness contract.

This repository is prepared for the SBSeg Tool Track / Salao de Ferramentas artifact. The public entry point is the `preqbmc` command, with the lower-level scripts kept under `src/scripts/` for reproducibility.

## Documentation Guide

- [Installation](docs/installation.md): Python dependencies, CBC/Gurobi options, and repo-local ESBMC installation.
- [Reproducibility](docs/REPRODUCIBILITY.md): smoke test, article runner, strict/diagnostic run modes, and manual commands.
- [Expected results](docs/expected_results.md): output files, status fields, aggregate tables, and how to interpret failed or unknown runs.


## Tool Description

The tool starts from a trained floating-point model, an input sample, and an input perturbation radius. It computes layer-wise preimage contracts and searches for compact fixed-point formats that preserve those contracts. Generated C harnesses are checked by ESBMC, and the selected configuration is evaluated through Python fixed-point and generated-C deployment backends.

The main engineering contributions in this artifact are:

- preimage-guided per-layer fixed-point bit-width synthesis;
- generated ESBMC C harnesses for layer-wise contract checking;
- block-wise hidden-layer ESBMC verification with a shared layer `<Q,I,F>`, not mixed precision;
- optional formal no-saturation checks and empirical saturation diagnostics;
- CBC through `python-mip` as the default open-source MILP backend, with Gurobi retained as an optional reference backend;
- generated C fixed-point backend and Python-vs-C exact-match diagnostics;
- article experiment runners, aggregation scripts, tables, plots, and machine-readable summaries.

## Scope

PreQ-BMC currently targets the supported affine/ReLU MLP-style benchmark pipeline in this repository, including Iris, Seeds, and selected MNIST architectures. Claims are local to the selected input region and benchmark configuration. `TIMEOUT`, `MEMOUT`, and `UNKNOWN` statuses are reported as inconclusive solver outcomes, not as proofs.

The strongest deployment-oriented claim should be read through `guarantee_level` in `reports/experiment_summary.json`:

- `deployed-transfer`: all transfer preconditions recorded by the artifact are satisfied;
- `harness-verified`: ESBMC harness contracts verified, but at least one transfer precondition is missing or diagnostic-only;
- `failed`: a definite formal or deployment-quality failure;
- `unknown`: insufficient conclusive evidence.

## Quick Installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[full]'
preqbmc install-esbmc
preqbmc verify-environment
```

The ESBMC installer downloads the latest matching ESBMC release into `.local/` and exposes `.local/bin/esbmc`. The runner checks `PREQBMC_ESBMC`, then `.local/bin/esbmc`, then the system `PATH`.

## Quick Artifact Demo

The cached Iris demo avoids Gurobi and exercises harness generation, ESBMC verification, deployment export, and reports:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Expected key outputs:

```text
output/demo_run/reports/pipeline_summary.json
output/demo_run/reports/experiment_summary.json
output/demo_run/reports/qnn_vs_keras_metrics.json
output/demo_run/layers/
output/demo_run/c_export/qnn_model.c
```

## Reproducing Article Experiments

Dry run one Iris case:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --only iris \
  --max-runs 1 \
  --dry-run
```

Run configured article experiments, aggregate CSV tables, and generate plots:

```bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```

Aggregate existing runs:

```bash
preqbmc aggregate \
  --input-root output/article_runs \
  --output-root output/article_results \
  --plots
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for strict soundness-oriented flags and diagnostic compatibility modes.

## License

This repository includes an Apache-2.0 `LICENSE` file. Confirm the final citation metadata and any third-party redistribution constraints before the camera-ready artifact release.
