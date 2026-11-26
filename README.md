# PreQ-BMC
📹 **Demonstration video (SBSeg 2026 - Salão de Ferramentas):** [https://www.youtube.com/watch?v=H2iu2q03Bcw](https://www.youtube.com/watch?v=H2iu2q03Bcw)

PreQ-BMC is a research tool for deployment-aware fixed-point quantization of neural-network classifiers. It synthesizes one fixed-point format `<Q,I,F>` per affine layer and checks whether the generated fixed-point implementation preserves a local robustness contract.

This repository accompanies the SBESC SSV 2026 and LADC 2026 papers listed below. The public entry point is the `preqbmc` command, with the lower-level scripts kept under `src/scripts/` for reproducibility.

## Documentation Guide

- [Installation](docs/installation.md): Python dependencies, CBC/Gurobi options, and repo-local ESBMC installation.
- [Reproducibility](docs/REPRODUCIBILITY.md): smoke test, article runner, JSON configuration reference, strict/diagnostic run modes, and manual commands.
- [Expected results](docs/expected_results.md): output files, status fields, aggregate tables, and how to interpret failed or unknown runs.
- [Result aggregation](docs/result_aggregation.md): multi-sample local-region summaries, certified fraction, and compact SBSeg tables.


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
python -m pip install -r requirements.txt
preqbmc install-esbmc
preqbmc verify-environment
```

The default requirements install the license-free `paper` dependency group,
including CBC. Gurobi remains optional through
`python -m pip install -e '.[gurobi]'`. The ESBMC installer downloads the
latest matching ESBMC release into `.local/` and exposes
`.local/bin/esbmc`. The runner checks `PREQBMC_ESBMC`, then
`.local/bin/esbmc`, then the system `PATH`.

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
See [docs/result_aggregation.md](docs/result_aggregation.md) for the multi-sample aggregation used for paper tables.

## Published Articles

1. **Verified Edge AI for Android Automotive: Deployment-Aware Verification of Fixed-Point Neural Networks** — SBESC SSV 2026. [Abstract](#sbesc-ssv-2026-abstract) · [BibTeX](#sbesc-ssv-2026)
2. **Preimage-Guided Bit-Precise Verification of Fixed-Point Quantized Neural Networks** — LADC 2026. [Abstract](#ladc-2026-abstract) · [BibTeX](#ladc-2026)

### SBESC SSV 2026 Abstract

As the automotive industry transitions toward Software-Defined Vehicles (SDVs), traffic sign classifiers must operate predictably under embedded resource constraints. This paper presents a deployment-aware case study of fixed-point Convolutional Neural Network (CNN) verification and execution on Android Automotive OS (AAOS) using the German Traffic Sign Recognition Benchmark (GTSRB). We quantized a trained CNN, generated a fixed-point C implementation, verified local robustness properties with ESBMC, and deployed the corresponding implementation on a Khadas VIM3. The selected 16-bit Q7.8 model achieved 91.35% test accuracy. During model selection, the evaluated 8-bit Q3.4 configuration achieved 80.75% validation accuracy. At an L∞ radius of one raw input byte, ESBMC certified two of 80 preregistered test image–radius pairs after a disclosed rerun. For the two certified test images, complete-network verification required approximately 49,000–51,000 solver calls and 12–20 hours summed over ESBMC calls per image. On the Cortex-A73, the Quantized Neural Network (QNN) implementations were slower than float32; their integer loops required longer scalar multiply–accumulate sequences, while the compact int16 variant used less memory. These results characterize the gap between practical fixed-point deployment and the scalability of implementation level robustness verification.

### LADC 2026 Abstract

Quantized neural networks (QNNs) reduce the memory and arithmetic costs of embedded inference, but a robustness guarantee established for an abstract model may not hold for the deployed fixed-point program. We present PreQ-BMC, a deployment-aware workflow that synthesizes per-layer fixed-point formats, generates C code, composes layer contracts, and verifies finite-width arithmetic and local robustness properties using the Efficient SMT-Based Context-Bounded Model Checker (ESBMC). The PreQ-BMC verifies the generated implementation, including input quantization, rounding, accumulation, saturation, and the ordering of operations. It targets small fully connected ReLU networks with per-layer quantization. We evaluate PreQ-BMC on small fully connected ReLU networks over MNIST, Iris, and Seeds. Across five networks and 26 encoded-domain regions, all contract checks passed, including 13 MNIST regions. For MNIST 10, reducing the block size from five neurons to one lowered the maximum hidden-query time from 68.5 to 2.6 s and sampled RSS from 610 to 190 MiB. Full-test C accuracy ranged from 89% to 91.31% across five networks, while nominal packed-parameter reductions relative to float32 ranged from 59.6% to 78.4%.

## Citation

### SBESC SSV 2026

```bibtex
@inproceedings{Souza2026VerifiedEdgeAI,
  author    = {Souza, João Victor Lima De and
               Torres, Gustavo Modesto and
               Conde, Wesley Borges and
               Ferreira, David Alan De Oliveira and
               Sousa, Janislley Oliveira De},
  title     = {Verified Edge {AI} for {Android Automotive}:
               Deployment-Aware Verification of Fixed-Point
               Neural Networks},
  booktitle = {2026 XVI Brazilian Symposium on Computing
               Systems Engineering (SBESC)},
  year      = {2026},
  month     = nov,
  address   = {Salvador, Bahia, Brazil},
  note      = {Special Session on Safe and Secure Vehicles
               (SSV 2026)}
}
```

### LADC 2026

```bibtex
@inproceedings{Souza2026PreQBMC,
  author    = {de Souza, João Victor L. and
               de Sousa, Janislley O. and
               de Bessa, Iury V. and
               Filho, Eddie B. L. and
               Cordeiro, Lucas C.},
  title     = {Preimage-Guided Bit-Precise Verification
               of Fixed-Point Quantized Neural Networks},
  booktitle = {Dependable and Secure Computing:
               15th Latin-American Symposium,
               LADC 2026, Proceedings},
  year      = {2026},
  publisher = {Springer Nature},
  note      = {Salvador, Bahia, Brazil,
               November 24--27, 2026}
}
```

## License

This repository includes an Apache-2.0 `LICENSE` file. Confirm the final citation metadata and any third-party redistribution constraints before the camera-ready artifact release.
