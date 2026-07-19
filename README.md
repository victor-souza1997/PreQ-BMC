# PreQ-BMC

## Documentation Guide
- **Step 1: Setup:** [[Complete installation guide](docs/installation.md)] 
- **Step 2: Validation:** [[Complete evaluation](docs/reproducing_article_results.md)] 
- **Step 3: Experiments:** [[Complete reproducibility](docs/artifact_evaluation.md)]

## Tool Description
PreQ-BMC is a methodology and tool pipeline designed to synthesize a fixed-point Quantized Neural Network (QNN) from a trained floating-point model, preserving local robustness properties and maintaining deployment accuracy.

The central goal is to find a per-layer fixed-point configuration `(Q, I, F)` that is compact enough for embedded systems, but still preserves the original robustness contract. Instead of simply verifying the entire network at once, the tool decomposes the problem: it uses preimages to create per-layer contracts and invokes `ESBMC` to mathematically prove that the quantized version respects these boundaries. This is complemented with empirical and formal saturation tests to ensure the C code behaves exactly as expected in the real world.

### Key Features
- Preimage-Guided Synthesis: Uses backward preimage propagation (via `DeepPoly` abstraction and `MILP`) to break the end-to-end problem into layer-wise robustness contracts.
- Rigorous Formal Verification: Generates C harnesses and invokes `ESBMC` to verify whether the quantized affine transformations respect the preimage conditions.
- Block-wise Verification: Scales the verification of larger layers by breaking them down into smaller blocks of neurons in `ESBMC`, without introducing mixed precision.
- Quality and Saturation Validation: Iteratively refines bit-widths `(Q, I, F)` if the network suffers accuracy drops, mismatch errors, or saturation, offering both empirical and formal guarantees against saturation.
- Absolute Semantic Fidelity: The Python interpreter and the generated C backend share the exact same arithmetic operations as `ESBMC` (such as `__int128` accumulators, round-half-away-from-zero, and exact clamp limits).

## Dependencies
- Python 3.10 or newer.
- ESBMC (can be installed locally in the repository via an included script).
- GCC (or another configured compiler) for local C backend compilation.
- CBC / python-mip (Installed by default via pip to ensure open-source reproducibility of the MILP synthesis).
- Optional: Gurobi and gurobipy (only required if you want to export new preimage caches or use the commercial solver as a baseline/reference).

## Installation
Clone this repository, create a virtual environment, and activate it:

```ash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
Install the package with all pipeline dependencies, including the default solver (CBC) and plotting libraries:
```Bash
pip install -e '.[full]'
```
Install ESBMC into the repository environment and verify that the tools are recognized:
```Bash
preqbmc install-esbmc
preqbmc verify-environment
```
## First Example
To confirm the entire architecture is working, run the quick demo. It uses the repository's included cache to perform the quantization, C generation, and verification steps without needing the Gurobi solver:

```Bash
preqbmc demo --no-gurobi --output output/demo_run
```
This command will create an `output/demo_run/` folder containing:

- The verification and synthesis reports (reports/).
- The generated C verification code (harnesses) for each layer (layers/).
- The final library and code ready for embedded systems (c_export/qnn_model.c).

## How to Reproduce Experiments
The tool already includes a runner programmed to execute the test batches required for the article and generate spreadsheets and plots automatically.

### 1. Dry Run
Before running everything, test if the config reader is working properly by simulating a single case without invoking ESBMC:

```Bash
preqbmc reproduce --config experiments/article_experiments.json --only iris --max-runs 1 --dry-run
```
### 2. Full Run and Aggregation
To run the experiments, generate raw data, consolidate them into .csv files, and create the final plots, use:

```Bash
preqbmc reproduce \
  --config experiments/article_experiments.json \
  --continue-on-error \
  --aggregate \
  --plots
```
### 3. Aggregation Only
If the experiments have already been successfully completed and the results are saved in output/article_runs, you can skip the synthesis and run only the table and plot generation:

```Bash
preqbmc aggregate \
  --input-root output/article_runs \
  --output-root output/article_results \
  --plots
```

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
