# Quadapter
This is the official webpage for paper *Certified Quantization Strategy Synthesis for Neural Networks*. In this paper, we make the following main contributions:
- We introduce the first quantization strategy synthesis method for neural networks which provably preserves desired properties after quantization;
- WeproposeanovelMILP-basedmethod,tocomputeanunder-approximation of the preimage for each layer efficiently and effectively;
- We implement our methods into a tool Quadapter and conduct extensive experiments to demonstrate the application of the certified quantization for preserving robustness and backdoor-freeness properties.

## Benchmarks in Sections 5.1 & 5.2:

The 50 randomly selected inputs from the test set of the respective dataset (shown by IDs):

```
5346  8564  7059  371   6984  5782  2127  8517  4520  8685
3877  463   5446  7775  9623  5739  5010  7668  892   8825
3523  7997  8561  1613  6934  6781  5554  6301  6220  9873
9384  130   9033  8620  6066  4973  8870  5032  8911  5224
5369  1451  7766  5126  9498  1382  3932  8302  9566  5750
```


## Setup
Please install gurobypy from PyPI:

```shell script
$ pip install gurobipy
```

Please install Gurobi on your machine.

## Running Quadapter for Certified Robustness
```shell script
# Preimage Computation Mode: MILP-based ('--preimg_mode milp'), Abstr-based ('--preimg_mode abstr')
# If relaxed version of Quadapter: yes ('--if_relax 1'), no ('--if_relax 0')
# Input=5346, Attack=2, preimg_mode=milp, OutputFolder=./output/

python Quadapter_robustness_main.py --dataset mnist --arch 1blk_100 --sample_id 5346 --eps 2 --preimg_mode milp --if_relax 0 --outputPath ./output/
```

### Running Quadapter for Certified Backdoor-freeness
```shell script
# Backdoor Info:  --loc_row 1  --loc_col 1 --stamp_size 3 --targetCls 8 --originalCls 10
# Paras for Hypothesis Testing: --K 5 --delta 0.05

python Quadapter_backdoor_main.py --dataset mnist --arch 1blk_100 --bit_lb 2 --loc_row 1  --loc_col 1  --stamp_size 3 --targetCls 8 --originalCls 10 --K 5 --delta 0.05 --preimg_mode milp --ifRelax 1  --outputPath ./output/
```


Running Script

```
# Notas de Execucao


// Gerar cache
python scripts/export_gurobi_preimage_cache.py \
  --datasets mnist \
  --archs 1blk_100 \
  --sample-ids 0 \
  --eps 1.0 \
  --preimage-mode milp \
  --cache-dir output/preimage_cache

// Executar lendo cache
python scripts/run_robustness_pipeline.py \
  --dataset iris_15x2  \
  --arch 2blk_15_15 \
  --sample-id 0 \
  --eps 1.0 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --no-gurobi \
  --preimage-cache-dir output/preimage_cache

// Execucao normal
python3 Quadapter_robustness_main.py --dataset iris_15x2 --arch 1blk_10 --sample_id 25 --eps 0.05 --preimg_mode milp --verify_mode esbmc --ifRelax 0 --outputPath ./output/


python scripts/export_gurobi_preimage_cache.py \
  --datasets iris_15x2 \
  --sample-ids 27 \
  --eps 0.05 \
  --preimage-mode milp \
  --cache-dir output/preimage_cache

```

# Run Experiments
Run from the repo root:

```bash
cd /home/joao/code/Quadapter
```

**Single Experiment**
Example for Iris, sample 25, eps 0.1, ESBMC verification:

```bash
python tool/scripts/run_robustness_pipeline.py \
  --dataset iris_15x2 \
  --arch 1blk_15 \
  --sample-id 25 \
  --eps 0.1 \
  --bit-lb 1 \
  --bit-ub 16 \
  --preimage-mode milp \
  --verify-mode esbmc \
  --compare-limit 0 \
  --output-dir output/iris_15x2_id25_eps0p1
```

By default this now runs:
- existing DeepPoly/backward-preimage verification
- existing ESBMC preimage/argmax checks
- formal ESBMC no-saturation checks
- empirical saturation diagnostics/gating
- quality refinement
- paper table export

Main outputs:

```text
output/iris_15x2_id25_eps0p1/reports/pipeline_summary.json
output/iris_15x2_id25_eps0p1/reports/experiment_summary.json
output/iris_15x2_id25_eps0p1/reports/qnn_vs_keras_metrics.json
output/iris_15x2_id25_eps0p1/reports/refinement_history.json
output/iris_15x2_id25_eps0p1/reports/table_formal_vs_refined.csv
output/iris_15x2_id25_eps0p1/reports/table_deployment_metrics.csv
output/iris_15x2_id25_eps0p1/reports/table_resource_metrics.csv
```

**Useful Flags**
Disable formal no-saturation ESBMC:

```bash
--no-formal-saturation-check
```

Disable empirical saturation rejection while still allowing diagnostics:

```bash
--no-empirical-saturation-check
```

Disable quality refinement entirely, closer to old behavior:

```bash
--max-quality-refinement-steps 0
```

Tune quality thresholds:

```bash
--accuracy-drop-threshold 0.05 \
--saturation-threshold 0.01 \
--mismatch-threshold 0.05 \
--max-quality-refinement-steps 10
```

Add external baseline data:

```bash
--baseline-results-json path/to/baseline.json
```

**Batch Experiments**
Create a config, for example:

```json
{
  "runs": [
    {
      "dataset": "iris_15x2",
      "arch": "1blk_15",
      "sample_ids": [0, 5, 10, 15, 20, 25],
      "eps_values": [0.05, 0.1, 0.2],
      "bit_lb": 1,
      "bit_ub": 16,
      "verify_mode": "esbmc",
      "preimg_mode": "milp",
      "compare_limit": 0
    }
  ]
}
```

Save it as, for example:

```text
experiments/sbesc_iris_seeds.json
```

Run:

```bash
python tool/scripts/run_paper_experiments.py \
  --config experiments/sbesc_iris_seeds.json
```

Aggregate outputs:

```text
output/paper_results/all_experiments.json
output/paper_results/all_experiments.csv
output/paper_results/runs/.../reports/experiment_summary.json
```

You need ESBMC available on `PATH`. If using `preimg_mode=milp`, you also need Gurobi working.