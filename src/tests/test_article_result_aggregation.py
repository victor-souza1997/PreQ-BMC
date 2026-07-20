from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
import unittest

from scripts import aggregate_article_results as aggregate_script
from scripts import run_article_experiments as article_runner


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _experiment_summary(sample_id: int, *, c_accuracy: float, runtime: float) -> tuple[dict, dict, dict]:
    method = {
        "success": True,
        "accepted": True,
        "Q": [8],
        "I": [3],
        "F": [4],
        "total_bits_sum": 8,
        "weighted_avg_bits_per_parameter": 8.0,
        "contract_verified": True,
        "contract_status": "VERIFIED",
        "no_saturation_status": "SKIPPED",
        "no_saturation_verified": False,
        "deployment_quality_accepted": True,
        "python_c_exact_match": True,
        "final_status": "VERIFIED",
        "guarantee_level": "deployed-transfer",
        "deployment_metrics": {
            "quantized_keras_accuracy": 0.9,
            "python_fixed_accuracy": c_accuracy,
            "c_fixed_accuracy": c_accuracy,
            "mismatch_rate_vs_keras": 0.1 * sample_id,
            "max_abs_logit_error": 0.01 * (sample_id + 1),
            "mean_abs_logit_error": 0.005,
            "max_saturation_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "python_c_exact_match": True,
        },
        "resource_metrics": {
            "num_parameters": 50,
            "float_parameter_memory_bytes": 200,
            "fixed_parameter_memory_bytes": 50,
            "compression_ratio_vs_float32": 4.0,
            "activation_memory_bytes_estimate": 10,
            "peak_activation_values": 10,
        },
        "verification_stats": {"esbmc_calls": 2},
    }
    experiment = {
        "benchmark": {
            "dataset": "iris",
            "arch": "1blk_10",
            "sample_id": sample_id,
            "eps": 0.01,
            "clean_margin": 1.0 + sample_id,
        },
        "reference": {
            "full_precision_keras_accuracy": 0.95,
            "predicted_label": sample_id,
            "sample_label": sample_id,
            "clean_margin": 1.0 + sample_id,
        },
        "formal_only": dict(method),
        "quality_refined": dict(method),
        "guarantee_level": "deployed-transfer",
    }
    pipeline = {
        "dataset": "iris",
        "arch": "1blk_10",
        "sample_id": sample_id,
        "eps": 0.01,
        "input_epsilon": 0.01,
        "clean_margin": 1.0 + sample_id,
        "sample_label": sample_id,
        "predicted_label": sample_id,
        "timing_metrics": {
            "total_runtime_seconds": runtime,
            "total_esbmc_time_seconds": runtime / 2.0,
            "number_of_esbmc_calls": 2,
            "max_esbmc_query_time_seconds": runtime / 4.0,
        },
        "esbmc_status_counts": {
            "esbmc_verified_count": 2,
            "esbmc_failed_count": 0,
            "esbmc_timeout_count": 0,
            "esbmc_memout_count": 0,
            "esbmc_unknown_count": 0,
            "esbmc_total_count": 2,
        },
        "blockwise_verification": {
            "enabled": True,
            "block_size": 10,
            "total_blocks": 1,
            "verified_blocks": 1,
            "failed_blocks": 0,
            "timeout_blocks": 0,
            "memout_blocks": 0,
            "unknown_blocks": 0,
            "largest_neurons_per_query": 10,
            "largest_input_dim_per_query": 4,
            "largest_estimated_macs_per_query": 40,
        },
    }
    status = {
        "name": f"iris_sample{sample_id}",
        "dataset": "iris",
        "arch": "1blk_10",
        "sample_id": sample_id,
        "input_epsilon": 0.01,
        "status": "success",
        "elapsed_seconds": runtime,
    }
    return experiment, pipeline, status


class ArticleResultAggregationTest(unittest.TestCase):
    def test_aggregate_writes_multisample_summary_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "runs"
            output_root = root / "results"
            for sample_id, runtime in enumerate([10.0, 20.0]):
                experiment, pipeline, status = _experiment_summary(
                    sample_id,
                    c_accuracy=0.8 + 0.1 * sample_id,
                    runtime=runtime,
                )
                run_dir = input_root / f"iris_sample{sample_id}"
                _write_json(run_dir / "reports" / "experiment_summary.json", experiment)
                _write_json(run_dir / "reports" / "pipeline_summary.json", pipeline)
                _write_json(run_dir / "run_status.json", status)
                _write_json(run_dir / "run_config.json", {"mode": "full_pipeline"})

            artifacts = aggregate_script.aggregate(input_root, output_root)

            self.assertIn("table_region_certification_summary_csv", artifacts)
            self.assertTrue((output_root / "latex" / "table_main_summary_compact.tex").exists())
            with (output_root / "all_experiments.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("clean_margin", rows[0])

            with (output_root / "table_region_certification_summary.csv").open(newline="", encoding="utf-8") as handle:
                region_rows = list(csv.DictReader(handle))
            quality_region = next(row for row in region_rows if row["method"] == "quality_refined")
            self.assertEqual(quality_region["n_regions"], "2")
            self.assertEqual(quality_region["certified_count"], "2")
            self.assertEqual(float(quality_region["certified_fraction"]), 1.0)
            self.assertEqual(quality_region["L3_count"], "2")

            with (output_root / "table_deployment_quality_summary.csv").open(newline="", encoding="utf-8") as handle:
                deployment_rows = list(csv.DictReader(handle))
            quality_deployment = next(row for row in deployment_rows if row["method"] == "quality_refined")
            self.assertAlmostEqual(float(quality_deployment["c_fixed_accuracy_mean"]), 0.85)

            with (output_root / "table_runtime_summary.csv").open(newline="", encoding="utf-8") as handle:
                runtime_rows = list(csv.DictReader(handle))
            quality_runtime = next(row for row in runtime_rows if row["method"] == "quality_refined")
            self.assertAlmostEqual(float(quality_runtime["total_runtime_seconds_median"]), 15.0)
            self.assertAlmostEqual(float(quality_runtime["total_runtime_seconds_iqr"]), 5.0)


class ArticleSampleSelectionTest(unittest.TestCase):
    def test_stratified_margin_selection_expands_low_median_high_regions(self) -> None:
        def fake_records(dataset: str, arch: str) -> list[dict]:
            return [
                {"sample_id": 0, "predicted_label": 0, "sample_label": 0, "clean_margin": 0.1, "correctly_classified": True},
                {"sample_id": 1, "predicted_label": 1, "sample_label": 1, "clean_margin": 1.0, "correctly_classified": True},
                {"sample_id": 2, "predicted_label": 2, "sample_label": 2, "clean_margin": 10.0, "correctly_classified": True},
            ]

        original = article_runner._margin_records_for_benchmark
        article_runner._margin_records_for_benchmark = fake_records
        try:
            args = argparse.Namespace(
                mrr_eps_values=None,
                mrr_binary_low=None,
                mrr_binary_high=None,
                mrr_binary_iters=8,
                unsound_contract_tolerance=None,
                propagate_contract_tolerance=None,
                enforce_contract_chaining=None,
                mrr_mode=None,
                include_disabled=False,
                only=[],
                skip=[],
            )
            runs = article_runner._expand_runs(
                {
                    "defaults": {},
                    "runs": [
                        {
                            "name": "iris_stratified",
                            "dataset": "iris",
                            "arch": "1blk_10",
                            "sample_selection": "stratified_by_margin",
                            "samples_per_stratum": 1,
                            "eps": 0.01,
                        }
                    ],
                },
                args,
            )
        finally:
            article_runner._margin_records_for_benchmark = original

        self.assertEqual([run["sample_id"] for run in runs], [0, 1, 2])
        self.assertEqual([run["sample_selection_stratum"] for run in runs], ["low", "median", "high"])
        self.assertEqual([run["clean_margin"] for run in runs], [0.1, 1.0, 10.0])


if __name__ == "__main__":
    unittest.main()
