from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})
    return path


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def export_paper_tables(experiment_summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Export paper-oriented CSV tables from the consolidated summary."""

    reports_dir = output_dir / "reports"
    benchmark = experiment_summary.get("benchmark", {})
    formal = experiment_summary.get("formal_only", {})
    refined = experiment_summary.get("quality_refined", {})

    formal_vs_refined_path = _write_csv(
        reports_dir / "table_formal_vs_refined.csv",
        [
            "dataset",
            "arch",
            "sample_id",
            "eps",
            "method",
            "Q",
            "I",
            "F",
            "verified",
            "keras_quantized_accuracy",
            "python_fixed_accuracy",
            "c_fixed_accuracy",
            "mismatch_rate_vs_keras",
            "max_saturation_rate",
            "mean_abs_logit_error",
            "total_time",
            "refinement_steps",
        ],
        [
            _formal_refined_row(benchmark, "formal_only", formal, formal.get("success"), 0),
            _formal_refined_row(
                benchmark,
                "quality_refined",
                refined,
                refined.get("accepted"),
                refined.get("refinement_steps"),
            ),
        ],
    )

    deployment_path = _write_csv(
        reports_dir / "table_deployment_metrics.csv",
        [
            "dataset",
            "arch",
            "sample_id",
            "eps",
            "method",
            "quantized_keras_accuracy",
            "python_fixed_accuracy",
            "c_fixed_accuracy",
            "max_saturation_rate",
            "mean_saturation_rate",
            "mismatch_rate_vs_keras",
            "max_abs_logit_error",
            "mean_abs_logit_error",
            "python_c_exact_match",
        ],
        [
            _deployment_row(benchmark, "formal_only", formal),
            _deployment_row(benchmark, "quality_refined", refined),
        ],
    )

    resource_path = _write_csv(
        reports_dir / "table_resource_metrics.csv",
        [
            "dataset",
            "arch",
            "sample_id",
            "eps",
            "method",
            "num_layers",
            "num_parameters",
            "float_parameter_memory_bytes",
            "fixed_parameter_memory_bytes",
            "compression_ratio_vs_float32",
            "weighted_avg_bits_per_parameter",
            "activation_memory_bytes_estimate",
            "peak_activation_values",
            "c_source_lines",
            "c_shared_library_size_bytes",
        ],
        [
            _resource_row(benchmark, "formal_only", formal.get("resource_metrics", {})),
            _resource_row(benchmark, "quality_refined", refined.get("resource_metrics", {})),
        ],
    )

    history_path = _write_csv(
        reports_dir / "table_refinement_history.csv",
        [
            "dataset",
            "arch",
            "sample_id",
            "eps",
            "step",
            "configuration",
            "accepted",
            "quality_failures",
            "esbmc_status",
            "refinement_action",
        ],
        [
            {
                **_benchmark_cells(benchmark),
                "step": step.get("step"),
                "configuration": step.get("configuration"),
                "accepted": step.get("accepted"),
                "quality_failures": step.get("quality_failures"),
                "esbmc_status": step.get("esbmc", {}).get("status"),
                "refinement_action": step.get("refinement_action"),
            }
            for step in refined.get("steps", [])
        ],
    )

    return {
        "table_formal_vs_refined": str(formal_vs_refined_path),
        "table_deployment_metrics": str(deployment_path),
        "table_resource_metrics": str(resource_path),
        "table_refinement_history": str(history_path),
    }


def _benchmark_cells(benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": benchmark.get("dataset"),
        "arch": benchmark.get("arch"),
        "sample_id": benchmark.get("sample_id"),
        "eps": benchmark.get("eps"),
    }


def _formal_refined_row(
    benchmark: dict[str, Any],
    method: str,
    section: dict[str, Any],
    verified: Any,
    refinement_steps: Any,
) -> dict[str, Any]:
    deployment = section.get("deployment_metrics", {})
    stats = section.get("verification_stats", {})
    return {
        **_benchmark_cells(benchmark),
        "method": method,
        "Q": section.get("Q"),
        "I": section.get("I"),
        "F": section.get("F"),
        "verified": verified,
        "keras_quantized_accuracy": deployment.get("quantized_keras_accuracy"),
        "python_fixed_accuracy": deployment.get("python_fixed_accuracy"),
        "c_fixed_accuracy": deployment.get("c_fixed_accuracy"),
        "mismatch_rate_vs_keras": deployment.get("mismatch_rate_vs_keras"),
        "max_saturation_rate": deployment.get("max_saturation_rate"),
        "mean_abs_logit_error": deployment.get("mean_abs_logit_error"),
        "total_time": stats.get("total_time"),
        "refinement_steps": refinement_steps,
    }


def _deployment_row(benchmark: dict[str, Any], method: str, section: dict[str, Any]) -> dict[str, Any]:
    deployment = section.get("deployment_metrics", {})
    return {**_benchmark_cells(benchmark), "method": method, **deployment}


def _resource_row(benchmark: dict[str, Any], method: str, resource: dict[str, Any]) -> dict[str, Any]:
    return {**_benchmark_cells(benchmark), "method": method, **resource}
