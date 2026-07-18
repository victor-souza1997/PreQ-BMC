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
            "contract_verified",
            "contract_status",
            "no_saturation_formally_checked",
            "no_saturation_status",
            "no_saturation_verified",
            "deployment_quality_accepted",
            "python_c_exact_match",
            "final_status",
            "keras_quantized_accuracy",
            "python_fixed_accuracy",
            "c_fixed_accuracy",
            "mismatch_rate_vs_keras",
            "max_saturation_rate",
            "mean_abs_logit_error",
            "total_time",
            "refinement_steps",
            "formal_saturation_check",
            "empirical_saturation_check",
            "no_saturation_blocks_total",
            "no_saturation_blocks_verified",
            "no_saturation_blocks_failed",
            "no_saturation_blocks_timeout",
            "no_saturation_blocks_memout",
            "no_saturation_blocks_unknown",
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

    semantics_path = _write_csv(
        reports_dir / "table_fixed_point_semantics.csv",
        [
            "dataset",
            "arch",
            "method",
            "layer_index",
            "Q",
            "I",
            "F",
            "scale_factor",
            "q_min_int",
            "q_max_int",
            "real_min",
            "real_max",
            "overflow_mode",
            "rounding_mode",
            "max_abs_accumulator",
            "fits_int64",
            "max_saturation_rate",
        ],
        _fixed_point_semantics_rows(benchmark, "formal_only", formal)
        + _fixed_point_semantics_rows(benchmark, "quality_refined", refined),
    )

    return {
        "table_formal_vs_refined": str(formal_vs_refined_path),
        "table_deployment_metrics": str(deployment_path),
        "table_resource_metrics": str(resource_path),
        "table_refinement_history": str(history_path),
        "table_fixed_point_semantics": str(semantics_path),
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
        "contract_verified": section.get("contract_verified"),
        "contract_status": section.get("contract_status"),
        "no_saturation_formally_checked": section.get("no_saturation_formally_checked"),
        "no_saturation_status": _no_saturation_status(section),
        "no_saturation_verified": section.get("no_saturation_verified"),
        "deployment_quality_accepted": section.get("deployment_quality_accepted"),
        "python_c_exact_match": section.get("python_c_exact_match"),
        "final_status": section.get("final_status"),
        "keras_quantized_accuracy": deployment.get("quantized_keras_accuracy"),
        "python_fixed_accuracy": deployment.get("python_fixed_accuracy"),
        "c_fixed_accuracy": deployment.get("c_fixed_accuracy"),
        "mismatch_rate_vs_keras": deployment.get("mismatch_rate_vs_keras"),
        "max_saturation_rate": deployment.get("max_saturation_rate"),
        "mean_abs_logit_error": deployment.get("mean_abs_logit_error"),
        "total_time": stats.get("total_time"),
        "refinement_steps": refinement_steps,
        "formal_saturation_check": section.get("formal_saturation_check_enabled"),
        "empirical_saturation_check": section.get("empirical_saturation_check_enabled"),
        "no_saturation_blocks_total": section.get("no_saturation_blocks_total"),
        "no_saturation_blocks_verified": section.get("no_saturation_blocks_verified"),
        "no_saturation_blocks_failed": section.get("no_saturation_blocks_failed"),
        "no_saturation_blocks_timeout": section.get("no_saturation_blocks_timeout"),
        "no_saturation_blocks_memout": section.get("no_saturation_blocks_memout"),
        "no_saturation_blocks_unknown": section.get("no_saturation_blocks_unknown"),
    }


def _deployment_row(benchmark: dict[str, Any], method: str, section: dict[str, Any]) -> dict[str, Any]:
    deployment = section.get("deployment_metrics", {})
    return {**_benchmark_cells(benchmark), "method": method, **deployment}


def _resource_row(benchmark: dict[str, Any], method: str, resource: dict[str, Any]) -> dict[str, Any]:
    return {**_benchmark_cells(benchmark), "method": method, **resource}


def _fixed_point_semantics_rows(
    benchmark: dict[str, Any],
    method: str,
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    layers = section.get("fixed_point_semantics", {}).get("layers", [])
    accumulator_by_layer = {
        int(layer.get("layer_index", index)): layer
        for index, layer in enumerate(section.get("accumulator_range", []))
    }
    saturation_rates = section.get("deployment_metrics", {}).get("per_layer_saturation_rates", [])
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        layer_index = int(layer.get("layer_index", index))
        accumulator = accumulator_by_layer.get(layer_index, {})
        saturation_rate = (
            saturation_rates[layer_index]
            if layer_index < len(saturation_rates)
            else None
        )
        rows.append(
            {
                **_benchmark_cells(benchmark),
                "method": method,
                "layer_index": layer_index,
                "Q": layer.get("total_bits"),
                "I": layer.get("integer_bits"),
                "F": layer.get("fractional_bits"),
                "scale_factor": layer.get("scale_factor"),
                "q_min_int": layer.get("q_min_int"),
                "q_max_int": layer.get("q_max_int"),
                "real_min": layer.get("real_min"),
                "real_max": layer.get("real_max"),
                "overflow_mode": layer.get("overflow_mode"),
                "rounding_mode": layer.get("rounding_mode"),
                "max_abs_accumulator": accumulator.get("max_abs_accumulator"),
                "fits_int64": accumulator.get("fits_int64"),
                "max_saturation_rate": saturation_rate,
            }
        )
    return rows


def _no_saturation_status(section: dict[str, Any]) -> str:
    explicit_status = section.get("no_saturation_status")
    if explicit_status:
        return str(explicit_status)
    if not section.get("formal_saturation_check_enabled", False):
        return "SKIPPED"
    if section.get("no_saturation_verified_all_layers", False):
        return "VERIFIED"
    if section.get("no_saturation_failed_layers"):
        return "FAILED"
    return "UNKNOWN"
