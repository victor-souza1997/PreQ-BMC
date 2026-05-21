from __future__ import annotations

from typing import Any


def summarize_saturation(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize per-layer saturation diagnostics."""

    layers = (diagnostics or {}).get("layers", [])
    rates = [float(layer.get("saturation_rate", 0.0)) for layer in layers]
    if not rates:
        return {
            "max_saturation_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "layer_with_max_saturation": None,
            "per_layer_saturation_rates": [],
        }
    max_index = max(range(len(rates)), key=lambda index: rates[index])
    return {
        "max_saturation_rate": float(rates[max_index]),
        "mean_saturation_rate": float(sum(rates) / len(rates)),
        "layer_with_max_saturation": int(layers[max_index].get("layer_index", max_index)),
        "per_layer_saturation_rates": rates,
    }


def _bits_payload(result: dict[str, Any]) -> dict[str, Any]:
    total_bits = [int(value) for value in result.get("total_bits", [])]
    integer_bits = [int(value) for value in result.get("integer_bits", [])]
    fractional_bits = [int(value) for value in result.get("fractional_bits", [])]
    return {
        "Q": total_bits,
        "I": integer_bits,
        "F": fractional_bits,
        "total_bits_sum": int(sum(total_bits)),
    }


def _weighted_avg_bits(resource_metrics: dict[str, Any] | None, result: dict[str, Any]) -> float:
    if resource_metrics and resource_metrics.get("weighted_avg_bits_per_parameter") is not None:
        return float(resource_metrics["weighted_avg_bits_per_parameter"])
    bits = _bits_payload(result)["Q"]
    return float(sum(bits) / len(bits)) if bits else 0.0


def _python_c_exact_match(metrics: dict[str, Any]) -> bool | None:
    comparison = metrics.get("python_c_integer_comparison")
    if comparison is None:
        return None
    return bool(comparison.get("exact_match", False))


def deployment_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostics = metrics.get("fixed_point_diagnostics", {}).get("python", {})
    saturation = summarize_saturation(diagnostics)
    return {
        "quantized_keras_accuracy": metrics.get("keras_quantized_accuracy"),
        "python_fixed_accuracy": metrics.get("python_qnn_accuracy"),
        "c_fixed_accuracy": metrics.get("c_qnn_accuracy"),
        "max_saturation_rate": saturation["max_saturation_rate"],
        "mean_saturation_rate": saturation["mean_saturation_rate"],
        "layer_with_max_saturation": saturation["layer_with_max_saturation"],
        "per_layer_saturation_rates": saturation["per_layer_saturation_rates"],
        "mismatch_rate_vs_keras": metrics.get("python_qnn_mismatch_rate_vs_keras"),
        "max_abs_logit_error": metrics.get("python_qnn_max_abs_error"),
        "mean_abs_logit_error": metrics.get("python_qnn_mean_abs_error"),
        "python_c_exact_match": _python_c_exact_match(metrics),
    }


def _formal_saturation_controls(
    pipeline_summary: dict[str, Any],
    layers_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    verification = pipeline_summary.get("formal_saturation_verification", {})
    enabled = bool(verification.get("enabled", pipeline_summary.get("formal_saturation_check_enabled", False)))
    empirical_enabled = bool(pipeline_summary.get("empirical_saturation_check_enabled", True))
    layers = layers_override if layers_override is not None else verification.get("layers", [])
    failed_layers = [
        int(layer.get("layer_index", index))
        for index, layer in enumerate(layers)
        if layer.get("no_saturation_status") not in {"VERIFIED", "DISABLED"}
    ]
    verified_all = bool(
        enabled
        and layers
        and all(layer.get("no_saturation_status") == "VERIFIED" for layer in layers)
    )
    return {
        "formal_saturation_check_enabled": enabled,
        "empirical_saturation_check_enabled": empirical_enabled,
        "no_saturation_verified_all_layers": verified_all,
        "no_saturation_failed_layers": failed_layers,
    }


def build_experiment_summary(
    *,
    pipeline_summary: dict[str, Any],
    formal_metrics: dict[str, Any] | None,
    refined_metrics: dict[str, Any] | None,
    formal_resource_metrics: dict[str, Any] | None,
    refined_resource_metrics: dict[str, Any] | None,
    external_baselines: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Build the paper-ready consolidated experiment summary."""

    synthesis = pipeline_summary.get("synthesis", {})
    formal_synthesis = pipeline_summary.get("formal_synthesis", synthesis)
    quality = pipeline_summary.get("quality_refinement", {})
    comparison = refined_metrics or pipeline_summary.get("comparison", {})
    formal_metrics = formal_metrics or comparison
    refined_metrics = refined_metrics or comparison

    formal_bits = _bits_payload(formal_synthesis)
    refined_bits = _bits_payload(synthesis)
    formal_stats = formal_synthesis.get("stats", {})
    refined_stats = synthesis.get("stats", {})
    samples_evaluated = comparison.get("samples_evaluated")
    refinement_steps = quality.get("steps", [])
    formal_esbmc_layers = (
        refinement_steps[0].get("esbmc", {}).get("layers", [])
        if refinement_steps
        else pipeline_summary.get("formal_saturation_verification", {}).get("layers", [])
    )
    refined_esbmc_layers = _final_esbmc_layers(quality) or pipeline_summary.get(
        "formal_saturation_verification", {}
    ).get("layers", [])
    formal_saturation_controls = _formal_saturation_controls(pipeline_summary, formal_esbmc_layers)
    refined_saturation_controls = _formal_saturation_controls(pipeline_summary, refined_esbmc_layers)
    semantics_by_method = pipeline_summary.get("fixed_point_semantics_by_method", {})
    accumulator_by_method = pipeline_summary.get("accumulator_range_by_method", {})
    formal_semantics = semantics_by_method.get("formal_only", pipeline_summary.get("fixed_point_semantics", {}))
    refined_semantics = semantics_by_method.get("quality_refined", pipeline_summary.get("fixed_point_semantics", {}))
    formal_accumulator_range = accumulator_by_method.get("formal_only", pipeline_summary.get("accumulator_range", []))
    refined_accumulator_range = accumulator_by_method.get(
        "quality_refined",
        pipeline_summary.get("accumulator_range", []),
    )

    formal_section = {
        "success": bool(formal_synthesis.get("success", False)),
        **formal_bits,
        "weighted_avg_bits_per_parameter": _weighted_avg_bits(formal_resource_metrics, formal_synthesis),
        "verification_stats": {
            "backward_time": formal_stats.get("backward_time"),
            "forward_time": formal_stats.get("forward_time"),
            "total_time": formal_stats.get("total_time"),
            "esbmc_calls": formal_stats.get("esbmc_calls"),
        },
        "deployment_metrics": deployment_metrics(formal_metrics),
        "resource_metrics": formal_resource_metrics or {},
        "fixed_point_semantics": formal_semantics,
        "accumulator_range": formal_accumulator_range,
        **formal_saturation_controls,
    }

    refined_section = {
        "enabled": bool(quality.get("enabled", False)),
        "accepted": bool(quality.get("accepted", False)),
        **refined_bits,
        "refinement_steps": len(quality.get("steps", [])),
        "final_reason": quality.get("final_reason"),
        "esbmc_status_per_layer": _final_esbmc_layers(quality),
        "verification_stats": {
            "backward_time": refined_stats.get("backward_time"),
            "forward_time": refined_stats.get("forward_time"),
            "total_time": refined_stats.get("total_time"),
            "esbmc_calls": refined_stats.get("esbmc_calls"),
        },
        "deployment_metrics": deployment_metrics(refined_metrics),
        "resource_metrics": refined_resource_metrics or {},
        "fixed_point_semantics": refined_semantics,
        "accumulator_range": refined_accumulator_range,
        **refined_saturation_controls,
    }

    return {
        "benchmark": {
            "dataset": pipeline_summary.get("dataset"),
            "base_dataset": pipeline_summary.get("base_dataset"),
            "arch": pipeline_summary.get("arch"),
            "sample_id": pipeline_summary.get("sample_id"),
            "eps": pipeline_summary.get("eps"),
            "compare_split": pipeline_summary.get("compare_split"),
            "samples_evaluated": samples_evaluated,
        },
        "reference": {
            "full_precision_keras_accuracy": pipeline_summary.get("baseline", {}).get("reference_accuracy"),
            "predicted_label": pipeline_summary.get("predicted_label"),
            "sample_label": pipeline_summary.get("sample_label"),
        },
        "formal_only": formal_section,
        "quality_refined": refined_section,
        "fixed_point_semantics": refined_semantics,
        "accumulator_range": refined_accumulator_range,
        "verification_claims": pipeline_summary.get(
            "verification_claims",
            {
                "fixed_point_semantics": "declared_backend_semantics",
                "accumulator_range": "static_interval_analysis",
                "deployment_metrics": "empirical_dataset_evaluation",
                "formal_saturation_verification": "formal_esbmc_when_enabled",
            },
        ),
        **refined_saturation_controls,
        "external_baselines": external_baselines,
        "artifacts": artifacts,
    }


def _final_esbmc_layers(quality: dict[str, Any]) -> list[dict[str, Any]]:
    for step in reversed(quality.get("steps", [])):
        layers = step.get("esbmc", {}).get("layers")
        if layers:
            return layers
    return []
