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
    required = bool(
        verification.get(
            "required_for_acceptance",
            pipeline_summary.get("require_formal_no_saturation", True),
        )
    )
    empirical_enabled = bool(pipeline_summary.get("empirical_saturation_check_enabled", True))
    layers = layers_override if layers_override is not None else verification.get("layers", [])
    failed_layers = [
        int(layer.get("layer_index", index))
        for index, layer in enumerate(layers)
        if layer.get("no_saturation_status") not in {"VERIFIED", "DISABLED", "SKIPPED"}
    ]
    verified_all = bool(
        enabled
        and layers
        and all(layer.get("no_saturation_status") == "VERIFIED" for layer in layers)
    )
    return {
        "formal_saturation_check_enabled": enabled,
        "require_formal_no_saturation": required,
        "empirical_saturation_check_enabled": empirical_enabled,
        "no_saturation_verified_all_layers": verified_all,
        "no_saturation_failed_layers": failed_layers,
    }


def _aggregate_status(
    layers: list[dict[str, Any]],
    field: str,
    *,
    default: str = "UNKNOWN",
) -> str:
    statuses = {
        "VERIFIED" if str(layer.get(field, default)) == "VERIFIED_BY_SYNTHESIS" else str(layer.get(field, default))
        for layer in layers
    }
    if not statuses:
        return default
    if statuses == {"VERIFIED"}:
        return "VERIFIED"
    for status in ("FAILED", "TIMEOUT", "MEMOUT", "UNKNOWN", "SKIPPED", "DISABLED"):
        if status in statuses:
            return "SKIPPED" if status == "DISABLED" else status
    return default


def _final_status(
    *,
    contract_verified: bool,
    contract_status: str,
    no_saturation_status: str,
    no_saturation_verified: bool,
    deployment_quality_accepted: bool,
    require_formal_no_saturation: bool,
) -> str:
    if contract_status == "FAILED" or not deployment_quality_accepted:
        return "FAILED"
    if require_formal_no_saturation and no_saturation_status == "FAILED":
        return "FAILED"
    if contract_verified and deployment_quality_accepted and no_saturation_verified:
        return "VERIFIED"
    if contract_verified and deployment_quality_accepted and no_saturation_status in {
        "FAILED",
        "TIMEOUT",
        "MEMOUT",
        "UNKNOWN",
        "SKIPPED",
        "DISABLED",
    }:
        return "PARTIAL_VERIFIED"
    return "UNKNOWN"


def _formal_status_controls(
    layers: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    deployment_quality_accepted: bool,
    require_formal_no_saturation: bool,
) -> dict[str, Any]:
    contract_status = _aggregate_status(layers, "contract_status", default="UNKNOWN")
    contract_verified = bool(layers and all(layer.get("contract_verified", False) for layer in layers))
    no_saturation_status = _aggregate_status(layers, "no_saturation_status", default="SKIPPED")
    no_saturation_verified = bool(
        layers
        and all(layer.get("no_saturation_verified", False) for layer in layers)
    )
    python_c_exact_match = _python_c_exact_match(metrics)
    return {
        "contract_verified": contract_verified,
        "contract_status": contract_status,
        "no_saturation_formally_checked": bool(
            any(layer.get("no_saturation_formally_checked", False) for layer in layers)
        ),
        "no_saturation_status": no_saturation_status,
        "no_saturation_verified": no_saturation_verified,
        "deployment_quality_accepted": bool(deployment_quality_accepted),
        "python_c_exact_match": python_c_exact_match,
        "final_status": _final_status(
            contract_verified=contract_verified,
            contract_status=contract_status,
            no_saturation_status=no_saturation_status,
            no_saturation_verified=no_saturation_verified,
            deployment_quality_accepted=bool(deployment_quality_accepted),
            require_formal_no_saturation=bool(require_formal_no_saturation),
        ),
    }


def _pipeline_soundness_degraded(pipeline_summary: dict[str, Any]) -> bool:
    return str(pipeline_summary.get("soundness", "")).lower() == "degraded"


def _chaining_verified_for_transfer(pipeline_summary: dict[str, Any]) -> bool:
    chaining = pipeline_summary.get("chaining_ok", {})
    if not isinstance(chaining, dict):
        return False
    return bool(chaining.get("all_ok", False)) and bool(chaining.get("enforced", True))


def _no_saturation_continue_on_unknown(pipeline_summary: dict[str, Any]) -> bool:
    resource_controls = pipeline_summary.get("resource_controls", {})
    quality_thresholds = pipeline_summary.get("quality_thresholds", {})
    return bool(
        pipeline_summary.get(
            "no_saturation_continue_on_unknown",
            resource_controls.get(
                "no_saturation_continue_on_unknown",
                quality_thresholds.get("no_saturation_continue_on_unknown", False),
            ),
        )
    )


def _contract_harnesses_require_no_saturation(pipeline_summary: dict[str, Any]) -> bool:
    semantics = pipeline_summary.get("contract_harness_semantics", {})
    if not isinstance(semantics, dict) or not semantics:
        return True
    if "no_saturation_required_for_deployed_transfer" in semantics:
        return bool(semantics["no_saturation_required_for_deployed_transfer"])
    return not bool(semantics.get("clamp_in_contract_harnesses", True))


def _fidelity_by_construction(pipeline_summary: dict[str, Any]) -> bool:
    semantics = pipeline_summary.get("contract_harness_semantics", {})
    if isinstance(semantics, dict):
        return bool(semantics.get("uses_shared_deployed_arithmetic_kernel", False))
    return False


def _guarantee_level(
    *,
    pipeline_summary: dict[str, Any],
    status_controls: dict[str, Any],
) -> dict[str, Any]:
    contract_status = str(status_controls.get("contract_status", "UNKNOWN"))
    no_saturation_status = str(status_controls.get("no_saturation_status", "SKIPPED"))
    deployment_quality_accepted = bool(status_controls.get("deployment_quality_accepted", False))
    contract_verified = bool(status_controls.get("contract_verified", False))
    final_status = str(status_controls.get("final_status", "UNKNOWN"))

    if final_status == "FAILED" or contract_status == "FAILED" or not deployment_quality_accepted:
        return {
            "guarantee_level": "failed",
            "transfer_preconditions": {
                "contracts_verified": contract_verified,
                "fidelity_by_construction": _fidelity_by_construction(pipeline_summary),
                "chaining_ok": _chaining_verified_for_transfer(pipeline_summary),
                "no_saturation_required": _contract_harnesses_require_no_saturation(pipeline_summary),
                "no_saturation_verified_if_required": False,
                "no_saturation_continue_on_unknown": _no_saturation_continue_on_unknown(pipeline_summary),
            },
        }

    if not contract_verified:
        return {
            "guarantee_level": "unknown",
            "transfer_preconditions": {
                "contracts_verified": False,
                "fidelity_by_construction": _fidelity_by_construction(pipeline_summary),
                "chaining_ok": _chaining_verified_for_transfer(pipeline_summary),
                "no_saturation_required": _contract_harnesses_require_no_saturation(pipeline_summary),
                "no_saturation_verified_if_required": False,
                "no_saturation_continue_on_unknown": _no_saturation_continue_on_unknown(pipeline_summary),
            },
        }

    no_saturation_required = _contract_harnesses_require_no_saturation(pipeline_summary)
    no_saturation_verified_if_required = (
        bool(status_controls.get("no_saturation_verified", False))
        and no_saturation_status == "VERIFIED"
        if no_saturation_required
        else True
    )
    preconditions = {
        "contracts_verified": True,
        "fidelity_by_construction": _fidelity_by_construction(pipeline_summary),
        "chaining_ok": _chaining_verified_for_transfer(pipeline_summary),
        "soundness_not_degraded": not _pipeline_soundness_degraded(pipeline_summary),
        "no_saturation_required": no_saturation_required,
        "no_saturation_verified_if_required": no_saturation_verified_if_required,
        "no_saturation_continue_on_unknown": _no_saturation_continue_on_unknown(pipeline_summary),
    }
    deployed_transfer = (
        all(
            bool(preconditions[key])
            for key in (
                "contracts_verified",
                "fidelity_by_construction",
                "chaining_ok",
                "soundness_not_degraded",
                "no_saturation_verified_if_required",
            )
        )
        and not preconditions["no_saturation_continue_on_unknown"]
    )
    return {
        "guarantee_level": "deployed-transfer" if deployed_transfer else "harness-verified",
        "transfer_preconditions": preconditions,
    }


def _blockwise_controls(pipeline_summary: dict[str, Any]) -> dict[str, Any]:
    verification = pipeline_summary.get("blockwise_verification", {})
    return {
        "blockwise_verification_enabled": bool(verification.get("enabled", False)),
        "blockwise_block_size": verification.get("block_size", 0),
        "blockwise_policy": verification.get("policy", "shared_layer_qif"),
        "blockwise_total_blocks": verification.get("total_blocks", 0),
        "blockwise_verified_blocks": verification.get("verified_blocks", 0),
        "blockwise_failed_blocks": verification.get("failed_blocks", 0),
        "blockwise_timeout_blocks": verification.get("timeout_blocks", 0),
        "no_saturation_blocks_total": pipeline_summary.get("no_saturation_blocks_total", 0),
        "no_saturation_blocks_verified": pipeline_summary.get("no_saturation_blocks_verified", 0),
        "no_saturation_blocks_failed": pipeline_summary.get("no_saturation_blocks_failed", 0),
        "no_saturation_blocks_timeout": pipeline_summary.get("no_saturation_blocks_timeout", 0),
        "no_saturation_blocks_memout": pipeline_summary.get("no_saturation_blocks_memout", 0),
        "no_saturation_blocks_unknown": pipeline_summary.get("no_saturation_blocks_unknown", 0),
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
    formal_status_controls = _formal_status_controls(
        formal_esbmc_layers,
        formal_metrics,
        deployment_quality_accepted=bool(formal_synthesis.get("success", False)),
        require_formal_no_saturation=bool(formal_saturation_controls.get("require_formal_no_saturation", True)),
    )
    refined_status_controls = _formal_status_controls(
        refined_esbmc_layers,
        refined_metrics,
        deployment_quality_accepted=bool(quality.get("accepted", False)),
        require_formal_no_saturation=bool(refined_saturation_controls.get("require_formal_no_saturation", True)),
    )
    formal_guarantee_controls = _guarantee_level(
        pipeline_summary=pipeline_summary,
        status_controls=formal_status_controls,
    )
    refined_guarantee_controls = _guarantee_level(
        pipeline_summary=pipeline_summary,
        status_controls=refined_status_controls,
    )
    blockwise_controls = _blockwise_controls(pipeline_summary)
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
        **formal_status_controls,
        **formal_guarantee_controls,
        **blockwise_controls,
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
        **refined_status_controls,
        **refined_guarantee_controls,
        **blockwise_controls,
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
            "clean_margin": pipeline_summary.get("clean_margin"),
        },
        "reference": {
            "full_precision_keras_accuracy": pipeline_summary.get("baseline", {}).get("reference_accuracy"),
            "predicted_label": pipeline_summary.get("predicted_label"),
            "sample_label": pipeline_summary.get("sample_label"),
            "clean_margin": pipeline_summary.get("clean_margin"),
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
                "blockwise_verification": "equivalent_hidden_contract_decomposition_when_enabled",
            },
        ),
        "contract_harness_semantics": pipeline_summary.get("contract_harness_semantics", {}),
        "blockwise_verification": pipeline_summary.get("blockwise_verification", {}),
        "no_saturation_blocks": pipeline_summary.get("no_saturation_blocks", []),
        **refined_saturation_controls,
        **refined_status_controls,
        **refined_guarantee_controls,
        **blockwise_controls,
        "external_baselines": external_baselines,
        "artifacts": artifacts,
    }


def _final_esbmc_layers(quality: dict[str, Any]) -> list[dict[str, Any]]:
    for step in reversed(quality.get("steps", [])):
        layers = step.get("esbmc", {}).get("layers")
        if layers:
            return layers
    return []
