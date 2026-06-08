from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from backends.c_qnn_generator import (
    CompiledCQNN,
    compare_python_c_fixed_point_outputs,
    compile_c_qnn_shared_library,
    write_c_qnn_source,
)
from backends.fixed_point import (
    FixedPointNetwork,
    LayerQuantizationSpec,
    build_fixed_point_network,
    clone_quantized_keras_model,
    compute_accumulator_range_analysis,
    fixed_point_semantics_for_network,
    forward_fixed_point_batch,
    forward_fixed_point_batch_with_diagnostics,
)
from datasets.loaders import DatasetBundle, load_dataset, select_split
from models.loading import (
    build_and_load_deep_model,
    infer_dense_architecture_from_h5,
    normalize_dataset_selection,
    parse_architecture,
    resolve_weight_path,
)
from reports.baseline_import import load_external_baselines
from reports.experiment_summary import build_experiment_summary
from reports.resource_metrics import compute_fixed_point_resource_metrics
from reports.table_export import export_paper_tables
from synthesis.forward import forward_dnn
from synthesis.preimage_cache import build_preimage_cache_identity
from synthesis.quadapter import GPEncoding, QuadapterConfig, SynthesisResult
from utils.logging_utils import get_logger
from verification.esbmc import ESBMCConfig, ESBMCProfile
from verification.properties import ClassificationProperty

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RobustnessPipelineConfig:
    """Top-level configuration for the robustness pipeline CLI."""

    dataset: str
    arch: str
    sample_id: int
    eps: float
    bit_lb: int
    bit_ub: int
    preimg_mode: str
    verify_mode: str
    output_dir: Path
    if_relax: bool = False
    target_label: int | None = None
    valid_labels: tuple[int, ...] | None = None
    compare_split: str = "test"
    compare_limit: int | None = 100
    compile_c_backend: bool = True
    compiler: str = "gcc"
    enable_diagnostics: bool = True
    formal_saturation_check: bool = True
    empirical_saturation_check: bool = True
    accuracy_drop_threshold: float | None = 0.05
    saturation_threshold: float | None = 0.01
    mismatch_threshold: float | None = 0.05
    max_quality_refinement_steps: int = 10
    no_gurobi: bool = False
    save_preimage_cache: bool = False
    preimage_cache_dir: Path | None = None
    preimage_cache_key: str | None = None
    esbmc_layer_block_size: int = 10
    blockwise_fail_fast: bool = True
    blockwise_run_all_blocks_on_failure: bool = False
    esbmc_jobs: int = 1
    esbmc_memlimit: str = "12g"
    esbmc_profile: ESBMCProfile = "paper-fast"
    esbmc_timeout_seconds: int = 9000
    gurobi_threads: int = 4
    export_paper_tables: bool = True
    baseline_results_json: Path | None = None


def _predict_logits(model: Any, features: np.ndarray) -> np.ndarray:
    logits = model(np.asarray(features, dtype=np.float32), training=False)
    return np.asarray(logits.numpy(), dtype=np.float64)


def _compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def _normalize_features(features: np.ndarray, input_scale: float) -> np.ndarray:
    if input_scale in (None, 0):
        return np.asarray(features, dtype=np.float64)
    return np.asarray(features, dtype=np.float64) / float(input_scale)


def _fixed_point_input_bounds(
    network: FixedPointNetwork,
    x_low_real: np.ndarray,
    x_high_real: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 1 << network.input_fractional_bits
    return (
        np.floor(np.asarray(x_low_real, dtype=np.float64) * scale).astype(object),
        np.ceil(np.asarray(x_high_real, dtype=np.float64) * scale).astype(object),
    )


def _build_layer_specs(result: SynthesisResult) -> list[LayerQuantizationSpec]:
    return [
        LayerQuantizationSpec(
            total_bits=result.total_bits[index],
            integer_bits=result.integer_bits[index],
            fractional_bits=result.fractional_bits[index],
        )
        for index in range(len(result.total_bits))
    ]


def _specs_to_dicts(layer_specs: list[LayerQuantizationSpec]) -> list[dict[str, int]]:
    return [
        {
            "layer_index": int(index),
            "total_bits": int(spec.total_bits),
            "integer_bits": int(spec.integer_bits),
            "fractional_bits": int(spec.fractional_bits),
        }
        for index, spec in enumerate(layer_specs)
    ]


def _spec_to_dict(layer_index: int, spec: LayerQuantizationSpec) -> dict[str, int]:
    return {
        "layer_index": int(layer_index),
        "total_bits": int(spec.total_bits),
        "integer_bits": int(spec.integer_bits),
        "fractional_bits": int(spec.fractional_bits),
    }


def _result_from_specs(
    base_result: SynthesisResult,
    layer_specs: list[LayerQuantizationSpec],
    *,
    success: bool,
) -> SynthesisResult:
    return SynthesisResult(
        success=success,
        total_bits=[int(spec.total_bits) for spec in layer_specs],
        fractional_bits=[int(spec.fractional_bits) for spec in layer_specs],
        integer_bits=[int(spec.integer_bits) for spec in layer_specs],
        stats=base_result.stats,
    )


def _threshold_enabled(threshold: float | None) -> bool:
    return threshold is not None and threshold >= 0


def _quality_gate_enabled(config: RobustnessPipelineConfig) -> bool:
    threshold_enabled = (
        _threshold_enabled(config.accuracy_drop_threshold)
        or _threshold_enabled(config.mismatch_threshold)
        or (
            config.empirical_saturation_check
            and _threshold_enabled(config.saturation_threshold)
        )
    )
    return config.max_quality_refinement_steps > 0 and (
        config.formal_saturation_check or threshold_enabled
    )


def _quality_thresholds_payload(config: RobustnessPipelineConfig) -> dict[str, Any]:
    return {
        "accuracy_drop_threshold": config.accuracy_drop_threshold,
        "saturation_threshold": config.saturation_threshold,
        "mismatch_threshold": config.mismatch_threshold,
        "max_quality_refinement_steps": int(config.max_quality_refinement_steps),
        "formal_saturation_check": bool(config.formal_saturation_check),
        "empirical_saturation_check": bool(config.empirical_saturation_check),
        "esbmc_layer_block_size": int(config.esbmc_layer_block_size),
        "blockwise_fail_fast": bool(config.blockwise_fail_fast),
        "blockwise_run_all_blocks_on_failure": bool(config.blockwise_run_all_blocks_on_failure),
        "esbmc_jobs": int(config.esbmc_jobs),
        "esbmc_memlimit": str(config.esbmc_memlimit),
        "esbmc_profile": str(config.esbmc_profile),
        "esbmc_timeout_seconds": int(config.esbmc_timeout_seconds),
        "gurobi_threads": int(config.gurobi_threads),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_mismatches_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "label",
        "keras_pred",
        "python_qnn_pred",
        "c_qnn_pred",
        "max_abs_error",
        "keras_logits",
        "python_qnn_logits",
        "c_qnn_logits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _format_architecture_from_layers(layer_units: list[int]) -> str:
    hidden_units = layer_units[:-1]
    if hidden_units:
        return f"{len(hidden_units)}blk_" + "_".join(str(width) for width in hidden_units)
    return f"1blk_{layer_units[-1]}"


def _quality_failures(metrics: dict[str, Any], config: RobustnessPipelineConfig) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    accuracy_drop = float(metrics["keras_quantized_accuracy"] - metrics["python_qnn_accuracy"])
    if _threshold_enabled(config.accuracy_drop_threshold) and accuracy_drop > float(config.accuracy_drop_threshold):
        failures.append(
            {
                "kind": "accuracy_drop",
                "value": accuracy_drop,
                "threshold": float(config.accuracy_drop_threshold),
            }
        )

    mismatch_rate = float(metrics["python_qnn_mismatch_rate_vs_keras"])
    if _threshold_enabled(config.mismatch_threshold) and mismatch_rate > float(config.mismatch_threshold):
        failures.append(
            {
                "kind": "mismatch_rate",
                "value": mismatch_rate,
                "threshold": float(config.mismatch_threshold),
            }
        )

    if config.empirical_saturation_check and _threshold_enabled(config.saturation_threshold):
        for layer_diag in metrics.get("fixed_point_diagnostics", {}).get("python", {}).get("layers", []):
            saturation_rate = float(layer_diag.get("saturation_rate", 0.0))
            if saturation_rate > float(config.saturation_threshold):
                failures.append(
                    {
                        "kind": "saturation",
                        "layer_index": int(layer_diag["layer_index"]),
                        "value": saturation_rate,
                        "threshold": float(config.saturation_threshold),
                    }
                )

    return failures


def _highest_saturation_layer(metrics: dict[str, Any]) -> int | None:
    layers = metrics.get("fixed_point_diagnostics", {}).get("python", {}).get("layers", [])
    if not layers:
        return None
    return int(max(layers, key=lambda layer: float(layer.get("saturation_rate", 0.0)))["layer_index"])


def _refine_specs_after_failure(
    layer_specs: list[LayerQuantizationSpec],
    metrics: dict[str, Any],
    failures: list[dict[str, Any]],
    bit_ub: int,
) -> tuple[list[LayerQuantizationSpec] | None, dict[str, Any]]:
    next_specs = list(layer_specs)
    if any(failure["kind"] == "saturation" for failure in failures):
        layer_index = _highest_saturation_layer(metrics)
        if layer_index is None:
            return None, {"kind": "blocked", "reason": "saturation_failure_without_layer_diagnostics"}
        current = next_specs[layer_index]
        proposed = LayerQuantizationSpec(
            total_bits=current.total_bits + 1,
            integer_bits=current.integer_bits + 1,
            fractional_bits=current.fractional_bits,
        )
        action = {
            "kind": "increase_integer_bits",
            "layer_index": int(layer_index),
            "from": _spec_to_dict(layer_index, current),
            "to": _spec_to_dict(layer_index, proposed),
            "reason": "highest_saturation_rate",
        }
    else:
        layer_index = len(next_specs) - 1
        current = next_specs[layer_index]
        proposed = LayerQuantizationSpec(
            total_bits=current.total_bits + 1,
            integer_bits=current.integer_bits,
            fractional_bits=current.fractional_bits + 1,
        )
        action = {
            "kind": "increase_fractional_bits",
            "layer_index": int(layer_index),
            "from": _spec_to_dict(layer_index, current),
            "to": _spec_to_dict(layer_index, proposed),
            "reason": "layer_error_unavailable_default_output_layer",
        }

    if proposed.total_bits > bit_ub:
        action["kind"] = "blocked"
        action["reason"] = f"refined total_bits would exceed bit_ub={bit_ub}"
        return None, action

    next_specs[layer_index] = proposed
    return next_specs, action


def _refine_specs_after_formal_saturation_failure(
    layer_specs: list[LayerQuantizationSpec],
    esbmc_records: list[dict[str, Any]],
    bit_ub: int,
) -> tuple[list[LayerQuantizationSpec] | None, dict[str, Any]]:
    failed_record = next(
        (
            record
            for record in esbmc_records
            if record.get("no_saturation_status") not in {"VERIFIED", "DISABLED"}
        ),
        None,
    )
    if failed_record is None:
        return None, {"kind": "blocked", "reason": "formal_saturation_failure_without_layer_record"}

    layer_index = int(failed_record["layer_index"])
    next_specs = list(layer_specs)
    current = next_specs[layer_index]
    proposed = LayerQuantizationSpec(
        total_bits=current.total_bits + 1,
        integer_bits=current.integer_bits + 1,
        fractional_bits=current.fractional_bits,
    )
    action: dict[str, Any] = {
        "kind": "increase_integer_bits",
        "layer_index": int(layer_index),
        "from": _spec_to_dict(layer_index, current),
        "to": _spec_to_dict(layer_index, proposed),
        "reason": "formal_saturation_possible",
    }
    if proposed.total_bits > bit_ub:
        action["kind"] = "blocked"
        action["reason"] = f"refined total_bits would exceed bit_ub={bit_ub}"
        return None, action

    next_specs[layer_index] = proposed
    return next_specs, action


def _formal_saturation_failure(esbmc_records: list[dict[str, Any]]) -> bool:
    return any(record.get("failure_type") == "formal_saturation_possible" for record in esbmc_records)


def _formal_saturation_summary(
    config: RobustnessPipelineConfig,
    quality_summary: dict[str, Any],
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for step in reversed(quality_summary.get("steps", [])):
        step_layers = step.get("esbmc", {}).get("layers")
        if step_layers:
            layers = [
                {
                    "layer_index": int(layer.get("layer_index", index)),
                    "Q": layer.get("total_bits"),
                    "I": layer.get("integer_bits"),
                    "F": layer.get("fractional_bits"),
                    "contract_status": layer.get("contract_status", layer.get("status")),
                    "no_saturation_status": layer.get("no_saturation_status", "DISABLED"),
                    "blocks": layer.get("blocks", []),
                    "resource_control": layer.get("resource_control"),
                    "no_saturation_resource_control": layer.get("no_saturation_resource_control"),
                }
                for index, layer in enumerate(step_layers)
            ]
            break

    return {
        "enabled": bool(config.formal_saturation_check),
        "used_for_acceptance": bool(_quality_gate_enabled(config) and config.formal_saturation_check),
        "layers": layers,
    }


def compare_qnn_to_keras(
    *,
    dataset: DatasetBundle,
    quantized_model: Any,
    fixed_point_network: FixedPointNetwork,
    split: str,
    limit: int | None,
    output_dir: Path,
    compile_c_backend: bool,
    compiler: str,
    enable_diagnostics: bool = True,
    accuracy_drop_threshold: float | None = 0.05,
    saturation_threshold: float | None = 0.01,
    mismatch_threshold: float | None = 0.05,
    empirical_saturation_check: bool = True,
) -> dict[str, Any]:
    """Compare Python and generated-C QNN execution against the quantized Keras model."""

    features, labels = select_split(dataset, split)  # type: ignore[arg-type]
    if limit is not None and limit > 0:
        features = features[:limit]
        labels = labels[:limit]

    normalized_features = _normalize_features(features, dataset.input_scale)
    keras_logits = _predict_logits(quantized_model, features)
    if enable_diagnostics:
        python_qnn_logits, python_diagnostics = forward_fixed_point_batch_with_diagnostics(
            fixed_point_network,
            normalized_features,
        )
    else:
        python_qnn_logits = forward_fixed_point_batch(fixed_point_network, normalized_features)
        python_diagnostics = {"enabled": False, "samples": int(labels.shape[0]), "layers": []}

    c_qnn_logits: np.ndarray | None = None
    c_backend_status = "SKIPPED"
    c_source_path: Path | None = None
    c_shared_path: Path | None = None
    python_c_integer_comparison: dict[str, Any] | None = None
    if compile_c_backend:
        try:
            c_source_path = write_c_qnn_source(fixed_point_network, output_dir / "c_export" / "qnn_model.c")
            c_shared_path = compile_c_qnn_shared_library(
                c_source_path,
                output_dir / "c_export" / "qnn_model.so",
                compiler=compiler,
            )
            compiled_qnn = CompiledCQNN(fixed_point_network, c_shared_path)
            c_outputs = []
            for sample in normalized_features:
                output_int = compiled_qnn.forward(sample)
                c_outputs.append(output_int.astype(np.float64) / float(2 ** fixed_point_network.output_fractional_bits))
            c_qnn_logits = np.asarray(c_outputs, dtype=np.float64)
            python_c_integer_comparison = compare_python_c_fixed_point_outputs(
                fixed_point_network,
                normalized_features,
                compiled_qnn,
            )
            c_backend_status = "OK"
        except Exception as exc:
            c_backend_status = f"ERROR: {exc}"
            LOGGER.exception("Failed to compile or execute the generated C QNN.")

    keras_pred = np.argmax(keras_logits, axis=1)
    python_pred = np.argmax(python_qnn_logits, axis=1)
    c_pred = np.argmax(c_qnn_logits, axis=1) if c_qnn_logits is not None else None

    abs_error = np.abs(python_qnn_logits - keras_logits)
    mismatch_mask = python_pred != keras_pred
    keras_vs_python_mismatch_rate = float(np.mean(mismatch_mask))
    keras_vs_c_mismatch_rate = float(np.mean(c_pred != keras_pred)) if c_pred is not None else None
    keras_quantized_accuracy = _compute_accuracy(keras_logits, labels)
    python_qnn_accuracy = _compute_accuracy(python_qnn_logits, labels)
    c_qnn_accuracy = _compute_accuracy(c_qnn_logits, labels) if c_qnn_logits is not None else None
    warnings: list[str] = []
    if _threshold_enabled(accuracy_drop_threshold) and keras_quantized_accuracy - python_qnn_accuracy > float(
        accuracy_drop_threshold
    ):
        warnings.append(
            f"Fixed-point accuracy is more than {float(accuracy_drop_threshold):.2%} below quantized Keras accuracy; "
            "inspect saturation, scaling, and layer-wise diagnostics."
        )
    for layer_diag in python_diagnostics.get("layers", []):
        if (
            empirical_saturation_check
            and _threshold_enabled(saturation_threshold)
            and float(layer_diag.get("saturation_rate", 0.0)) > float(saturation_threshold)
        ):
            warnings.append(
                f"Layer {int(layer_diag['layer_index'])} has saturation_rate > {float(saturation_threshold):.2%}; "
                "consider increasing integer_bits for that layer."
            )
    if _threshold_enabled(mismatch_threshold) and keras_vs_python_mismatch_rate > float(mismatch_threshold):
        warnings.append(
            "Python fixed-point mismatch rate is above the configured threshold; "
            "inspect scaling and layer-wise diagnostics."
        )
    if python_c_integer_comparison is not None and not python_c_integer_comparison.get("exact_match", False):
        warnings.append(
            "Generated C fixed-point outputs differ from Python fixed-point outputs; "
            f"max integer difference is {python_c_integer_comparison.get('max_integer_difference')}."
        )
    mismatch_indices = np.flatnonzero(mismatch_mask)
    mismatch_rows: list[dict[str, Any]] = []
    for index in mismatch_indices[:25]:
        mismatch_rows.append(
            {
                "index": int(index),
                "label": int(labels[index]),
                "keras_pred": int(keras_pred[index]),
                "python_qnn_pred": int(python_pred[index]),
                "c_qnn_pred": int(c_pred[index]) if c_pred is not None else "",
                "max_abs_error": float(np.max(abs_error[index])),
                "keras_logits": keras_logits[index].tolist(),
                "python_qnn_logits": python_qnn_logits[index].tolist(),
                "c_qnn_logits": c_qnn_logits[index].tolist() if c_qnn_logits is not None else [],
            }
        )

    metrics = {
        "samples_evaluated": int(labels.shape[0]),
        "keras_quantized_accuracy": keras_quantized_accuracy,
        "python_qnn_accuracy": python_qnn_accuracy,
        "python_qnn_mismatch_rate_vs_keras": keras_vs_python_mismatch_rate,
        "python_qnn_max_abs_error": float(np.max(abs_error)),
        "python_qnn_mean_abs_error": float(np.mean(abs_error)),
        "python_qnn_max_abs_error_per_output": np.max(abs_error, axis=0).tolist(),
        "c_backend_status": c_backend_status,
        "c_qnn_accuracy": c_qnn_accuracy,
        "c_qnn_mismatch_rate_vs_keras": keras_vs_c_mismatch_rate,
        "python_c_integer_comparison": python_c_integer_comparison,
        "fixed_point_diagnostics": {
            "python": python_diagnostics,
        },
        "semantic_gap": {
            "keras_vs_python_mismatch_rate": keras_vs_python_mismatch_rate,
            "keras_vs_c_mismatch_rate": keras_vs_c_mismatch_rate,
            "max_abs_logit_error": float(np.max(abs_error)),
            "mean_abs_logit_error": float(np.mean(abs_error)),
        },
        "warnings": warnings,
        "artifacts": {
            "c_source": str(c_source_path) if c_source_path is not None else None,
            "c_shared_library": str(c_shared_path) if c_shared_path is not None else None,
        },
    }

    report_path = _save_json(output_dir / "reports" / "qnn_vs_keras_metrics.json", metrics)
    mismatches_path = _save_mismatches_csv(output_dir / "reports" / "qnn_vs_keras_mismatches.csv", mismatch_rows)
    metrics["artifacts"]["metrics_json"] = str(report_path)
    metrics["artifacts"]["mismatches_csv"] = str(mismatches_path)
    return metrics


def _compare_specs(
    *,
    dataset: DatasetBundle,
    model: Any,
    layer_specs: list[LayerQuantizationSpec],
    split: str,
    limit: int | None,
    output_dir: Path,
    compile_c_backend: bool,
    compiler: str,
    config: RobustnessPipelineConfig,
) -> tuple[dict[str, Any], Any, FixedPointNetwork]:
    quantized_model = clone_quantized_keras_model(model, layer_specs)
    fixed_point_network = build_fixed_point_network(model, layer_specs)
    comparison = compare_qnn_to_keras(
        dataset=dataset,
        quantized_model=quantized_model,
        fixed_point_network=fixed_point_network,
        split=split,
        limit=limit,
        output_dir=output_dir,
        compile_c_backend=compile_c_backend,
        compiler=compiler,
        enable_diagnostics=config.enable_diagnostics or _quality_gate_enabled(config),
        accuracy_drop_threshold=config.accuracy_drop_threshold,
        saturation_threshold=config.saturation_threshold,
        mismatch_threshold=config.mismatch_threshold,
        empirical_saturation_check=config.empirical_saturation_check,
    )
    return comparison, quantized_model, fixed_point_network


def _run_quality_refinement(
    *,
    dataset: DatasetBundle,
    model: Any,
    synthesizer: GPEncoding,
    initial_result: SynthesisResult,
    initial_specs: list[LayerQuantizationSpec],
    config: RobustnessPipelineConfig,
) -> tuple[list[LayerQuantizationSpec], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    del initial_result
    enabled = _quality_gate_enabled(config)
    attempts: list[dict[str, Any]] = []
    thresholds = _quality_thresholds_payload(config)
    current_specs = initial_specs
    accepted = False
    final_reason = "quality gate disabled"
    final_comparison: dict[str, Any] | None = None
    final_quantized_model: Any | None = None

    if not enabled:
        final_comparison, final_quantized_model, _ = _compare_specs(
            dataset=dataset,
            model=model,
            layer_specs=current_specs,
            split=config.compare_split,
            limit=config.compare_limit,
            output_dir=config.output_dir,
            compile_c_backend=config.compile_c_backend,
            compiler=config.compiler,
            config=config,
        )
        accepted = True
        attempts.append(
            {
                "step": 0,
                "configuration": _specs_to_dicts(current_specs),
                "quality_failures": [],
                "accepted": True,
                "esbmc": {
                    "required_for_acceptance": False,
                    "status": "NOT_RERUN_QUALITY_GATE_DISABLED",
                    "no_saturation_enabled": False,
                },
            }
        )
    else:
        for step in range(config.max_quality_refinement_steps + 1):
            attempt_dir = config.output_dir / "quality_refinement" / f"attempt_{step}"
            comparison, _, _ = _compare_specs(
                dataset=dataset,
                model=model,
                layer_specs=current_specs,
                split=config.compare_split,
                limit=config.compare_limit,
                output_dir=attempt_dir,
                compile_c_backend=False,
                compiler=config.compiler,
                config=config,
            )
            failures = _quality_failures(comparison, config)
            attempt: dict[str, Any] = {
                "step": int(step),
                "configuration": _specs_to_dicts(current_specs),
                "quality_failures": failures,
                "comparison": comparison,
                "accepted": False,
                "esbmc": {
                    "required_for_acceptance": True,
                    "no_saturation_enabled": bool(config.formal_saturation_check),
                    "status": "NOT_RUN_QUALITY_FAILED" if failures else "PENDING",
                },
            }
            attempts.append(attempt)

            if not failures:
                if step == 0 and config.verify_mode == "esbmc" and not config.formal_saturation_check:
                    esbmc_verified = True
                    esbmc_records = [
                        {
                            "layer_index": int(index),
                            "total_bits": int(spec.total_bits),
                            "integer_bits": int(spec.integer_bits),
                            "fractional_bits": int(spec.fractional_bits),
                            "status": "VERIFIED_BY_SYNTHESIS",
                            "contract_status": "VERIFIED_BY_SYNTHESIS",
                            "no_saturation_status": "DISABLED",
                        }
                        for index, spec in enumerate(current_specs)
                    ]
                else:
                    esbmc_verified, esbmc_records = synthesizer.verify_exported_quantization_with_esbmc(
                        total_bits=[spec.total_bits for spec in current_specs],
                        fractional_bits=[spec.fractional_bits for spec in current_specs],
                        integer_bits=[spec.integer_bits for spec in current_specs],
                        formal_saturation_check=config.formal_saturation_check,
                    )
                attempt["esbmc"] = {
                    "required_for_acceptance": True,
                    "no_saturation_enabled": bool(config.formal_saturation_check),
                    "status": "VERIFIED" if esbmc_verified else "FAILED",
                    "layers": esbmc_records,
                }
                formal_saturation_failed = _formal_saturation_failure(esbmc_records)
                if formal_saturation_failed:
                    attempt["esbmc"]["failure_type"] = "formal_saturation_possible"
                attempt["accepted"] = bool(esbmc_verified)
                if esbmc_verified:
                    accepted = True
                    final_reason = "accepted after ESBMC and deployment-quality checks"
                    break

                if (
                    config.formal_saturation_check
                    and formal_saturation_failed
                    and step < config.max_quality_refinement_steps
                ):
                    refined_specs, action = _refine_specs_after_formal_saturation_failure(
                        current_specs,
                        esbmc_records,
                        config.bit_ub,
                    )
                    attempt["refinement_action"] = action
                    if refined_specs is None:
                        final_reason = str(action.get("reason", "formal saturation refinement blocked"))
                        break
                    current_specs = refined_specs
                    final_reason = "refining formal no-saturation failure"
                    continue

                if formal_saturation_failed:
                    final_reason = "formal no-saturation failed and max refinement steps was reached"
                else:
                    final_reason = "quality checks passed but ESBMC verification failed"
                break

            if step >= config.max_quality_refinement_steps:
                final_reason = "quality checks failed and max refinement steps was reached"
                break

            refined_specs, action = _refine_specs_after_failure(current_specs, comparison, failures, config.bit_ub)
            attempt["refinement_action"] = action
            if refined_specs is None:
                final_reason = str(action.get("reason", "quality refinement blocked"))
                break
            current_specs = refined_specs
            final_reason = "refining rejected fixed-point configuration"

        final_comparison, final_quantized_model, _ = _compare_specs(
            dataset=dataset,
            model=model,
            layer_specs=current_specs,
            split=config.compare_split,
            limit=config.compare_limit,
            output_dir=config.output_dir,
            compile_c_backend=config.compile_c_backend,
            compiler=config.compiler,
            config=config,
        )

    history = {
        "enabled": bool(enabled),
        "accepted": bool(accepted),
        "thresholds": thresholds,
        "initial_configuration": _specs_to_dicts(initial_specs),
        "final_configuration": _specs_to_dicts(current_specs),
        "attempts": attempts,
        "final_reason": final_reason,
    }
    history_path = config.output_dir / "reports" / "refinement_history.json"
    history["artifacts"] = {"refinement_history": str(history_path)}
    _save_json(history_path, history)
    quality_summary = {
        "enabled": bool(enabled),
        "accepted": bool(accepted),
        "steps": attempts,
        "final_reason": final_reason,
        "artifacts": history["artifacts"],
    }

    if final_comparison is None or final_quantized_model is None:
        raise RuntimeError("Quality refinement did not produce a final comparison.")
    return current_specs, final_comparison, final_quantized_model, quality_summary, history


def run_robustness_pipeline(repo_root: Path, config: RobustnessPipelineConfig) -> dict[str, Any]:
    """Run the full robustness synthesis pipeline and return a structured summary."""

    selection = normalize_dataset_selection(config.dataset)
    dataset = load_dataset(selection.base_name)
    weights_path = resolve_weight_path(repo_root, config.dataset, config.arch)
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not resolve benchmark weights at {weights_path}")

    inferred_arch = infer_dense_architecture_from_h5(weights_path)
    input_dim = dataset.input_dim
    num_classes = dataset.num_classes
    layer_units = parse_architecture(config.arch, num_classes)
    if inferred_arch:
        input_dim = inferred_arch[0]
        layer_units = inferred_arch[1:]
        num_classes = layer_units[-1]
    cache_arch = _format_architecture_from_layers(layer_units) if selection.benchmark_name is not None else config.arch

    model = build_and_load_deep_model(
        input_dim=input_dim,
        layer_units=layer_units,
        weights_path=weights_path,
        input_scale=dataset.input_scale,
    )

    sample = dataset.x_test[config.sample_id]
    sample_label = int(dataset.y_test[config.sample_id])
    sample_logits = _predict_logits(model, np.expand_dims(sample, axis=0))[0]
    predicted_label = int(np.argmax(sample_logits))
    if predicted_label != sample_label:
        raise ValueError(
            f"Selected sample {config.sample_id} is misclassified by the reference model "
            f"(pred={predicted_label}, label={sample_label})."
        )

    x_low = np.clip(sample - config.eps, dataset.clip_low, dataset.clip_high)
    x_high = np.clip(sample + config.eps, dataset.clip_low, dataset.clip_high)
    baseline_logits = _predict_logits(model, dataset.x_test)
    full_precision_accuracy = _compute_accuracy(baseline_logits, dataset.y_test)

    property_spec = ClassificationProperty(
        target_label=config.target_label if config.target_label is not None else predicted_label,
        valid_labels=config.valid_labels,
    )
    property_spec.validate(num_classes)
    preimage_cache_key = config.preimage_cache_key
    preimage_cache_metadata: dict[str, Any] | None = None
    if config.no_gurobi or config.save_preimage_cache:
        derived_key, preimage_cache_metadata = build_preimage_cache_identity(
            dataset=config.dataset,
            arch=cache_arch,
            sample_id=config.sample_id,
            eps=config.eps,
            preimg_mode=config.preimg_mode,
            if_relax=config.if_relax,
            target_label=int(property_spec.target_label if property_spec.target_label is not None else predicted_label),
            valid_labels=property_spec.valid_labels,
            weights_path=weights_path,
        )
        preimage_cache_key = preimage_cache_key or derived_key

    synth_config = QuadapterConfig(
        bit_lb=config.bit_lb,
        bit_ub=config.bit_ub,
        preimg_mode=config.preimg_mode,
        verify_mode=config.verify_mode,
        sample_id=config.sample_id,
        eps=config.eps,
        output_dir=config.output_dir,
        if_relax=config.if_relax,
        no_gurobi=config.no_gurobi,
        save_preimage_cache=config.save_preimage_cache,
        preimage_cache_dir=config.preimage_cache_dir,
        preimage_cache_key=preimage_cache_key,
        preimage_cache_metadata=preimage_cache_metadata,
        esbmc_layer_block_size=max(0, int(config.esbmc_layer_block_size)),
        blockwise_fail_fast=bool(config.blockwise_fail_fast),
        blockwise_run_all_blocks_on_failure=bool(config.blockwise_run_all_blocks_on_failure),
        esbmc_jobs=max(1, int(config.esbmc_jobs)),
        gurobi_threads=max(1, int(config.gurobi_threads)),
        esbmc=ESBMCConfig(
            timeout_seconds=max(1, int(config.esbmc_timeout_seconds)),
            memlimit=str(config.esbmc_memlimit),
            default_profile=config.esbmc_profile,
        ),
    )
    synthesizer = GPEncoding(
        arch=[input_dim] + layer_units,
        model=model,
        config=synth_config,
        original_prediction=predicted_label,
        x_low_real=np.asarray(x_low / dataset.input_scale, dtype=np.float32),
        x_high_real=np.asarray(x_high / dataset.input_scale, dtype=np.float32),
        property_spec=property_spec,
    )

    forward_dnn(np.asarray(sample / dataset.input_scale, dtype=np.float32), synthesizer)
    synthesis_result = synthesizer.run(
        np.asarray(x_low / dataset.input_scale, dtype=np.float32),
        np.asarray(x_high / dataset.input_scale, dtype=np.float32),
    )

    summary: dict[str, Any] = {
        "dataset": config.dataset,
        "base_dataset": selection.base_name,
        "arch": config.arch,
        "weights_path": str(weights_path),
        "sample_id": config.sample_id,
        "eps": config.eps,
        "compare_split": config.compare_split,
        "sample_label": sample_label,
        "predicted_label": predicted_label,
        "sample_logits": sample_logits.tolist(),
        "synthesis": synthesis_result.to_dict(),
        "baseline": {
            "reference_accuracy": full_precision_accuracy,
        },
        "formal_saturation_check_enabled": bool(config.formal_saturation_check),
        "empirical_saturation_check_enabled": bool(config.empirical_saturation_check),
        "formal_saturation_verification": {
            "enabled": bool(config.formal_saturation_check),
            "used_for_acceptance": False,
            "layers": [],
        },
        "blockwise_verification": synthesizer.blockwise_verification_summary(),
        "fixed_point_semantics": {
            "claim_type": "declared_backend_semantics",
            "layers": [],
        },
        "accumulator_range": [],
        "verification_claims": {
            "fixed_point_semantics": "declared_backend_semantics",
            "accumulator_range": "static_interval_analysis",
            "deployment_metrics": "empirical_dataset_evaluation",
            "formal_saturation_verification": "formal_esbmc_when_enabled",
            "blockwise_verification": "equivalent_hidden_contract_decomposition_when_enabled",
        },
    }
    if preimage_cache_key:
        summary["preimage_cache"] = {
            "key": preimage_cache_key,
            "dir": str(config.preimage_cache_dir or (config.output_dir / "preimage_cache")),
            "mode": "load" if config.no_gurobi else "save" if config.save_preimage_cache else "unused",
        }

    if not synthesis_result.success:
        summary["blockwise_verification"] = synthesizer.blockwise_verification_summary()
        quant_config_path = _save_json(config.output_dir / "reports" / "quantization_config.json", summary)
        summary["artifacts"] = {"quantization_config": str(quant_config_path)}
        pipeline_summary_path = _save_json(config.output_dir / "reports" / "pipeline_summary.json", summary)
        external_baselines = load_external_baselines(config.baseline_results_json)
        experiment_artifacts = {
            "pipeline_summary": str(pipeline_summary_path),
            "qnn_vs_keras_metrics": None,
            "mismatches_csv": None,
            "refinement_history": None,
            "c_source": None,
            "c_shared_library": None,
        }
        experiment_summary = build_experiment_summary(
            pipeline_summary=summary,
            formal_metrics=None,
            refined_metrics=None,
            formal_resource_metrics=None,
            refined_resource_metrics=None,
            external_baselines=external_baselines,
            artifacts=experiment_artifacts,
        )
        experiment_summary_path = _save_json(config.output_dir / "reports" / "experiment_summary.json", experiment_summary)
        summary["artifacts"]["pipeline_summary"] = str(pipeline_summary_path)
        summary["artifacts"]["experiment_summary"] = str(experiment_summary_path)
        if config.export_paper_tables:
            table_artifacts = export_paper_tables(experiment_summary, config.output_dir)
            summary["artifacts"].update(table_artifacts)
            experiment_summary["artifacts"].update(table_artifacts)
            _save_json(experiment_summary_path, experiment_summary)
        _save_json(pipeline_summary_path, summary)
        return summary

    layer_specs = _build_layer_specs(synthesis_result)
    final_specs, comparison, quantized_model, quality_summary, refinement_history = _run_quality_refinement(
        dataset=dataset,
        model=model,
        synthesizer=synthesizer,
        initial_result=synthesis_result,
        initial_specs=layer_specs,
        config=config,
    )
    final_synthesis_result = _result_from_specs(
        synthesis_result,
        final_specs,
        success=bool(synthesis_result.success and quality_summary["accepted"]),
    )
    if final_specs != layer_specs or _quality_gate_enabled(config):
        summary["formal_synthesis"] = synthesis_result.to_dict()
    summary["synthesis"] = final_synthesis_result.to_dict()
    summary["quality_refinement"] = quality_summary
    summary["formal_saturation_verification"] = _formal_saturation_summary(config, quality_summary)
    summary["blockwise_verification"] = synthesizer.blockwise_verification_summary()
    summary["comparison"] = comparison

    quantized_logits = _predict_logits(quantized_model, dataset.x_test)
    summary["baseline"]["quantized_keras_accuracy"] = _compute_accuracy(quantized_logits, dataset.y_test)

    formal_metrics = (
        refinement_history.get("attempts", [{}])[0].get("comparison")
        if refinement_history.get("attempts")
        else comparison
    )
    final_network = build_fixed_point_network(model, final_specs)
    formal_network = build_fixed_point_network(model, layer_specs)
    formal_fixed_point_semantics = fixed_point_semantics_for_network(formal_network)
    refined_fixed_point_semantics = fixed_point_semantics_for_network(final_network)
    x_low_real = np.asarray(x_low / dataset.input_scale, dtype=np.float64)
    x_high_real = np.asarray(x_high / dataset.input_scale, dtype=np.float64)
    formal_accumulator_range = compute_accumulator_range_analysis(
        formal_network,
        input_bounds=_fixed_point_input_bounds(formal_network, x_low_real, x_high_real),
    )
    refined_accumulator_range = compute_accumulator_range_analysis(
        final_network,
        input_bounds=_fixed_point_input_bounds(final_network, x_low_real, x_high_real),
    )
    summary["fixed_point_semantics"] = refined_fixed_point_semantics
    summary["accumulator_range"] = refined_accumulator_range
    summary["fixed_point_semantics_by_method"] = {
        "formal_only": formal_fixed_point_semantics,
        "quality_refined": refined_fixed_point_semantics,
    }
    summary["accumulator_range_by_method"] = {
        "formal_only": formal_accumulator_range,
        "quality_refined": refined_accumulator_range,
    }
    final_artifacts = comparison.get("artifacts", {})
    formal_artifacts = formal_metrics.get("artifacts", {}) if isinstance(formal_metrics, dict) else {}
    formal_resource_metrics = compute_fixed_point_resource_metrics(
        formal_network,
        c_source_path=formal_artifacts.get("c_source"),
        c_shared_library_path=formal_artifacts.get("c_shared_library"),
    )
    refined_resource_metrics = compute_fixed_point_resource_metrics(
        final_network,
        c_source_path=final_artifacts.get("c_source"),
        c_shared_library_path=final_artifacts.get("c_shared_library"),
    )

    quant_config_path = _save_json(config.output_dir / "reports" / "quantization_config.json", summary)
    summary["artifacts"] = {"quantization_config": str(quant_config_path)}
    pipeline_summary_path = _save_json(config.output_dir / "reports" / "pipeline_summary.json", summary)
    external_baselines = load_external_baselines(config.baseline_results_json)
    experiment_artifacts = {
        "pipeline_summary": str(pipeline_summary_path),
        "qnn_vs_keras_metrics": final_artifacts.get("metrics_json"),
        "mismatches_csv": final_artifacts.get("mismatches_csv"),
        "refinement_history": refinement_history.get("artifacts", {}).get("refinement_history"),
        "c_source": final_artifacts.get("c_source"),
        "c_shared_library": final_artifacts.get("c_shared_library"),
    }
    experiment_summary = build_experiment_summary(
        pipeline_summary=summary,
        formal_metrics=formal_metrics,
        refined_metrics=comparison,
        formal_resource_metrics=formal_resource_metrics,
        refined_resource_metrics=refined_resource_metrics,
        external_baselines=external_baselines,
        artifacts=experiment_artifacts,
    )
    experiment_summary_path = _save_json(config.output_dir / "reports" / "experiment_summary.json", experiment_summary)
    summary["artifacts"].update(
        {
            "pipeline_summary": str(pipeline_summary_path),
            "experiment_summary": str(experiment_summary_path),
        }
    )
    _save_json(pipeline_summary_path, summary)
    if config.export_paper_tables:
        table_artifacts = export_paper_tables(experiment_summary, config.output_dir)
        summary["artifacts"].update(table_artifacts)
        experiment_summary["artifacts"].update(table_artifacts)
        _save_json(experiment_summary_path, experiment_summary)
        _save_json(pipeline_summary_path, summary)
    return summary
