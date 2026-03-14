from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from backends.c_qnn_generator import CompiledCQNN, compile_c_qnn_shared_library, write_c_qnn_source
from backends.fixed_point import (
    FixedPointNetwork,
    LayerQuantizationSpec,
    build_fixed_point_network,
    clone_quantized_keras_model,
    forward_fixed_point_batch,
)
from datasets.loaders import DatasetBundle, load_dataset, select_split
from models.loading import (
    build_and_load_deep_model,
    infer_dense_architecture_from_h5,
    parse_architecture,
    resolve_weight_path,
)
from synthesis.forward import forward_dnn
from synthesis.quadapter import GPEncoding, QuadapterConfig, SynthesisResult
from utils.logging_utils import get_logger
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


def _build_layer_specs(result: SynthesisResult) -> list[LayerQuantizationSpec]:
    return [
        LayerQuantizationSpec(
            total_bits=result.total_bits[index],
            integer_bits=result.integer_bits[index],
            fractional_bits=result.fractional_bits[index],
        )
        for index in range(len(result.total_bits))
    ]


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
) -> dict[str, Any]:
    """Compare Python and generated-C QNN execution against the quantized Keras model."""

    features, labels = select_split(dataset, split)  # type: ignore[arg-type]
    if limit is not None and limit > 0:
        features = features[:limit]
        labels = labels[:limit]

    keras_logits = _predict_logits(quantized_model, features)
    python_qnn_logits = forward_fixed_point_batch(
        fixed_point_network,
        _normalize_features(features, dataset.input_scale),
    )

    c_qnn_logits: np.ndarray | None = None
    c_backend_status = "SKIPPED"
    c_source_path: Path | None = None
    c_shared_path: Path | None = None
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
            normalized_features = _normalize_features(features, dataset.input_scale)
            for sample in normalized_features:
                output_int = compiled_qnn.forward(sample)
                c_outputs.append(output_int.astype(np.float64) / float(2 ** fixed_point_network.output_fractional_bits))
            c_qnn_logits = np.asarray(c_outputs, dtype=np.float64)
            c_backend_status = "OK"
        except Exception as exc:
            c_backend_status = f"ERROR: {exc}"
            LOGGER.exception("Failed to compile or execute the generated C QNN.")

    keras_pred = np.argmax(keras_logits, axis=1)
    python_pred = np.argmax(python_qnn_logits, axis=1)
    c_pred = np.argmax(c_qnn_logits, axis=1) if c_qnn_logits is not None else None

    abs_error = np.abs(python_qnn_logits - keras_logits)
    mismatch_mask = python_pred != keras_pred
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
        "keras_quantized_accuracy": _compute_accuracy(keras_logits, labels),
        "python_qnn_accuracy": _compute_accuracy(python_qnn_logits, labels),
        "python_qnn_mismatch_rate_vs_keras": float(np.mean(mismatch_mask)),
        "python_qnn_max_abs_error": float(np.max(abs_error)),
        "python_qnn_mean_abs_error": float(np.mean(abs_error)),
        "python_qnn_max_abs_error_per_output": np.max(abs_error, axis=0).tolist(),
        "c_backend_status": c_backend_status,
        "c_qnn_accuracy": _compute_accuracy(c_qnn_logits, labels) if c_qnn_logits is not None else None,
        "c_qnn_mismatch_rate_vs_keras": float(np.mean(c_pred != keras_pred)) if c_pred is not None else None,
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


def run_robustness_pipeline(repo_root: Path, config: RobustnessPipelineConfig) -> dict[str, Any]:
    """Run the full robustness synthesis pipeline and return a structured summary."""

    dataset = load_dataset(config.dataset)  # type: ignore[arg-type]
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

    property_spec = ClassificationProperty(
        target_label=config.target_label if config.target_label is not None else predicted_label,
        valid_labels=config.valid_labels,
    )
    property_spec.validate(num_classes)

    synth_config = QuadapterConfig(
        bit_lb=config.bit_lb,
        bit_ub=config.bit_ub,
        preimg_mode=config.preimg_mode,
        verify_mode=config.verify_mode,
        sample_id=config.sample_id,
        eps=config.eps,
        output_dir=config.output_dir,
        if_relax=config.if_relax,
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
        "arch": config.arch,
        "weights_path": str(weights_path),
        "sample_id": config.sample_id,
        "sample_label": sample_label,
        "predicted_label": predicted_label,
        "sample_logits": sample_logits.tolist(),
        "synthesis": synthesis_result.to_dict(),
    }

    quant_config_path = _save_json(config.output_dir / "reports" / "quantization_config.json", summary)
    summary["artifacts"] = {"quantization_config": str(quant_config_path)}

    if not synthesis_result.success:
        return summary

    layer_specs = _build_layer_specs(synthesis_result)
    quantized_model = clone_quantized_keras_model(model, layer_specs)
    fixed_point_network = build_fixed_point_network(model, layer_specs)
    comparison = compare_qnn_to_keras(
        dataset=dataset,
        quantized_model=quantized_model,
        fixed_point_network=fixed_point_network,
        split=config.compare_split,
        limit=config.compare_limit,
        output_dir=config.output_dir,
        compile_c_backend=config.compile_c_backend,
        compiler=config.compiler,
    )
    summary["comparison"] = comparison

    baseline_logits = _predict_logits(model, dataset.x_test)
    quantized_logits = _predict_logits(quantized_model, dataset.x_test)
    summary["baseline"] = {
        "reference_accuracy": _compute_accuracy(baseline_logits, dataset.y_test),
        "quantized_keras_accuracy": _compute_accuracy(quantized_logits, dataset.y_test),
    }
    _save_json(config.output_dir / "reports" / "pipeline_summary.json", summary)
    return summary
