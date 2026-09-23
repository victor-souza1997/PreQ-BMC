"""Measure compact fixed-point convolution quality before a formal campaign."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from backends.c_qnn_generator import compile_c_qnn_shared_library
from backends.conv_fixed_point import (
    forward_conv_fixed_point_single,
    generate_conv_qnn_source,
    quantize_restricted_sequential,
)
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from scripts.evaluate_ssv_deep_source_model import restricted_from_npz
from scripts.run_ssv_cnn_gate import write_new_json


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(config_path: Path, output: Path, *, split="test", batch_size=128):
    config_path = Path(config_path).resolve()
    config = _json(config_path)
    if config.get("schema") != "ssv_conv_native_pilot_v1":
        raise ValueError("Expected the frozen convolution-native pilot configuration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository = config_path.parents[1]
    search_root = repository / config["source_search_output"]
    search = _json(search_root / "search_summary.json")
    selected = search["selected_candidate_summary"]
    params = Path(selected["folded_model_path"])
    if sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Folded model identity changed")
    base = _json(search["base_study_path"])
    if sha256(search["base_study_path"]) != search["base_study_sha256"]:
        raise ValueError("Frozen dataset study changed")
    rows = [row for row in base["records"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No frozen {split} records")
    specs = [LayerQuantizationSpec(**row) for row in config["qif"]]
    model = restricted_from_npz(selected, params)
    input_format = config["input_format"]
    encoder = ByteImageEncoder(
        model.input_shape,
        fractional_bits=input_format["fractional_bits"],
        total_bits=input_format["total_bits"],
        resize_mode=config["resize_mode"],
    )
    network = quantize_restricted_sequential(
        model, specs,
        input_fractional_bits=encoder.fractional_bits,
        input_total_bits=encoder.total_bits,
    )
    source_path = output / "qnn_conv_native.c"
    source_path.write_text(generate_conv_qnn_source(network, encoder_source=encoder.render_c()), encoding="utf-8")
    library_path = compile_c_qnn_shared_library(source_path, output / "qnn_conv_native.so")
    library = ctypes.CDLL(str(library_path.resolve()))
    pointer = ctypes.POINTER(ctypes.c_int64)
    library.qnn_forward_fixed.argtypes = [pointer, pointer]

    import tensorflow as tf

    source_model = tf.keras.models.load_model(params.parent / "source_with_batch_norm.keras")
    labels = []
    float_predictions = []
    fixed_predictions = []
    max_logit_error = 0.0
    sum_logit_error = 0.0
    logit_values = 0
    parity_indices = set(np.linspace(0, len(rows) - 1, min(16, len(rows)), dtype=int).tolist())
    parity_mismatches = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        resized = [encoder.resize(load_crop(base["dataset_root"], row)) for row in batch_rows]
        images = np.stack(resized)
        normalized = images.astype(np.float32) / np.float32(256)
        source_logits = np.asarray(source_model(normalized, training=False))
        for local_index, (row, image, float_logits) in enumerate(zip(
            batch_rows, images, source_logits, strict=True
        )):
            encoded = encoder.encode(image)
            fixed = np.zeros(network.output_size, dtype=np.int64)
            library.qnn_forward_fixed(
                encoded.ctypes.data_as(pointer), fixed.ctypes.data_as(pointer)
            )
            real_fixed = fixed.astype(np.float64) / (1 << network.output_fractional_bits)
            error = np.abs(real_fixed - float_logits.astype(np.float64))
            max_logit_error = max(max_logit_error, float(np.max(error)))
            sum_logit_error += float(np.sum(error))
            logit_values += len(error)
            labels.append(int(row["class_id"]))
            float_predictions.append(int(np.argmax(float_logits)))
            fixed_predictions.append(int(np.argmax(fixed)))
            global_index = start + local_index
            if global_index in parity_indices:
                python_fixed = forward_conv_fixed_point_single(network, encoded)
                parity_mismatches += int(not np.array_equal(python_fixed, fixed))
    labels = np.asarray(labels)
    float_predictions = np.asarray(float_predictions)
    fixed_predictions = np.asarray(fixed_predictions)
    report = {
        "schema": "ssv_conv_native_quantization_quality_v1",
        "status": "COMPLETED",
        "measurement_scope": split,
        "n_images": len(rows),
        "candidate_id": selected["candidate_id"],
        "folded_model_sha256": selected["folded_model_sha256"],
        "qif": [asdict(spec) for spec in specs],
        "source_float32_accuracy": float(np.mean(float_predictions == labels)),
        "conv_native_fixed_accuracy": float(np.mean(fixed_predictions == labels)),
        "accuracy_drop_percentage_points": float(100 * (
            np.mean(float_predictions == labels) - np.mean(fixed_predictions == labels)
        )),
        "prediction_mismatch_rate_vs_float": float(np.mean(
            fixed_predictions != float_predictions
        )),
        "max_abs_logit_error": max_logit_error,
        "mean_abs_logit_error": sum_logit_error / logit_values,
        "sampled_python_c_parity_images": len(parity_indices),
        "sampled_python_c_mismatches": parity_mismatches,
        "sampled_python_c_exact": parity_mismatches == 0,
        "generated_source_sha256": sha256(source_path),
        "compiled_library_sha256": sha256(library_path),
        "quality_claim": "empirical_not_formal",
        "formal_status_modified": False,
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
    }
    write_new_json(output / "quantization_quality.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = evaluate(args.config, args.output, split=args.split, batch_size=args.batch_size)
    print(json.dumps({key: report[key] for key in (
        "n_images", "source_float32_accuracy", "conv_native_fixed_accuracy",
        "accuracy_drop_percentage_points", "prediction_mismatch_rate_vs_float",
        "sampled_python_c_exact", "proof_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
