"""One-shot held-out test evaluation of a selected deep source model.

The search in ``search_ssv_deep_source_model`` selects on validation only and
records ``test_status: NOT_EVALUATED``. This script consumes the test split
exactly once, after selection, and refuses to overwrite an existing report so
that the number cannot be quietly re-rolled.

Nothing here is a proof. Every field is a measurement on the float32 source
model and on its folded affine lowering; quantization and ESBMC remain
untouched and are reported as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from scripts.run_ssv_cnn_gate import write_new_json

PARITY_REFERENCE_SAMPLE = 64


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected(search_output: Path):
    summary = json.loads((search_output / "search_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "SELECTED":
        raise ValueError(f"Search did not select a candidate: status={summary.get('status')}")
    if summary.get("test_status") != "NOT_EVALUATED":
        raise ValueError(f"Test split already consumed: test_status={summary.get('test_status')}")
    selected = summary["selected_candidate_summary"]
    params = Path(selected["folded_model_path"])
    if _sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Folded model digest does not match the search summary")
    return summary, selected, params


def restricted_from_npz(selected, params: Path) -> RestrictedSequentialCNN:
    arrays = np.load(params)
    geometries = tuple(
        ConvGeometry(input_shape=tuple(item["input_shape"]), kernel_shape=tuple(item["kernel_shape"]),
                     strides=tuple(item["strides"]), padding=item["padding"])
        for item in selected["geometries"]
    )
    conv = [(arrays[f"conv_{i}_kernel"], arrays[f"conv_{i}_bias"]) for i in range(len(geometries))]
    dense_count = len(selected["dense_hidden"]) + 1
    dense = [(arrays[f"dense_{i}_kernel"], arrays[f"dense_{i}_bias"]) for i in range(dense_count)]
    return RestrictedSequentialCNN(
        geometries=geometries,
        conv_kernels=tuple(kernel for kernel, _ in conv),
        conv_biases=tuple(bias for _, bias in conv),
        dense_kernels=tuple(kernel for kernel, _ in dense),
        dense_biases=tuple(bias for _, bias in dense),
        max_affine_entries=max(selected["affine_entries_per_layer"]),
    )


def folded_logits(tf, restricted: RestrictedSequentialCNN, images, batch_size):
    """Batched float32 evaluation of the folded affine/ReLU model.

    Uses the convolution kernels directly rather than the dense Toeplitz
    lowering, which for this model would be 97.6% zeros. Tied back to the
    audited Python reference by ``reference_parity``.
    """
    outputs = []
    for start in range(0, len(images), batch_size):
        value = tf.convert_to_tensor(images[start:start + batch_size], dtype=tf.float32)
        for geometry, kernel, bias in zip(restricted.geometries, restricted.conv_kernels,
                                          restricted.conv_biases):
            value = tf.nn.conv2d(value, kernel, strides=(1, *geometry.strides, 1),
                                 padding=geometry.padding)
            value = tf.nn.relu(value + bias)
        value = tf.reshape(value, (value.shape[0], -1))
        for index, (kernel, bias) in enumerate(zip(restricted.dense_kernels,
                                                   restricted.dense_biases)):
            value = value @ kernel + bias
            if index + 1 < len(restricted.dense_kernels):
                value = tf.nn.relu(value)
        outputs.append(np.asarray(value))
    return np.concatenate(outputs, axis=0)


def reference_parity(restricted, images, batched, indices):
    """Check the batched path against the independent sliding-window reference."""
    worst = 0.0
    for index in indices:
        expected = restricted.float_reference(images[index].astype(np.float32))
        worst = max(worst, float(np.max(np.abs(expected - batched[index]))))
    return worst


def balanced_accuracy(labels, predictions, classes):
    per_class = []
    for class_id in range(classes):
        mask = labels == class_id
        if np.any(mask):
            per_class.append(float(np.mean(predictions[mask] == class_id)))
    return float(np.mean(per_class)), per_class


def evaluate(search_output: Path, output: Path, batch_size: int = 128):
    search_output = Path(search_output).resolve()
    output = Path(output).resolve()
    report_path = output / "test_evaluation.json"
    if report_path.exists():
        # Refuse before importing a framework or decoding a single test image.
        raise FileExistsError(f"Test split already consumed at {report_path}")
    summary, selected, params = load_selected(search_output)

    import tensorflow as tf


    base_path = Path(summary["base_study_path"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if _sha256(base_path) != summary["base_study_sha256"]:
        raise ValueError("Frozen base study changed since the search ran")
    config = json.loads(Path(summary["config_path"]).read_text(encoding="utf-8"))

    rows = [row for row in base["records"] if row.get("split") == "test"]
    if not rows:
        raise ValueError("Frozen study has no test rows")
    geometry = ConvGeometry(**{key: tuple(value) if isinstance(value, list) else value
                               for key, value in selected["geometries"][0].items()})
    encoder = ByteImageEncoder(geometry.input_shape,
                               resize_mode=config["preprocessing"]["resize"])
    root = Path(base["dataset_root"])
    images = np.stack([encoder.resize(load_crop(root, row)) for row in rows])
    images = images.astype(np.float32) / np.float32(256)
    labels = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
    restricted = restricted_from_npz(selected, params)
    classes = len(restricted.dense_biases[-1])

    source = tf.keras.models.load_model(params.parent / "source_with_batch_norm.keras")
    source_logits = np.asarray(source.predict(images, batch_size=batch_size, verbose=0))
    source_predictions = np.argmax(source_logits, axis=1)

    lowered_logits = folded_logits(tf, restricted, images, batch_size)
    lowered_predictions = np.argmax(lowered_logits, axis=1)

    rng = np.random.default_rng(2026)
    sample = sorted(rng.choice(len(images), size=min(PARITY_REFERENCE_SAMPLE, len(images)),
                               replace=False).tolist())
    reference_error = reference_parity(restricted, images, lowered_logits, sample)

    source_accuracy = float(np.mean(source_predictions == labels))
    lowered_accuracy = float(np.mean(lowered_predictions == labels))
    source_balanced, _ = balanced_accuracy(labels, source_predictions, classes)
    lowered_balanced, per_class = balanced_accuracy(labels, lowered_predictions, classes)

    report = {
        "schema": "ssv_deep_source_model_test_evaluation_v1",
        "status": "COMPLETED",
        "test_set_policy": "evaluate_once_after_selection",
        "test_set_consumed": True,
        "search_output": str(search_output),
        "candidate_id": selected["candidate_id"],
        "folded_model_sha256": selected["folded_model_sha256"],
        "base_study_sha256": summary["base_study_sha256"],
        "n_test_images": len(images),
        "classes": classes,
        "validation_accuracy_reported_by_search": selected["validation_accuracy"],
        "test_accuracy_source_float32": source_accuracy,
        "test_accuracy_folded_affine": lowered_accuracy,
        "test_balanced_accuracy_source_float32": source_balanced,
        "test_balanced_accuracy_folded_affine": lowered_balanced,
        "generalization_gap_percentage_points":
            100 * (selected["validation_accuracy"] - lowered_accuracy),
        "worst_class_accuracy": min(per_class),
        "worst_class_id": int(np.argmin(per_class)),
        "per_class_accuracy_folded_affine": per_class,
        "folding_max_abs_logit_error": float(np.max(np.abs(source_logits - lowered_logits))),
        "folding_predictions_equal":
            bool(np.array_equal(source_predictions, lowered_predictions)),
        "folding_prediction_mismatches":
            int(np.sum(source_predictions != lowered_predictions)),
        "batched_vs_reference_sample_size": len(sample),
        "batched_vs_reference_max_abs_error": reference_error,
        "float_lowering_claim": "tested_numerical_equivalence_not_IEEE_proof",
        "quantization_status": "NOT_RUN",
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_new_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = evaluate(args.search_output, args.output, args.batch_size)
    print(json.dumps({key: report[key] for key in (
        "candidate_id", "n_test_images", "validation_accuracy_reported_by_search",
        "test_accuracy_source_float32", "test_accuracy_folded_affine",
        "test_balanced_accuracy_folded_affine", "generalization_gap_percentage_points",
        "worst_class_accuracy", "folding_predictions_equal",
        "batched_vs_reference_max_abs_error", "quantization_status", "proof_status",
    )}, indent=2))


if __name__ == "__main__":
    main()
