from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from utils.data.iris import load_train_test_data
from utils.data.load_onnx import extract_gemm_params
from utils.data.onnx2keras import build_keras_model_from_onnx
from utils.deep_models import DeepModel
import logging

logging.basicConfig(filename='logs/read_iris_test.log', level=logging.DEBUG)

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ONNX = THIS_DIR / "utils" / "data" / "iris_3x2.onnx"
DEFAULT_KERAS_WEIGHTS = THIS_DIR / "benchmark" / "iris" / "iris_weight.h5"


def infer_architecture_from_onnx(onnx_path: Path) -> Tuple[int, ...]:
    """Return (input_dim, *layer_units) inferred from Gemm weights."""
    layer_dims: list[int] = []
    for B, _bias, transB in extract_gemm_params(str(onnx_path)):
        weights = B.T if transB == 1 else B
        if not layer_dims:
            layer_dims.append(int(weights.shape[0]))
        layer_dims.append(int(weights.shape[1]))
    if len(layer_dims) < 2:
        raise ValueError(f"Unable to infer architecture from {onnx_path}")
    return tuple(layer_dims)


def load_reference_keras(weights_path: Path, arch: Tuple[int, ...]) -> tf.keras.Model:
    """Instantiate DeepModel with architecture inferred from ONNX Gemm stack."""
    input_dim, *layer_units = arch
    model = DeepModel(layer_units, last_layer_signed=True, input_scale=1.0)
    model.build((None, input_dim))
    model.load_weights(str(weights_path))
    logging.debug(f"Model summary {model.summary()}")
    return model


def select_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch Iris data in numpy form for the requested split."""
    (x_train, x_test), (y_train, y_test) = load_train_test_data()
    split_map = {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
    }
    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(split_map)}.")
    features, labels = split_map[split]
    return features.astype(np.float32), labels.astype(np.int64)


def run_inference(model: tf.keras.Model, features: np.ndarray) -> np.ndarray:
    predictions = model(tf.convert_to_tensor(features), training=False)
    return predictions.numpy()


def summarize_differences(
    keras_logits: np.ndarray,
    onnx_logits: np.ndarray,
    labels: np.ndarray,
    mismatches_to_show: int,
) -> None:
    logits_diff = onnx_logits - keras_logits
    abs_diff = np.abs(logits_diff)
    mean_abs = float(np.mean(abs_diff))
    max_abs = float(np.max(abs_diff))

    keras_pred = np.argmax(keras_logits, axis=1)
    onnx_pred = np.argmax(onnx_logits, axis=1)
    keras_acc = float(np.mean(keras_pred == labels))
    onnx_acc = float(np.mean(onnx_pred == labels))

    agreement = keras_pred == onnx_pred
    agreement_rate = float(np.mean(agreement))
    mismatches = np.flatnonzero(~agreement)

    print(f"Samples evaluated: {labels.size}")
    print(f"Keras accuracy: {keras_acc:.4f}")
    print(f"ONNX accuracy:  {onnx_acc:.4f}")
    print(f"Agreement rate: {agreement_rate:.4f} ({mismatches.size} mismatches)")
    print(f"Logit abs diff -> mean: {mean_abs:.6f}, max: {max_abs:.6f}")

    if mismatches_to_show > 0 and mismatches.size:
        count = min(mismatches_to_show, mismatches.size)
        print(f"\nFirst {count} mismatched indices:")
        for idx in mismatches[:count]:
            print(
                f"  idx={idx} label={labels[idx]} "
                f"keras={keras_pred[idx]} onnx={onnx_pred[idx]} "
                f"logit_diff={logits_diff[idx]}"
            )


def main() -> None:
    logging.info("Starting Iris model comparison tool...")
    parser = argparse.ArgumentParser(
        description="Compare ONNX and Keras models on the Iris dataset."
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_ONNX,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--keras-weights",
        type=Path,
        default=DEFAULT_KERAS_WEIGHTS,
        help="Path to the Keras weight file (HDF5).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which dataset split to evaluate.",
    )
    parser.add_argument(
        "--mismatches",
        type=int,
        default=0,
        help="Number of mismatched predictions to display.",
    )
    args = parser.parse_args()

    onnx_path = args.onnx_model.resolve()
    keras_weights_path = args.keras_weights.resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    if not keras_weights_path.exists():
        raise FileNotFoundError(f"Keras weights not found at {keras_weights_path}")

    features, labels = select_split(args.split)
    for print_idx in range(min(11, features.shape[0])):
        logging.debug(f"Feature[{print_idx}]: {features[print_idx]} Label: {labels[print_idx]}")
    # Use ONNX Gemm stack to infer architecture for the reference Keras model.
    architecture = infer_architecture_from_onnx(onnx_path)
    keras_reference = load_reference_keras(keras_weights_path, architecture)
    onnx_keras = build_keras_model_from_onnx(onnx_path)

    keras_logits = run_inference(keras_reference, features)
    onnx_logits = run_inference(onnx_keras, features)
    logging.debug(f"Labels: {labels}")
    
    for layer_idx, layer in enumerate(keras_reference.layers):
        logging.debug(f"Keras Reference Layer {layer_idx} weights: {layer.get_weights()}")
    for idx in range(min(30, features.shape[0])):
        logging.debug(f"Feature[{idx}]: {features[idx]}\n Keras output:{np.argmax(keras_logits[idx])}\n Onnx output: {np.argmax(onnx_logits[idx])}\n label {labels[idx]}")

    summarize_differences(keras_logits, onnx_logits, labels, args.mismatches)


if __name__ == "__main__":
    main()
