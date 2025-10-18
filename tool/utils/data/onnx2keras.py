from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import onnx
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "onnx2keras requires the 'onnx' package. "
        "Install it with `pip install onnx`."
    ) from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "onnx2keras requires TensorFlow. "
        "Install it with `pip install tensorflow`."
    ) from exc

from load_onnx import (
    extract_gemm_params,
    infer_onnx_input_dim,
    load_onnx_model,
    load_onnx_weights_into_keras,
)

LOGGER = logging.getLogger(__name__)

# Map ONNX activation node type to Keras activation string.
_ACTIVATION_MAP: Dict[str, str] = {
    "Relu": "relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "Softsign": "softsign",
    "Softplus": "softplus",
    "Selu": "selu",
    "Elu": "elu",
    "Identity": "linear",
}


def _collect_consumers(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    """Build a mapping from tensor name to the list of nodes that consume it."""
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _select_activation(
    gemm_output: str,
    consumers: Dict[str, List[onnx.NodeProto]],
) -> str:
    """Pick the best matching activation for a Gemm node output."""
    for consumer in consumers.get(gemm_output, []):
        if not consumer.input:
            continue
        if consumer.input[0] != gemm_output:
            # Activation nodes we care about take Gemm output as their first input.
            continue
        activation = _ACTIVATION_MAP.get(consumer.op_type)
        if activation:
            return activation

    return "linear"


def _dense_layer_configs(model_proto: onnx.ModelProto) -> List[Tuple[str, str]]:
    """
    Extract layer name and activation for each Gemm node in graph order.

    Returns a list whose index matches the ordering produced by extract_gemm_params.
    """
    graph = model_proto.graph
    consumers = _collect_consumers(graph)

    configs: List[Tuple[str, str]] = []
    gemm_index = 0
    used_names: Dict[str, int] = {}

    for node in graph.node:
        if node.op_type != "Gemm":
            continue

        raw_name = node.name or f"dense_{gemm_index}"
        name_count = used_names.get(raw_name, 0)
        if name_count:
            sanitized_name = f"{raw_name}_{name_count}"
        else:
            sanitized_name = raw_name
        used_names[raw_name] = name_count + 1

        output_name = node.output[0] if node.output else f"{sanitized_name}_output"
        activation = _select_activation(output_name, consumers)

        configs.append((sanitized_name, activation))
        gemm_index += 1

    return configs


def build_keras_model_from_onnx(onnx_path: Path | str) -> tf.keras.Model:
    """
    Reconstruct a Keras dense model that mirrors the ONNX Gemm stack and load weights.

    Parameters
    ----------
    onnx_path:
        Path to the source ONNX model.

    Returns
    -------
    tf.keras.Model
        The reconstructed Keras model with weights populated from the ONNX file.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    LOGGER.debug("Loading ONNX model from %s", onnx_path)
    model_proto = load_onnx_model(str(onnx_path))

    gemm_params = extract_gemm_params(str(onnx_path))
    layer_meta = _dense_layer_configs(model_proto)

    if len(layer_meta) != len(gemm_params):
        LOGGER.warning(
            "Detected %d Gemm nodes but found %d activation hints. "
            "Falling back to 'linear' for missing activations.",
            len(gemm_params),
            len(layer_meta),
        )
        # Extend with default configs where required.
        while len(layer_meta) < len(gemm_params):
            idx = len(layer_meta)
            layer_meta.append((f"dense_{idx}", "linear"))
        # Trim if we somehow recorded extra hints (should not happen).
        layer_meta = layer_meta[: len(gemm_params)]

    input_dim = infer_onnx_input_dim(str(onnx_path))
    LOGGER.debug("Inferred input dimension: %s", input_dim)

    keras_input = tf.keras.Input(shape=(input_dim,), name=f"{onnx_path.stem}_input")
    x = keras_input

    for idx, ((name, activation), (B, _, transB)) in enumerate(
        zip(layer_meta, gemm_params)
    ):
        weight_matrix = B.T if transB == 1 else B
        units = int(weight_matrix.shape[1])

        LOGGER.debug(
            "Creating Dense layer %d: name=%s, units=%d, activation=%s",
            idx,
            name,
            units,
            activation,
        )
        dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            name=name,
        )
        x = dense_layer(x)

    keras_model = tf.keras.Model(
        inputs=keras_input, outputs=x, name=f"{onnx_path.stem}_keras"
    )

    LOGGER.debug("Transferring weights from ONNX Gemm nodes to Keras model.")
    load_onnx_weights_into_keras(keras_model, str(onnx_path))

    return keras_model


def convert_onnx_to_keras(
    onnx_path: Path | str,
    model_output_path: Path | str,
    *,
    weights_output_path: Optional[Path | str] = None,
    overwrite: bool = True,
) -> tf.keras.Model:
    """
    Convert an ONNX dense network into a Keras model and persist it as .h5 artifacts.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model file.
    model_output_path:
        Destination path for the serialized Keras model (`model.save`).
    weights_output_path:
        Destination path for the standalone weights (`model.save_weights`). If omitted,
        `<model_output_stem>_weights.h5` is used next to the model file.
    overwrite:
        Whether to overwrite existing output files.

    Returns
    -------
    tf.keras.Model
        The converted and weight-populated Keras model.
    """
    onnx_path = Path(onnx_path)
    model_output_path = Path(model_output_path)
    weights_output_path = (
        Path(weights_output_path)
        if weights_output_path is not None
        else model_output_path.with_name(f"{model_output_path.stem}_weights.h5")
    )

    keras_model = build_keras_model_from_onnx(onnx_path)

    if model_output_path.suffix.lower() != ".h5":
        model_output_path = model_output_path.with_suffix(".h5")
    if weights_output_path.suffix.lower() != ".h5":
        weights_output_path = weights_output_path.with_suffix(".h5")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving Keras model to %s", model_output_path)
    keras_model.save(model_output_path, overwrite=overwrite)

    LOGGER.info("Saving Keras weights to %s", weights_output_path)
    keras_model.save_weights(weights_output_path, overwrite=overwrite)

    return keras_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model composed of Gemm nodes to Keras (.h5)."
    )
    parser.add_argument("onnx_path", type=Path, help="Path to the ONNX model.")
    parser.add_argument(
        "--model-out",
        "-m",
        dest="model_output",
        type=Path,
        required=True,
        help="Path to save the Keras model (.h5).",
    )
    parser.add_argument(
        "--weights-out",
        "-w",
        dest="weights_output",
        type=Path,
        default=None,
        help="Optional path to save a separate weights .h5 file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for logging messages.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Do not overwrite existing output files.",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    convert_onnx_to_keras(
        onnx_path=args.onnx_path,
        model_output_path=args.model_output,
        weights_output_path=args.weights_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
