import os
from typing import Optional, Sequence


def save_keras_to_onnx(model, output_path: str, input_shape: Optional[Sequence[int]] = None, opset: Optional[int] = None) -> str:
    """
    Export a tf.keras.Model to ONNX.

    Tries tf2onnx.from_keras first; for subclassed models falls back to
    tf2onnx.from_function using a traced concrete function.

    Parameters:
    - model: tf.keras.Model instance (already built)
    - output_path: destination .onnx file path
    - input_shape: optional iterable with input dimensions (excluding batch).
                   If not provided, attempts to infer from model.input_shape or model._input_shape.
    - opset: optional ONNX opset to use (defaults to tf2onnx default)

    Returns:
    - The path to the saved ONNX file
    """
    # Import locally to avoid hard dependency when user doesn't request export
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        raise RuntimeError("TensorFlow is required for ONNX export but was not found.") from e

    try:
        import tf2onnx  # type: ignore
        from tf2onnx import convert
    except Exception as e:
        raise RuntimeError(
            "tf2onnx is required to export Keras models to ONNX. Please install it (e.g., pip install tf2onnx)."
        ) from e

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Infer input shape if not provided
    inferred = None
    if input_shape is not None:
        inferred = list(input_shape)
    else:
        # Try common attributes
        for attr in ("input_shape", "_input_shape"):
            shp = getattr(model, attr, None)
            if shp is None:
                continue
            # shp could be Tuple[None, D] or List of shapes
            if isinstance(shp, (list, tuple)) and shp and isinstance(shp[0], (list, tuple)):
                # handle list of shapes (multi-input), take first
                shp = shp[0]
            if isinstance(shp, (list, tuple)) and len(shp) >= 1:
                # drop batch dim if present
                dims = list(shp)[1:] if shp[0] is None else list(shp)
                inferred = dims
                break
    if inferred is None:
        raise RuntimeError("Unable to infer input shape for ONNX export. Provide input_shape explicitly.")

    # Ensure we have a static batch size for tracing and export
    batch = 1
    input_spec = [tf.TensorSpec(shape=[batch] + list(inferred), dtype=tf.float32, name="input")]

    # Try from_keras first
    try:
        convert.from_keras(
            model,
            input_signature=input_spec,
            opset=opset,
            output_path=output_path,
        )
        return output_path
    except Exception:
        pass  # Fall back to from_function

    # Fallback: trace a concrete function from the model call
    @tf.function(input_signature=input_spec)
    def _forward(x):
        return model(x)

    concrete = _forward.get_concrete_function()
    convert.from_function(
        concrete,
        opset=opset,
        output_path=output_path,
        input_signature=input_spec,
    )
    return output_path

