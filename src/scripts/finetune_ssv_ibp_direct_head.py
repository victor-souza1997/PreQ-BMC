"""Fine-tune a direct-head GTSRB model with sound interval-bound loss.

The result is a source-model candidate.  IBP certification does not replace the
convolution-native ESBMC proof of the quantized deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop
from models.restricted_conv import RestrictedSequentialCNN
from scripts.evaluate_ssv_deep_source_model import restricted_from_npz
from scripts.run_ssv_cnn_gate import write_new_json
from scripts.search_ssv_source_model import _sha256


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    if config.get("schema") != "ssv_conv_ibp_finetune_v1":
        raise ValueError("Expected ssv_conv_ibp_finetune_v1")
    for key in ("epochs", "batch_size", "seed", "warmup_epochs"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("learning_rate", "epsilon_raw_bytes", "maximum_robust_weight",
                "robust_logit_temperature"):
        value = config.get(key)
        if type(value) not in {int, float} or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["maximum_robust_weight"] > 1:
        raise ValueError("maximum_robust_weight cannot exceed one")
    minimum = config.get("minimum_validation_accuracy")
    if type(minimum) not in {int, float} or not 0 < minimum <= 1:
        raise ValueError("minimum_validation_accuracy must be in (0,1]")
    ramp = config.get("epsilon_warmup_epochs", 1)
    if type(ramp) is not int or ramp <= 0:
        raise ValueError("epsilon_warmup_epochs must be positive")
    training_epsilon = config.get("training_epsilon_raw_bytes", config["epsilon_raw_bytes"])
    if type(training_epsilon) not in {int, float} or not np.isfinite(training_epsilon) \
            or training_epsilon < config["epsilon_raw_bytes"]:
        raise ValueError("training_epsilon_raw_bytes must be at least epsilon_raw_bytes")


def group_masks(restricted: RestrictedSequentialCNN, conv_groups):
    """Block-diagonal masks that keep grouped kernels grouped during training.

    Folded models store grouped (e.g. depthwise) kernels densely with zeros
    between groups; unmasked gradients would silently turn them into full
    convolutions and change the verified architecture.
    """
    masks = []
    for geometry, groups in zip(restricted.geometries, conv_groups, strict=True):
        if groups == 1:
            masks.append(None)
            continue
        kh, kw, channels_in, channels_out = geometry.kernel_shape
        per_in, per_out = channels_in // groups, channels_out // groups
        mask = np.zeros(geometry.kernel_shape, dtype=np.float32)
        for group in range(groups):
            mask[:, :, group * per_in:(group + 1) * per_in,
                 group * per_out:(group + 1) * per_out] = 1
        masks.append(mask)
    return tuple(masks)


def _build_model(tf, restricted: RestrictedSequentialCNN):
    inputs = tf.keras.Input(shape=restricted.input_shape, name="input")
    value = inputs
    conv_layers = []
    for index, geometry in enumerate(restricted.geometries):
        layer = tf.keras.layers.Conv2D(
            geometry.kernel_shape[-1], geometry.kernel_shape[:2],
            strides=geometry.strides, padding=geometry.padding.lower(),
            use_bias=True, name=f"conv_{index}",
        )
        value = tf.keras.layers.ReLU(name=f"relu_{index}")(layer(value))
        conv_layers.append(layer)
    value = tf.keras.layers.Flatten(name="flatten")(value)
    if len(restricted.dense_kernels) != 1:
        raise ValueError("IBP fine-tuning currently requires a direct linear head")
    output_layer = tf.keras.layers.Dense(len(restricted.dense_biases[0]), name="logits")
    logits = output_layer(value)
    model = tf.keras.Model(inputs, logits, name="verification_aware_direct_head")
    for layer, kernel, bias in zip(
            conv_layers, restricted.conv_kernels, restricted.conv_biases, strict=True):
        layer.set_weights([kernel, bias])
    output_layer.set_weights([restricted.dense_kernels[0], restricted.dense_biases[0]])
    return model, tuple(conv_layers), output_layer


def interval_logits(tf, low, high, conv_layers, output_layer):
    """Sound box propagation through Conv/ReLU/Flatten/Dense."""
    for layer in conv_layers:
        kernel, bias = layer.kernel, layer.bias
        positive, negative = tf.maximum(kernel, 0), tf.minimum(kernel, 0)
        kwargs = {
            "strides": [1, *layer.strides, 1],
            "padding": layer.padding.upper(),
            "data_format": "NHWC",
        }
        next_low = (tf.nn.conv2d(low, positive, **kwargs)
                    + tf.nn.conv2d(high, negative, **kwargs) + bias)
        next_high = (tf.nn.conv2d(high, positive, **kwargs)
                     + tf.nn.conv2d(low, negative, **kwargs) + bias)
        low, high = tf.nn.relu(next_low), tf.nn.relu(next_high)
    low, high = tf.reshape(low, (tf.shape(low)[0], -1)), tf.reshape(high, (tf.shape(high)[0], -1))
    kernel, bias = output_layer.kernel, output_layer.bias
    positive, negative = tf.maximum(kernel, 0), tf.minimum(kernel, 0)
    return (
        low @ positive + high @ negative + bias,
        high @ positive + low @ negative + bias,
    )


def _arrays(base, input_shape, resize_mode):
    encoder = ByteImageEncoder(input_shape, resize_mode=resize_mode)
    root = Path(base["dataset_root"])
    result = {}
    for split in ("train", "validation"):
        rows = [row for row in base["records"] if row["split"] == split]
        images = np.stack([encoder.resize(load_crop(root, row)) for row in rows])
        result[split] = (
            images.astype(np.float32) / np.float32(256),
            np.asarray([row["class_id"] for row in rows], dtype=np.int64),
        )
    return result


def _evaluate(tf, model, conv_layers, output_layer, arrays, epsilon, batch_size):
    predictions, labels, certified, margins = [], [], [], []
    for start in range(0, len(arrays[0]), batch_size):
        x = tf.convert_to_tensor(arrays[0][start:start + batch_size])
        y = tf.convert_to_tensor(arrays[1][start:start + batch_size])
        logits = model(x, training=False)
        low, high = interval_logits(
            tf, tf.maximum(0.0, x - epsilon), tf.minimum(255 / 256, x + epsilon),
            conv_layers, output_layer,
        )
        target_low = tf.gather(low, y, batch_dims=1)
        masked = tf.where(tf.one_hot(y, tf.shape(high)[1], on_value=True, off_value=False),
                          tf.constant(-np.inf, high.dtype), high)
        margin = target_low - tf.reduce_max(masked, axis=1)
        predictions.extend(np.argmax(np.asarray(logits), axis=1).tolist())
        labels.extend(np.asarray(y).tolist())
        margins.extend(np.asarray(margin).tolist())
        certified.extend(np.asarray(margin > 0).tolist())
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "certified_fraction": float(np.mean(certified)),
        "mean_margin_lower_bound": float(np.mean(margins)),
        "minimum_margin_lower_bound": float(np.min(margins)),
    }


def finetune(config_path: Path, output: Path):
    import tensorflow as tf

    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    validate_config(config)
    source_root = Path(config["source_search_output"])
    if not source_root.is_absolute():
        source_root = (config_path.parents[1] / source_root).resolve()
    search = _load_json(source_root / "search_summary.json")
    selected = search.get("selected_candidate_summary")
    if not selected or selected.get("dense_hidden") != []:
        raise ValueError("Source search must select a direct-head candidate")
    params = Path(selected["folded_model_path"])
    if _sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Source candidate identity changed")
    base_path = Path(search["base_study_path"])
    if _sha256(base_path) != search["base_study_sha256"]:
        raise ValueError("Frozen data manifest changed")
    base = _load_json(base_path)
    restricted = restricted_from_npz(selected, params)
    masks = group_masks(restricted, selected.get("conv_groups", [1] * len(restricted.geometries)))
    for kernel, mask in zip(restricted.conv_kernels, masks, strict=True):
        if mask is not None and np.any(kernel[mask == 0]):
            raise ValueError("Grouped kernel has weights outside its groups")
    arrays = _arrays(base, restricted.input_shape, config["resize_mode"])

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(config["seed"])
    model, conv_layers, output_layer = _build_model(tf, restricted)
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    epsilon = tf.constant(config["epsilon_raw_bytes"] / 256, tf.float32)
    training_epsilon = config.get("training_epsilon_raw_bytes", config["epsilon_raw_bytes"]) / 256
    mask_pairs = [
        (layer.kernel, tf.constant(mask))
        for layer, mask in zip(conv_layers, masks, strict=True) if mask is not None
    ]
    dataset = tf.data.Dataset.from_tensor_slices(arrays["train"]).shuffle(
        len(arrays["train"][0]), seed=config["seed"], reshuffle_each_iteration=True,
    ).batch(config["batch_size"])

    initial_metrics = _evaluate(
        tf, model, conv_layers, output_layer, arrays["validation"],
        float(epsilon), config["batch_size"],
    )
    history = [{"epoch": 0, "loss": None, "robust_weight": 0.0, **initial_metrics}]
    best_weights = model.get_weights()
    best_score = (
        initial_metrics["accuracy"] >= config["minimum_validation_accuracy"],
        initial_metrics["certified_fraction"], initial_metrics["mean_margin_lower_bound"],
        initial_metrics["accuracy"],
    )
    stale = 0
    print(json.dumps(history[0]), flush=True)
    started = time.monotonic()
    for epoch in range(config["epochs"]):
        robust_weight = config["maximum_robust_weight"] * min(
            1.0, (epoch + 1) / config["warmup_epochs"]
        )
        # Ramp the training radius from near zero so the robust loss starts close
        # to the clean loss instead of hundreds of logit units away from it.
        step_epsilon = tf.constant(training_epsilon * min(
            1.0, (epoch + 1) / config.get("epsilon_warmup_epochs", 1)), tf.float32)
        losses = []
        for x, y in dataset:
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                low, high = interval_logits(
                    tf, tf.maximum(0.0, x - step_epsilon),
                    tf.minimum(255 / 256, x + step_epsilon),
                    conv_layers, output_layer,
                )
                worst_logits = high + tf.one_hot(y, tf.shape(high)[1], dtype=high.dtype) * (low - high)
                clean_loss = loss_function(y, logits)
                robust_loss = loss_function(
                    y, worst_logits / config["robust_logit_temperature"]
                )
                loss = (1 - robust_weight) * clean_loss + robust_weight * robust_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
            for kernel, mask in mask_pairs:
                kernel.assign(kernel * mask)
            losses.append(float(loss))
        metrics = _evaluate(
            tf, model, conv_layers, output_layer, arrays["validation"],
            float(epsilon), config["batch_size"],
        )
        score = (
            metrics["accuracy"] >= config["minimum_validation_accuracy"],
            metrics["certified_fraction"], metrics["mean_margin_lower_bound"], metrics["accuracy"],
        )
        improved = best_score is None or score > best_score
        if improved:
            best_score, best_weights, stale = score, model.get_weights(), 0
        else:
            stale += 1
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                  "robust_weight": robust_weight,
                  "training_epsilon_raw_bytes": float(step_epsilon) * 256, **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if stale >= config.get("early_stopping_patience", 10):
            break
    model.set_weights(best_weights)
    metrics = _evaluate(
        tf, model, conv_layers, output_layer, arrays["validation"],
        float(epsilon), config["batch_size"],
    )

    conv_kernels = tuple(np.asarray(layer.kernel) for layer in conv_layers)
    conv_biases = tuple(np.asarray(layer.bias) for layer in conv_layers)
    dense_kernels = (np.asarray(output_layer.kernel),)
    dense_biases = (np.asarray(output_layer.bias),)
    final = RestrictedSequentialCNN(
        restricted.geometries, conv_kernels, conv_biases, dense_kernels, dense_biases,
        restricted.max_affine_entries,
    )
    for kernel, mask in zip(conv_kernels, masks, strict=True):
        if mask is not None and np.any(kernel[mask == 0]):
            raise AssertionError("Grouped kernel lost its group structure")
    candidate_id = selected["candidate_id"] + config.get(
        "candidate_suffix", f"_ibp_eps{config['epsilon_raw_bytes']}")
    candidate_root = output / candidate_id
    candidate_root.mkdir()
    values = {}
    for index, (kernel, bias) in enumerate(zip(conv_kernels, conv_biases, strict=True)):
        values[f"conv_{index}_kernel"], values[f"conv_{index}_bias"] = kernel, bias
    values["dense_0_kernel"], values["dense_0_bias"] = dense_kernels[0], dense_biases[0]
    folded_path = candidate_root / "folded_model.npz"
    np.savez(folded_path, **values)
    model.save(candidate_root / "source_with_batch_norm.keras")
    write_new_json(candidate_root / "history.json", history)
    candidate = {
        **selected,
        "candidate_id": candidate_id,
        "validation_accuracy": metrics["accuracy"],
        "validation_ibp_certified_fraction": metrics["certified_fraction"],
        "validation_ibp_mean_margin_lower_bound": metrics["mean_margin_lower_bound"],
        "validation_ibp_minimum_margin_lower_bound": metrics["minimum_margin_lower_bound"],
        "ibp_epsilon_raw_bytes": config["epsilon_raw_bytes"],
        "training_elapsed_seconds": time.monotonic() - started,
        "epochs_completed": len(history),
        "folded_model_path": str(folded_path),
        "folded_model_sha256": _sha256(folded_path),
        "test_status": "NOT_EVALUATED",
        "quantization_status": "NOT_RUN",
        "proof_status": "NOT_RUN",
    }
    write_new_json(candidate_root / "candidate_summary.json", candidate)
    summary = {
        **search,
        "status": "SELECTED" if metrics["accuracy"] >= config["minimum_validation_accuracy"] else "NO_CANDIDATE_MET_THRESHOLD",
        "selected_candidate_id": candidate_id if metrics["accuracy"] >= config["minimum_validation_accuracy"] else None,
        "selected_candidate_summary": candidate if metrics["accuracy"] >= config["minimum_validation_accuracy"] else None,
        "candidates": [candidate],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_search_output": str(source_root),
        "test_status": "NOT_EVALUATED",
        "quantization_status": "NOT_RUN",
        "proof_status": "NOT_RUN",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }
    write_new_json(output / "search_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = _load_json(args.config)
    if args.check_config:
        validate_config(config)
        print(json.dumps({"configuration_valid": True, "test_status": "NOT_EVALUATED"}, indent=2))
        return
    result = finetune(args.config, args.output)
    print(json.dumps({"status": result["status"],
                      "selected_candidate_id": result["selected_candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
