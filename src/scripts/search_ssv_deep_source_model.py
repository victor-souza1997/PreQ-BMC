"""Search multi-stage GTSRB CNNs that lower to affine/ReLU PreQ-BMC models."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from scripts.run_ssv_cnn_gate import write_new_json
from scripts.search_ssv_source_model import (
    _class_weights,
    _load_search_arrays,
    _sha256,
)

@dataclass(frozen=True)
class DeepCandidatePlan:
    candidate_id: str
    geometries: tuple[ConvGeometry, ...]
    conv_groups: tuple[int, ...]
    dense_hidden: tuple[int, ...]
    global_average_pooling: bool
    average_pool_size: tuple[int, int] | None
    classes: int
    max_affine_entries: int
    shared_parameter_count: int
    affine_entries_per_layer: tuple[int, ...]
    relus_per_layer: tuple[int, ...]
    dropout_rates: tuple[float, ...]


def _available_memory_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

class SearchResourcePause(RuntimeError):
    """Training paused after an epoch checkpoint because memory became scarce."""


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def deep_candidate_plan(config: dict[str, Any], candidate: dict[str, Any]) -> DeepCandidatePlan:
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in candidate_id):
        raise ValueError("Invalid candidate id")
    shape = tuple(config["preprocessing"]["shape"])
    blocks = candidate.get("conv_blocks")
    if not isinstance(blocks, list) or len(blocks) < 2:
        raise ValueError("A deep candidate needs at least two convolution blocks")
    geometries = []
    conv_groups = []
    parameter_count = 0
    affine_entries = []
    relus = []
    current = shape
    for index, block in enumerate(blocks):
        filters = _positive_int(block.get("filters"), f"conv_blocks[{index}].filters")
        kernel = tuple(block.get("kernel_size", ()))
        strides = tuple(block.get("strides", ()))
        if len(kernel) != 2 or len(strides) != 2:
            raise ValueError("Each convolution needs two kernel and stride dimensions")
        groups_value = block.get("groups", 1)
        groups = current[-1] if groups_value == "depthwise" else groups_value
        groups = _positive_int(groups, f"conv_blocks[{index}].groups")
        if current[-1] % groups or filters % groups:
            raise ValueError("Convolution groups must divide input and output channels")
        if groups_value == "depthwise" and filters != current[-1]:
            raise ValueError("Depthwise convolution requires filters equal input channels")
        geometry = ConvGeometry(
            current,
            (*tuple(_positive_int(v, "kernel") for v in kernel), current[-1], filters),
            tuple(_positive_int(v, "stride") for v in strides),
            str(block.get("padding", "")).upper(),
        )
        entries = math.prod(geometry.input_shape) * math.prod(geometry.output_shape)
        if entries > config["max_affine_entries"]:
            raise ValueError(f"{candidate_id} layer {index} needs {entries} affine entries")
        parameter_count += math.prod(kernel) * (current[-1] // groups) * filters + filters
        affine_entries.append(entries)
        relus.append(math.prod(geometry.output_shape))
        geometries.append(geometry)
        conv_groups.append(groups)
        current = geometry.output_shape
    dense_hidden_values = candidate.get("dense_hidden", ())
    if not isinstance(dense_hidden_values, (list, tuple)):
        raise ValueError("dense_hidden must be a list; it may be empty for a direct head")
    dense_hidden = tuple(_positive_int(value, "dense_hidden")
                         for value in dense_hidden_values)
    previous = math.prod(current)
    global_average_pooling = candidate.get("global_average_pooling", False)
    if type(global_average_pooling) is not bool:
        raise ValueError("global_average_pooling must be boolean")
    if global_average_pooling and dense_hidden:
        raise ValueError("global_average_pooling and dense_hidden are mutually exclusive")
    pool_values = candidate.get("average_pool_size")
    if global_average_pooling and pool_values is not None:
        raise ValueError("Choose global_average_pooling or average_pool_size, not both")
    if global_average_pooling:
        average_pool_size = current[:2]
    elif pool_values is None:
        average_pool_size = None
    else:
        if not isinstance(pool_values, (list, tuple)) or len(pool_values) != 2:
            raise ValueError("average_pool_size must contain two positive integers")
        average_pool_size = tuple(_positive_int(value, "average_pool_size") for value in pool_values)
    if average_pool_size is not None:
        ph, pw = average_pool_size
        height, width, channels = current
        if height % ph or width % pw:
            raise ValueError("average_pool_size must divide the final feature map")
        pooled_units = (height // ph) * (width // pw) * channels
        entries = previous * pooled_units
        if entries > config["max_affine_entries"]:
            raise ValueError(f"{candidate_id} pooling layer needs {entries} affine entries")
        affine_entries.append(entries)
        relus.append(pooled_units)
        previous = pooled_units
    for width in (*dense_hidden, config["classes"]):
        entries = previous * width
        if entries > config["max_affine_entries"]:
            raise ValueError(f"{candidate_id} dense layer needs {entries} affine entries")
        parameter_count += entries + width
        affine_entries.append(entries)
        if width != config["classes"]:
            relus.append(width)
        previous = width
    dropout_rates = tuple(float(value) for value in candidate.get(
        "dropout_rates", [0.0] * (len(geometries) + len(dense_hidden)),
    ))
    if (len(dropout_rates) != len(geometries) + len(dense_hidden)
            or any(not np.isfinite(value) or not 0 <= value < 1 for value in dropout_rates)):
        raise ValueError("Need one dropout rate in [0,1) per hidden stage")
    return DeepCandidatePlan(
        candidate_id, tuple(geometries), tuple(conv_groups), dense_hidden, global_average_pooling,
        average_pool_size, config["classes"],
        config["max_affine_entries"], parameter_count,
        tuple(affine_entries), tuple(relus), dropout_rates,
    )


def validate_config(config: dict[str, Any]) -> list[DeepCandidatePlan]:
    if config.get("schema") != "ssv_deep_source_model_search_v1":
        raise ValueError("Expected ssv_deep_source_model_search_v1")
    if config.get("dataset") != "gtsrb_official" or config.get("classes") != 43:
        raise ValueError("Expected the official 43-class GTSRB dataset")
    pre = config.get("preprocessing", {})
    if (pre.get("input_domain") != "uint8_RGB_cropped_sign"
            or pre.get("resize") not in {"nearest_floor_top_left", "nearest_center"}
            or pre.get("normalization_divisor") != 256
            or pre.get("tie_break") != "lowest_class_index"
            or len(pre.get("shape", ())) != 3 or pre["shape"][-1] != 3):
        raise ValueError("Unsupported input semantics")
    if type(config.get("max_affine_entries")) is not int or config["max_affine_entries"] <= 0:
        raise ValueError("max_affine_entries must be positive")
    training = config.get("training", {})
    minimum_memory = config.get("minimum_available_memory_mib", 4096)
    if type(minimum_memory) is not int or minimum_memory < 0:
        raise ValueError("minimum_available_memory_mib must be a nonnegative integer")
    for key in ("seed", "epochs", "batch_size", "early_stopping_patience", "lr_plateau_patience"):
        if type(training.get(key)) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training.get("split_seed", 2026) != 2026 or training.get("validation_fraction") != 0.2:
        raise ValueError("The physical-track split is pinned to seed 2026 and fraction 0.2")
    if training.get("class_weighting") not in {"inverse_frequency", "none"}:
        raise ValueError("class_weighting must be inverse_frequency or none")
    if training.get("early_stopping_monitor", "val_loss") not in {"val_loss", "val_accuracy"}:
        raise ValueError("Unsupported early-stopping monitor")
    weight_decay = training.get("weight_decay", 0.0)
    if type(weight_decay) not in {int, float} or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for key in ("translation_fraction", "contrast_factor", "rotation_fraction", "zoom_fraction"):
        value = training.get("augmentation", {}).get(key)
        if type(value) not in {int, float} or not 0 <= value <= 0.5:
            raise ValueError(f"Invalid augmentation {key}")
    selection = config.get("selection", {})
    threshold = selection.get("minimum_validation_accuracy")
    if type(threshold) not in {int, float} or not 0 < threshold <= 1:
        raise ValueError("Invalid validation threshold")
    if selection.get("rule") != "ordered_first_passing" \
            or selection.get("test_set_policy") != "evaluate_once_after_selection":
        raise ValueError("Expected ordered validation selection with an untouched test set")
    plans = [deep_candidate_plan(config, item) for item in config.get("candidates", [])]
    if not plans or len({plan.candidate_id for plan in plans}) != len(plans):
        raise ValueError("Need unique candidates")
    if [plan.shared_parameter_count for plan in plans] != sorted(
            plan.shared_parameter_count for plan in plans):
        raise ValueError("Candidates must be ordered by increasing parameter count")
    return plans


def fold_batch_norm(kernel, batch_norm_weights, epsilon):
    gamma, beta, mean, variance = (np.asarray(value, dtype=np.float32)
                                   for value in batch_norm_weights)
    scale = gamma / np.sqrt(variance + np.float32(epsilon))
    if kernel.shape[-1] != len(scale):
        raise ValueError("Batch-normalization width mismatch")
    folded_kernel = np.asarray(kernel, dtype=np.float32) * scale.reshape(
        (1,) * (kernel.ndim - 1) + (-1,)
    )
    folded_bias = beta - mean * scale
    return folded_kernel, folded_bias


def _build_clean_model(tf, plan: DeepCandidatePlan):
    inputs = tf.keras.Input(shape=plan.geometries[0].input_shape, name="input")
    value = inputs
    for index, (geometry, groups) in enumerate(zip(plan.geometries, plan.conv_groups, strict=True)):
        value = tf.keras.layers.Conv2D(
            geometry.kernel_shape[-1], geometry.kernel_shape[:2],
            strides=geometry.strides, padding=geometry.padding.lower(),
            groups=groups, use_bias=False, name=f"conv_{index}",
        )(value)
        value = tf.keras.layers.BatchNormalization(name=f"bn_conv_{index}")(value)
        value = tf.keras.layers.ReLU(name=f"relu_conv_{index}")(value)
        if plan.dropout_rates[index]:
            value = tf.keras.layers.Dropout(plan.dropout_rates[index], name=f"dropout_conv_{index}")(value)
    if plan.average_pool_size is not None:
        value = tf.keras.layers.AveragePooling2D(
            pool_size=plan.average_pool_size,
            strides=plan.average_pool_size,
            padding="valid",
            name="average_pooling",
        )(value)
        value = tf.keras.layers.Flatten(name="pool_flatten")(value)
    else:
        value = tf.keras.layers.Flatten(name="flatten")(value)
    for index, width in enumerate(plan.dense_hidden):
        value = tf.keras.layers.Dense(width, use_bias=False, name=f"dense_{index}")(value)
        value = tf.keras.layers.BatchNormalization(name=f"bn_dense_{index}")(value)
        value = tf.keras.layers.ReLU(name=f"relu_dense_{index}")(value)
        dropout = plan.dropout_rates[len(plan.geometries) + index]
        if dropout:
            value = tf.keras.layers.Dropout(dropout, name=f"dropout_dense_{index}")(value)
    logits = tf.keras.layers.Dense(plan.classes, name="logits")(value)
    return tf.keras.Model(inputs, logits, name=plan.candidate_id)


def _training_model(tf, clean, config):
    aug = config["training"]["augmentation"]
    inputs = tf.keras.Input(shape=clean.input_shape[1:], name="training_input")
    value = tf.keras.layers.RandomTranslation(
        aug["translation_fraction"], aug["translation_fraction"],
        fill_mode="nearest", seed=config["training"]["seed"],
    )(inputs)
    value = tf.keras.layers.RandomRotation(
        aug["rotation_fraction"], fill_mode="nearest", seed=config["training"]["seed"] + 1,
    )(value)
    value = tf.keras.layers.RandomZoom(
        aug["zoom_fraction"], aug["zoom_fraction"], fill_mode="nearest",
        seed=config["training"]["seed"] + 2,
    )(value)
    value = tf.keras.layers.RandomContrast(
        aug["contrast_factor"], seed=config["training"]["seed"] + 3,
    )(value)
    return tf.keras.Model(inputs, clean(value), name=f"{clean.name}_training")


def _fold_model(clean, plan: DeepCandidatePlan):
    conv_kernels, conv_biases = [], []
    for index, (geometry, groups) in enumerate(zip(
        plan.geometries, plan.conv_groups, strict=True
    )):
        kernel = clean.get_layer(f"conv_{index}").get_weights()[0]
        if groups > 1:
            input_per_group = geometry.input_shape[-1] // groups
            output_per_group = geometry.output_shape[-1] // groups
            expanded = np.zeros(geometry.kernel_shape, dtype=np.float32)
            for group in range(groups):
                input_slice = slice(group * input_per_group, (group + 1) * input_per_group)
                output_slice = slice(group * output_per_group, (group + 1) * output_per_group)
                expanded[:, :, input_slice, output_slice] = kernel[:, :, :, output_slice]
            kernel = expanded
        bn = clean.get_layer(f"bn_conv_{index}")
        folded = fold_batch_norm(kernel, bn.get_weights(), bn.epsilon)
        conv_kernels.append(folded[0])
        conv_biases.append(folded[1])
    dense_kernels, dense_biases = [], []
    if plan.average_pool_size is not None:
        height, width, channels = plan.geometries[-1].output_shape
        ph, pw = plan.average_pool_size
        output_shape = (height // ph, width // pw, channels)
        pool_kernel = np.zeros((height * width * channels, math.prod(output_shape)), dtype=np.float32)
        scale = np.float32(1.0 / (ph * pw))
        for oy, ox, channel in np.ndindex(output_shape):
            output_index = np.ravel_multi_index((oy, ox, channel), output_shape)
            for ky, kx in np.ndindex((ph, pw)):
                input_index = np.ravel_multi_index(
                    (oy * ph + ky, ox * pw + kx, channel), (height, width, channels),
                )
                pool_kernel[input_index, output_index] = scale
        dense_kernels.append(pool_kernel)
        dense_biases.append(np.zeros(math.prod(output_shape), dtype=np.float32))

    for index in range(len(plan.dense_hidden)):
        kernel = clean.get_layer(f"dense_{index}").get_weights()[0]
        bn = clean.get_layer(f"bn_dense_{index}")
        folded = fold_batch_norm(kernel, bn.get_weights(), bn.epsilon)
        dense_kernels.append(folded[0])
        dense_biases.append(folded[1])
    output_kernel, output_bias = clean.get_layer("logits").get_weights()
    dense_kernels.append(output_kernel)
    dense_biases.append(output_bias)
    return RestrictedSequentialCNN(
        plan.geometries, tuple(conv_kernels), tuple(conv_biases),
        tuple(dense_kernels), tuple(dense_biases), plan.max_affine_entries,
    )


def _balanced_accuracy(labels, predictions, classes):
    return float(np.mean([
        np.mean(predictions[labels == label] == label) for label in range(classes)
    ]))


def train_candidate(tf, plan, arrays, config, output):
    training = config["training"]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(training["seed"])
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    clean = _build_clean_model(tf, plan)
    model = _training_model(tf, clean, config)
    weight_decay = float(training.get("weight_decay", 0.0))
    optimizer = (tf.keras.optimizers.AdamW(
        learning_rate=training["learning_rate"], weight_decay=weight_decay,
    ) if weight_decay else tf.keras.optimizers.Adam(learning_rate=training["learning_rate"]))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    early_monitor = training.get("early_stopping_monitor", "val_loss")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=early_monitor, mode="max" if early_monitor == "val_accuracy" else "min",
            patience=training["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=training["lr_plateau_patience"],
            min_lr=training["minimum_learning_rate"],
        ),
    ]
    backup_dir = output.parent / ".training_backups" / plan.candidate_id
    callbacks.append(tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(backup_dir), save_freq="epoch", delete_checkpoint=False,
    ))

    class AvailableMemoryGuard(tf.keras.callbacks.Callback):
        paused = False

        def on_epoch_end(self, epoch, logs=None):
            available = _available_memory_mib()
            minimum = config.get("minimum_available_memory_mib", 4096)
            if available is not None and available < minimum:
                self.paused = True
                self.model.stop_training = True

    memory_guard = AvailableMemoryGuard()
    callbacks.append(memory_guard)
    started = time.monotonic()
    history = model.fit(
        *arrays["train"], validation_data=arrays["validation"],
        epochs=training["epochs"], batch_size=training["batch_size"],
        class_weight=(_class_weights(arrays["train"][1], plan.classes)
                      if training["class_weighting"] == "inverse_frequency" else None),
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.monotonic() - started
    if memory_guard.paused:
        raise SearchResourcePause(
            f"Available memory fell below {config.get('minimum_available_memory_mib', 4096)} MiB"
        )
    logits = np.asarray(clean.predict(
        arrays["validation"][0], batch_size=training["batch_size"], verbose=0,
    ))
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == arrays["validation"][1]))
    balanced = _balanced_accuracy(arrays["validation"][1], predictions, plan.classes)
    restricted = _fold_model(clean, plan)
    lowered_logits = np.asarray(restricted.as_deep_model()(
        arrays["validation"][0].reshape(len(arrays["validation"][0]), -1), training=False,
    ))
    max_error = float(np.max(np.abs(logits - lowered_logits)))
    predictions_equal = bool(np.array_equal(predictions, np.argmax(lowered_logits, axis=1)))
    if not predictions_equal or not np.allclose(logits, lowered_logits, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Folded affine model parity failed: max_abs_error={max_error}")

    output.mkdir(parents=True, exist_ok=False)
    arrays_to_save = {}
    for index, value in enumerate(restricted.conv_kernels):
        arrays_to_save[f"conv_{index}_kernel"] = value
        arrays_to_save[f"conv_{index}_bias"] = restricted.conv_biases[index]
    for index, value in enumerate(restricted.dense_kernels):
        arrays_to_save[f"dense_{index}_kernel"] = value
        arrays_to_save[f"dense_{index}_bias"] = restricted.dense_biases[index]
    params = output / "folded_model.npz"
    np.savez(params, **arrays_to_save)
    clean.save(output / "source_with_batch_norm.keras")
    serial_history = {key: [float(item) for item in values]
                      for key, values in history.history.items()}
    write_new_json(output / "history.json", serial_history)
    result = {
        "schema": "ssv_deep_source_model_candidate_v1",
        "candidate_id": plan.candidate_id,
        "status": "COMPLETED",
        "validation_accuracy": accuracy,
        "validation_balanced_accuracy": balanced,
        "epochs_completed": len(serial_history["loss"]),
        "best_validation_accuracy_observed": float(max(serial_history["val_accuracy"])),
        "training_elapsed_seconds": float(elapsed),
        "geometries": [asdict(geometry) for geometry in plan.geometries],
        # The fixed pooling map remains an ordinary non-output affine layer.
        "dense_hidden": ([plan.relus_per_layer[-1]]
                         if plan.average_pool_size is not None else list(plan.dense_hidden)),
        "conv_groups": list(plan.conv_groups),
        "global_average_pooling": plan.global_average_pooling,
        "average_pool_size": list(plan.average_pool_size) if plan.average_pool_size else None,
        "shared_parameter_count": plan.shared_parameter_count,
        "affine_entries_per_layer": list(plan.affine_entries_per_layer),
        "relus_per_layer": list(plan.relus_per_layer),
        "dropout_rates_during_training": list(plan.dropout_rates),
        "training_seed": training["seed"],
        "weight_decay": weight_decay,
        "total_relus": sum(plan.relus_per_layer),
        "folded_model_path": str(params.resolve()),
        "folded_model_sha256": _sha256(params),
        "float_lowering_max_abs_error": max_error,
        "float_lowering_predictions_equal": predictions_equal,
        "float_lowering_claim": "tested_numerical_equivalence_not_IEEE_proof",
        "test_status": "NOT_EVALUATED",
        "quantization_status": "NOT_RUN",
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
    }
    write_new_json(output / "candidate_summary.json", result)
    return result


def search(config_path: Path, output: Path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plans = validate_config(config)
    base_path = Path(config["base_study"])
    if not base_path.is_absolute():
        base_path = (config_path.parents[1] / base_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("schema") != "ssv_restricted_cnn_v1":
        raise ValueError("Invalid frozen base study")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_summary = output / "search_summary.json"
    if final_summary.exists():
        existing = json.loads(final_summary.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != _sha256(config_path):
            raise ValueError("Existing search output belongs to a different configuration")
        return existing

    results, selected = [], None
    threshold = config["selection"]["minimum_validation_accuracy"]
    next_plan_index = 0
    for index, plan in enumerate(plans):
        candidate_summary = output / plan.candidate_id / "candidate_summary.json"
        if not candidate_summary.exists():
            next_plan_index = index
            break
        result = json.loads(candidate_summary.read_text(encoding="utf-8"))
        if result.get("candidate_id") != plan.candidate_id:
            raise ValueError(f"Existing candidate output mismatch: {candidate_summary}")
        results.append(result)
        next_plan_index = index + 1
        if result["validation_accuracy"] >= threshold:
            selected = result
            break

    minimum_memory = config.get("minimum_available_memory_mib", 4096)
    available_memory = _available_memory_mib()
    paused = (selected is None and next_plan_index < len(plans)
              and available_memory is not None and available_memory < minimum_memory)
    tf = None
    geometry = plans[0].geometries[0]
    if not paused and selected is None and next_plan_index < len(plans):
        import tensorflow as tf_module
        tf = tf_module
        if config["preprocessing"]["resize"] == "nearest_floor_top_left":
            arrays = _load_search_arrays(base, geometry)
        else:
            encoder = ByteImageEncoder(
                geometry.input_shape, resize_mode=config["preprocessing"]["resize"]
            )
            root = Path(base["dataset_root"])
            arrays = {}
            for split in ("train", "validation"):
                rows = [row for row in base["records"] if row["split"] == split]
                x = np.stack([encoder.resize(load_crop(root, row)) for row in rows])
                arrays[split] = (
                    x.astype(np.float32) / np.float32(256),
                    np.asarray([row["class_id"] for row in rows], dtype=np.int64),
                )
        for plan in plans[next_plan_index:]:
            available_memory = _available_memory_mib()
            if available_memory is not None and available_memory < minimum_memory:
                paused = True
                break
            try:
                result = train_candidate(tf, plan, arrays, config, output / plan.candidate_id)
            except SearchResourcePause:
                paused = True
                break
            results.append(result)
            if result["validation_accuracy"] >= threshold:
                selected = result
                break

    summary = {
        "schema": config["schema"],
        "status": ("SELECTED" if selected else
                   "RESOURCE_PAUSED" if paused else "NO_CANDIDATE_MET_THRESHOLD"),
        "minimum_validation_accuracy": threshold,
        "minimum_available_memory_mib": minimum_memory,
        "available_memory_mib_at_stop": _available_memory_mib(),
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_summary": selected,
        "candidates": results,
        "untrained_larger_candidates": [plan.candidate_id for plan in plans[len(results):]],
        "selection_rule": config["selection"]["rule"],
        "selection_data": "track_safe_validation_only",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_study_path": str(base_path),
        "base_study_sha256": _sha256(base_path),
        "test_status": "NOT_EVALUATED",
        "quantization_status": "NOT_RUN",
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__ if tf is not None else None,
    }
    if paused:
        _write_progress(output / "search_progress.json", summary)
    else:
        write_new_json(final_summary, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/sign_deep_source_model_search.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check_config:
        plans = validate_config(config)
        print(json.dumps({"configuration_valid": True,
                          "candidates": [asdict(plan) for plan in plans],
                          "test_status": "NOT_EVALUATED"}, indent=2))
        return
    result = search(args.config, args.output)
    print(json.dumps({"status": result["status"],
                      "selected_candidate_id": result["selected_candidate_id"],
                      "summary": str((args.output / (
                          "search_progress.json" if result["status"] == "RESOURCE_PAUSED"
                          else "search_summary.json")).resolve())}, indent=2))


if __name__ == "__main__":
    main()
