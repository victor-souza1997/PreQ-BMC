"""Validation-only search for a verifier-compatible GTSRB source model.

The command deliberately does not evaluate the official test split or invoke a
solver.  It selects an architecture using the frozen track-safe validation
split, leaving test accuracy, quantization, ESBMC, and Android measurements for
the subsequent frozen-model study.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from backends.image_encoder import ByteImageEncoder
from models.restricted_conv import ConvGeometry, RestrictedCNN
from scripts.run_ssv_cnn_gate import write_new_json


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    geometry: ConvGeometry
    classes: int
    max_affine_entries: int
    shared_parameter_count: int
    expanded_affine_parameter_count: int
    affine_lowering_entries: int
    hidden_relu_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def candidate_plan(preprocessing: dict[str, Any], candidate: dict[str, Any],
                   *, classes: int, max_affine_entries: int) -> CandidatePlan:
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str) or not candidate_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in candidate_id):
        raise ValueError("Candidate id must use lowercase ASCII letters, digits, '_' or '-'")
    shape = tuple(preprocessing.get("shape", ()))
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError("Search requires an HWC RGB input shape")
    kernel = tuple(candidate.get("kernel_size", ()))
    strides = tuple(candidate.get("strides", ()))
    filters = _require_int(candidate.get("filters"), f"{candidate_id}.filters")
    if len(kernel) != 2 or len(strides) != 2:
        raise ValueError("kernel_size and strides must each contain two integers")
    geometry = ConvGeometry(
        tuple(_require_int(v, "input shape") for v in shape),
        (*tuple(_require_int(v, "kernel size") for v in kernel), 3, filters),
        tuple(_require_int(v, "stride") for v in strides),
        str(candidate.get("padding", "")).upper(),
    )
    input_values = math.prod(geometry.input_shape)
    hidden_values = math.prod(geometry.output_shape)
    affine_entries = input_values * hidden_values
    if affine_entries > max_affine_entries:
        raise ValueError(
            f"Candidate {candidate_id} needs {affine_entries} affine entries; "
            f"limit={max_affine_entries}"
        )
    conv_parameters = math.prod(geometry.kernel_shape) + filters
    dense_parameters = hidden_values * classes + classes
    # Lowering repeats each convolutional bias and materializes a dense matrix.
    expanded_parameters = affine_entries + hidden_values + dense_parameters
    return CandidatePlan(
        candidate_id=candidate_id,
        geometry=geometry,
        classes=classes,
        max_affine_entries=max_affine_entries,
        shared_parameter_count=conv_parameters + dense_parameters,
        expanded_affine_parameter_count=expanded_parameters,
        affine_lowering_entries=affine_entries,
        hidden_relu_count=hidden_values,
    )


def validate_search_config(config: dict[str, Any]) -> list[CandidatePlan]:
    if config.get("schema") != "ssv_source_model_search_v1":
        raise ValueError("Expected ssv_source_model_search_v1")
    if config.get("dataset") != "gtsrb_official":
        raise ValueError("This search supports only the official GTSRB crop dataset")
    base_study = config.get("base_study")
    if not isinstance(base_study, str) or not base_study:
        raise ValueError("base_study must identify the frozen track-safe split")
    pre = config.get("preprocessing", {})
    if (pre.get("input_domain") != "uint8_RGB_cropped_sign"
            or pre.get("resize") != "nearest_floor_top_left"
            or pre.get("normalization_divisor") != 256
            or pre.get("tie_break") != "lowest_class_index"):
        raise ValueError("Unsupported preprocessing semantics")
    training = config.get("training", {})
    _require_int(training.get("seed"), "training.seed", minimum=0)
    _require_int(training.get("epochs"), "training.epochs")
    _require_int(training.get("batch_size"), "training.batch_size")
    for key in ("early_stopping_patience", "lr_plateau_patience"):
        _require_int(training.get(key), f"training.{key}")
    for key in ("learning_rate", "minimum_learning_rate"):
        value = training.get(key)
        if type(value) not in {int, float} or not np.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if training["minimum_learning_rate"] > training["learning_rate"]:
        raise ValueError("minimum_learning_rate cannot exceed learning_rate")
    augmentation = training.get("augmentation", {})
    for key in ("translation_fraction", "contrast_factor"):
        value = augmentation.get(key)
        if type(value) not in {int, float} or not 0 <= value <= 0.5:
            raise ValueError(f"training.augmentation.{key} must be in [0,0.5]")
    if training.get("class_weighting") != "inverse_frequency":
        raise ValueError("This search pins inverse-frequency class weighting")
    selection = config.get("selection", {})
    threshold = selection.get("minimum_validation_accuracy")
    if type(threshold) not in {int, float} or not 0 < threshold <= 1:
        raise ValueError("minimum_validation_accuracy must be in (0,1]")
    if selection.get("rule") != "smallest_passing_then_highest_accuracy":
        raise ValueError("Unsupported selection rule")
    if selection.get("test_set_policy") != "evaluate_once_after_selection":
        raise ValueError("The test set must remain untouched during model selection")
    classes = _require_int(config.get("classes"), "classes", minimum=2)
    max_entries = _require_int(config.get("max_affine_entries"), "max_affine_entries")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("At least one candidate is required")
    plans = [candidate_plan(pre, candidate, classes=classes, max_affine_entries=max_entries)
             for candidate in candidates]
    if len({plan.candidate_id for plan in plans}) != len(plans):
        raise ValueError("Candidate ids must be unique")
    return plans


def select_candidate(results: list[dict[str, Any]], minimum_validation_accuracy: float) -> dict[str, Any] | None:
    passing = [result for result in results
               if result.get("status") == "COMPLETED"
               and float(result["validation_accuracy"]) >= minimum_validation_accuracy]
    if not passing:
        return None
    return min(passing, key=lambda result: (
        int(result["shared_parameter_count"]),
        -float(result["validation_accuracy"]),
        str(result["candidate_id"]),
    ))


def _load_search_arrays(study: dict[str, Any], geometry: ConvGeometry):
    from datasets.gtsrb_study import load_crop

    root = Path(study["dataset_root"])
    encoder = ByteImageEncoder(geometry.input_shape)
    arrays = {}
    for split in ("train", "validation"):
        rows = [row for row in study["records"] if row.get("split") == split]
        if not rows:
            raise ValueError(f"Frozen study has no {split} rows")
        x = np.stack([encoder.resize(load_crop(root, row)) for row in rows])
        x = x.astype(np.float32) / np.float32(256.0)
        y = np.asarray([row["class_id"] for row in rows], dtype=np.int64)
        arrays[split] = (x, y)
    return arrays


def _class_weights(labels: np.ndarray, classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=classes)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training split")
    return {index: float(len(labels) / (classes * count)) for index, count in enumerate(counts)}


def _clean_model(tf, plan: CandidatePlan):
    inputs = tf.keras.Input(shape=plan.geometry.input_shape, name="input")
    hidden = tf.keras.layers.Conv2D(
        plan.geometry.kernel_shape[-1], plan.geometry.kernel_shape[:2],
        strides=plan.geometry.strides, padding=plan.geometry.padding.lower(),
        activation="relu", name="conv",
    )(inputs)
    flat = tf.keras.layers.Flatten(name="flatten")(hidden)
    logits = tf.keras.layers.Dense(plan.classes, name="dense")(flat)
    return tf.keras.Model(inputs, logits, name=plan.candidate_id)


def _train_candidate(tf, plan: CandidatePlan, arrays, config, output: Path) -> dict[str, Any]:
    training = config["training"]
    seed = int(training["seed"])
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    clean = _clean_model(tf, plan)
    aug_spec = training["augmentation"]
    train_inputs = tf.keras.Input(shape=plan.geometry.input_shape, name="training_input")
    augmented = tf.keras.layers.RandomTranslation(
        aug_spec["translation_fraction"], aug_spec["translation_fraction"],
        fill_mode="nearest", seed=seed, name="train_translation",
    )(train_inputs)
    augmented = tf.keras.layers.RandomContrast(
        aug_spec["contrast_factor"], seed=seed + 1, name="train_contrast",
    )(augmented)
    logits = clean(augmented)
    training_model = tf.keras.Model(train_inputs, logits, name=f"{plan.candidate_id}_training")
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=training["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=training["lr_plateau_patience"],
            min_lr=training["minimum_learning_rate"],
        ),
    ]
    started = time.monotonic()
    history = training_model.fit(
        *arrays["train"], validation_data=arrays["validation"],
        epochs=training["epochs"], batch_size=training["batch_size"],
        class_weight=_class_weights(arrays["train"][1], plan.classes),
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.monotonic() - started
    validation_logits = np.asarray(clean.predict(
        arrays["validation"][0], batch_size=training["batch_size"], verbose=0,
    ))
    validation_accuracy = float(np.mean(
        np.argmax(validation_logits, axis=1) == arrays["validation"][1]
    ))
    validation_loss = float(tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            arrays["validation"][1], validation_logits, from_logits=True,
        )
    ).numpy())
    conv = clean.get_layer("conv")
    dense = clean.get_layer("dense")
    ck, cb = conv.get_weights()
    dk, db = dense.get_weights()
    restricted = RestrictedCNN(
        plan.geometry, ck, cb, dk, db, max_affine_entries=plan.max_affine_entries,
    )
    # Materializing this representation is the compatibility gate for later
    # DeepPoly/MILP preimage construction.
    restricted.affine_parameters()
    lowered = restricted.as_deep_model()
    lowered_logits = np.asarray(lowered(
        arrays["validation"][0].reshape(len(arrays["validation"][0]), -1), training=False,
    ))
    lowering_max_abs_error = float(np.max(np.abs(lowered_logits - validation_logits)))
    lowering_predictions_equal = bool(np.array_equal(
        np.argmax(lowered_logits, axis=1), np.argmax(validation_logits, axis=1),
    ))
    if (not lowering_predictions_equal
            or not np.allclose(lowered_logits, validation_logits, rtol=1e-5, atol=1e-5)):
        raise ValueError(
            f"Candidate {plan.candidate_id} failed empirical float lowering parity: "
            f"max_abs_error={lowering_max_abs_error}"
        )

    output.mkdir(parents=True, exist_ok=False)
    params = output / "model.npz"
    np.savez(params, conv_kernel=ck, conv_bias=cb, dense_kernel=dk, dense_bias=db)
    clean.save(output / "source.keras")
    serial_history = {key: [float(value) for value in values] for key, values in history.history.items()}
    write_new_json(output / "history.json", serial_history)
    result = {
        "schema": "ssv_source_model_candidate_v1",
        "candidate_id": plan.candidate_id,
        "status": "COMPLETED",
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "epochs_completed": len(serial_history["loss"]),
        "best_validation_loss": float(min(serial_history["val_loss"])),
        "best_validation_accuracy_observed": float(max(serial_history["val_accuracy"])),
        "training_elapsed_seconds": float(elapsed),
        "geometry": asdict(plan.geometry),
        "shared_parameter_count": plan.shared_parameter_count,
        "expanded_affine_parameter_count": plan.expanded_affine_parameter_count,
        "affine_lowering_entries": plan.affine_lowering_entries,
        "hidden_relu_count": plan.hidden_relu_count,
        "float_lowering_max_abs_error": lowering_max_abs_error,
        "float_lowering_predictions_equal": lowering_predictions_equal,
        "float_lowering_claim": "tested_numerical_equivalence_not_IEEE_proof",
        "model_path": str(params.resolve()),
        "model_sha256": _sha256(params),
        "keras_model_path": str((output / "source.keras").resolve()),
        "test_status": "NOT_EVALUATED",
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
    }
    write_new_json(output / "candidate_summary.json", result)
    return result


def search(config_path: Path, output: Path) -> dict[str, Any]:
    import tensorflow as tf

    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plans = validate_search_config(config)
    base_path = Path(config["base_study"])
    if not base_path.is_absolute():
        base_path = (config_path.parents[1] / base_path).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("schema") != "ssv_restricted_cnn_v1":
        raise ValueError("base_study is not a frozen restricted-CNN study")
    records = base.get("records", [])
    if any(row.get("split") not in {"train", "validation", "test"} for row in records):
        raise ValueError("Invalid split in frozen base study")
    if len({row.get("id") for row in records}) != len(records):
        raise ValueError("Frozen base study contains duplicate records")
    if config["training"].get("validation_fraction") != 0.2:
        raise ValueError("Search must preserve the base study's 0.2 track-safe validation split")
    base_training = base.get("config", {}).get("training", {})
    if base_training.get("seed") != 2026 or base_training.get("validation_fraction") != 0.2:
        raise ValueError("Frozen base study does not carry the required track-safe split identity")
    output.mkdir(parents=True, exist_ok=False)

    results = []
    loaded = {}
    for plan in plans:
        shape = plan.geometry.input_shape
        if shape not in loaded:
            loaded[shape] = _load_search_arrays(base, plan.geometry)
        results.append(_train_candidate(tf, plan, loaded[shape], config, output / plan.candidate_id))
    selected = select_candidate(results, config["selection"]["minimum_validation_accuracy"])
    status = "SELECTED" if selected is not None else "NO_CANDIDATE_MET_THRESHOLD"
    summary = {
        "schema": config["schema"],
        "status": status,
        "selection_rule": config["selection"]["rule"],
        "minimum_validation_accuracy": config["selection"]["minimum_validation_accuracy"],
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_summary": selected,
        "candidates": results,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_study_path": str(base_path),
        "base_study_sha256": _sha256(base_path),
        "dataset_root": base["dataset_root"],
        "split_counts": {split: sum(row["split"] == split for row in records)
                         for split in ("train", "validation", "test")},
        "selection_data": "track_safe_validation_only",
        "test_status": "NOT_EVALUATED",
        "proof_status": "NOT_RUN",
        "android_status": "NOT_MEASURED",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
    }
    write_new_json(output / "search_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/sign_source_model_search.json"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.check_config:
        plans = validate_search_config(config)
        print(json.dumps({"configuration_valid": True, "candidates": [asdict(plan) for plan in plans],
                          "test_status": "NOT_EVALUATED"}, indent=2))
        return
    result = search(args.config, args.output)
    print(json.dumps({"status": result["status"],
                      "selected_candidate_id": result["selected_candidate_id"],
                      "summary": str((args.output / "search_summary.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
