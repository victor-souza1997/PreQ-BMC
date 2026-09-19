"""Train the restricted CNN and freeze a solver-outcome-blind GTSRB study manifest.

Consumes locally extracted official archives. Does not download data, accept
dataset terms, run proofs, or overwrite earlier experiment artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np

from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import official_records, load_crop, select_regions, sha256
from models.restricted_conv import ConvGeometry, RestrictedCNN
from scripts.run_ssv_cnn_gate import write_new_json


def validate_config(config):
    if config.get("schema") != "ssv_restricted_cnn_v1":
        raise ValueError("This command accepts only ssv_restricted_cnn_v1 study specifications")
    pre = config["preprocessing"]
    if (pre["normalization_divisor"] != 256 or pre["resize"] != "nearest_floor_top_left"
            or pre["tie_break"] != "lowest_class_index" or pre["input_domain"] != "uint8_RGB_cropped_sign"
            or pre["shape"][-1] != 3):
        raise ValueError("Unsupported encoder semantics")
    spec = config["model"]
    if spec["type"] != "Conv2D_ReLU_Flatten_Dense" or spec["classes"] != 43:
        raise ValueError("Expected the restricted 43-class GTSRB architecture")
    geom = ConvGeometry(tuple(pre["shape"]), (*spec["kernel_size"], 3, spec["filters"]),
                        tuple(spec["strides"]), spec["padding"])
    if np.prod(geom.input_shape) * np.prod(geom.output_shape) > spec["max_affine_entries"]:
        raise ValueError("Model exceeds the affine-lowering feasibility budget")
    v = config["verification"]
    expected = {"solver": "cbc", "esbmc_jobs": 1, "preimg_mode": "milp", "harness_scope": "layer",
                "error_budget_mode": "derived", "enforce_contract_chaining": True,
                "tighten_verified_bounds": True, "unsound_contract_tolerance": False,
                "e2e_fallback": False}
    if any(v.get(key) != value for key, value in expected.items()) or not 8 <= v["bit_lb"] <= v["bit_ub"] <= 16:
        raise ValueError("Study requires derived preimages, strict chaining and input F >= 8")
    if config["selection"]["type"] != "clean_margin_tertiles" or not config["selection"]["freeze_before_verification"]:
        raise ValueError("Outcome-blind fixed selection is required")
    offset = config["selection"].get("rank_offset_per_stratum", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Selection rank offset must be a nonnegative integer")
    if config["training"]["seed"] != 2026 or config["training"]["validation_fraction"] != 0.2:
        raise ValueError("This manifest schema pins the track split to seed=2026 and fraction=0.2")
    if any(type(b) is not int or b < 0 for b in config["block_sizes"]) or any(float(e) < 0 for e in config["epsilon_raw_bytes"]):
        raise ValueError("Invalid experiment decomposition or radius")
    return geom


def prepare(config_path, terms_record):
    import tensorflow as tf
    config = json.loads(Path(config_path).read_text())
    geom = validate_config(config)
    if not terms_record.strip():
        raise ValueError("Record the dataset terms and source; do not invent a license")
    root = Path(config["dataset_root"]).resolve()
    records = official_records(root)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(config["training"]["seed"])
    tf.config.experimental.enable_op_determinism()
    encoder = ByteImageEncoder(geom.input_shape)
    arrays = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in records if row["split"] == split]
        x = np.stack([encoder.resize(load_crop(root, row)) for row in rows]).astype(np.float32) / 256
        y = np.array([row["class_id"] for row in rows], dtype=np.int64)
        arrays[split] = (x, y, rows)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=geom.input_shape),
        tf.keras.layers.Conv2D(geom.kernel_shape[-1], geom.kernel_shape[:2], strides=geom.strides,
                               padding=geom.padding.lower(), activation="relu"),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(43)])
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    history = model.fit(*arrays["train"][:2], validation_data=arrays["validation"][:2],
                        epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"],
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])
    ck, cb = model.layers[0].get_weights()
    dk, db = model.layers[-1].get_weights()
    cnn = RestrictedCNN(geom, ck, cb, dk, db, config["model"]["max_affine_entries"])
    params = output / "model.npz"
    np.savez(params, conv_kernel=ck, conv_bias=cb, dense_kernel=dk, dense_bias=db)
    model.save(output / "source.keras")
    test_x, test_y, test_rows = arrays["test"]
    logits = np.asarray(model.predict(test_x, batch_size=128))
    lowered = cnn.as_deep_model()
    lowered_logits = np.asarray(lowered(test_x.reshape(len(test_x), -1)))
    max_gap = float(np.max(np.abs(lowered_logits - logits)))
    predictions_equal = bool(np.array_equal(np.argmax(logits, axis=1), np.argmax(lowered_logits, axis=1)))
    # This is an empirical float32 check, never an IEEE-equivalence certificate.
    if not predictions_equal or not np.allclose(logits, lowered_logits, rtol=1e-5, atol=1e-5):
        write_new_json(output / "blocked.json", {"status": "FLOAT_LOWERING_PARITY_BLOCKED", "max_abs_gap": max_gap})
        raise ValueError("Float32 lowering parity gate failed; retain evidence and stop")
    selected = select_regions(
        test_rows, logits,
        per_stratum=config["selection"]["per_stratum"],
        rank_offset_per_stratum=config["selection"].get("rank_offset_per_stratum", 0),
    )
    runs = []
    for index, row in enumerate(selected):
        for eps in config["epsilon_raw_bytes"]:
            for beta in config["block_sizes"]:
                for cuts in config["margin_cuts"]:
                    runs.append({"run_id": f"image{index}_eps{eps}_beta{beta}_cuts{int(cuts)}",
                                 "sample": row, "epsilon": eps, "block_size": beta, "margin_cuts": cuts,
                                 **config["verification"]})
    write_new_json(output / "study.json", {
        "schema": config["schema"], "dataset_root": str(root), "dataset_terms": terms_record,
        "dataset_source_url": config["dataset_source_url"], "config": config,
        "config_sha256": sha256(config_path), "records": records, "geometry": asdict(geom),
        "model_path": str(params.resolve()), "model_sha256": sha256(params),
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "source_float32_test_accuracy": float(np.mean(np.argmax(logits, axis=1) == test_y)),
        "float_lowering_max_abs_error": max_gap, "float_lowering_predictions_equal": predictions_equal,
        "float_lowering_claim": "tested_numerical_equivalence_not_IEEE_proof",
        "shared_parameter_count": int(model.count_params()),
        "expanded_parameter_count": sum(int(k.size + b.size) for k, b in cnn.affine_parameters()),
        "history": history.history, "selected_images": selected, "runs": runs,
        "n_images": len(selected), "n_regions": len(selected) * len(config["epsilon_raw_bytes"]),
        "proof_status": "NOT_RUN", "android_status": "NOT_MEASURED"})
    print(output / "study.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ssv2026_gtsrb_pilot.json"))
    parser.add_argument("--dataset-terms-record", help="Source and applicable terms reviewed by the experiment author")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.check_config:
        geom = validate_config(json.loads(args.config.read_text()))
        print(json.dumps({"configuration_valid": True, "geometry": asdict(geom), "experiment_status": "NOT_RUN"}))
    else:
        if not args.dataset_terms_record:
            parser.error("--dataset-terms-record is required before using the dataset")
        prepare(args.config, args.dataset_terms_record)


if __name__ == "__main__":
    main()
