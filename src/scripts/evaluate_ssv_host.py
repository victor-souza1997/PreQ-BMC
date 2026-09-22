"""Full-test host accuracy and layer parity; not Android measurements."""
import argparse
import ctypes
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source, compile_c_qnn_shared_library
from backends.fixed_point import FixedPointNetwork, LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from models.restricted_conv import RestrictedCNN
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.run_ssv_cnn_gate import write_new_json
from models.ssv_artifact import load_artifact
from reports.ssv_quality import QualityComparison
from utils.fixed_point import quantize_int


def keras_logits(cnn, study, rows, specs=None):
    """Reconstruct the restricted float32 architecture from identity-checked weights.

    Quantized parameters here still execute in float32, with no activation
    quantization. This is a diagnostic baseline, not the deployed integer model.
    """
    import tensorflow as tf
    g = cnn.geometry
    model = tf.keras.Sequential([
        tf.keras.Input(shape=g.input_shape),
        tf.keras.layers.Conv2D(g.kernel_shape[-1], g.kernel_shape[:2], strides=g.strides,
                              padding=g.padding.lower(), activation="relu"),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(len(cnn.dense_bias))])
    params = [(cnn.conv_kernel, cnn.conv_bias), (cnn.dense_kernel, cnn.dense_bias)]
    for index, layer in enumerate((model.layers[0], model.layers[-1])):
        values = params[index]
        if specs is not None:
            spec = specs[index]
            values = [np.asarray(quantize_int(v, spec.total_bits, spec.fractional_bits),
                                 dtype=np.float32) / spec.scale_factor for v in values]
        layer.set_weights(values)
    encoder = ByteImageEncoder(g.input_shape)
    outputs = []
    for start in range(0, len(rows), 128):
        images = np.stack([encoder.resize(load_crop(study["dataset_root"], row))
                           for row in rows[start:start + 128]]).astype(np.float32) / 256
        outputs.append(np.asarray(model(images, training=False)))
    return np.concatenate(outputs)


def evaluate(study, region, region_dir, output, *, max_accuracy_drop_pp=0.0):
    geom = validate_config(study["config"])
    if not np.isfinite(max_accuracy_drop_pp) or max_accuracy_drop_pp < 0:
        raise ValueError("Accuracy drop allowance must be finite and nonnegative")
    if sha256(study["model_path"]) != study["model_sha256"] or region["model_sha256"] != study["model_sha256"]:
        raise ValueError("Model identity mismatch")
    if sha256(region_dir / "qnn.c") != region["generated_source_sha256"]:
        raise ValueError("Certified source identity mismatch")
    rows = [r for r in study["records"] if r["split"] == "test"]
    if not rows:
        raise ValueError("No test images")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate test image identities")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(study["model_path"], allow_pickle=False) as weights:
        cnn = RestrictedCNN(geom, **dict(weights), max_affine_entries=study["config"]["model"]["max_affine_entries"])
    source_logits = keras_logits(cnn, study, rows)
    variants = {"ordinary_uniform_Q16_F8": [LayerQuantizationSpec(16, 7, 8)] * 2,
                "preqbmc_selected": [LayerQuantizationSpec(**s) for s in region["qif"]]}
    results = []
    for name, specs in variants.items():
        directory = output / name
        directory.mkdir()
        quantized_float_logits = keras_logits(cnn, study, rows, specs)
        c_vs_float, qfloat_vs_float = QualityComparison(), QualityComparison()
        python_vs_float, c_vs_qfloat = QualityComparison(), QualityComparison()
        encoder = ByteImageEncoder(geom.input_shape, specs[0].fractional_bits, specs[0].total_bits)
        network = cnn.quantized(specs, input_fractional_bits=specs[0].fractional_bits, input_total_bits=specs[0].total_bits)
        source = directory / "qnn.c"
        source.write_text(generate_c_qnn_source(network) + encoder.render_c())
        if name == "preqbmc_selected" and sha256(source) != region["generated_source_sha256"]:
            raise ValueError("Re-export differs from certified C")
        library_path = compile_c_qnn_shared_library(source, directory / "qnn.so")
        ptr, byte_ptr = ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8)
        lib = ctypes.CDLL(str(library_path.resolve()))
        lib.qnn_forward_fixed.argtypes = [ptr, ptr]
        lib.qnn_encode_bytes.argtypes = [byte_ptr, ctypes.c_int, ctypes.c_int, ptr]
        prefix = directory / "hidden.c"
        prefix.write_text(generate_c_qnn_source(FixedPointNetwork(network.input_fractional_bits, network.input_total_bits, network.layers[:1])) + encoder.render_c())
        hidden_lib = ctypes.CDLL(str(compile_c_qnn_shared_library(prefix, directory / "hidden.so").resolve()))
        hidden_lib.qnn_forward_fixed.argtypes = [ptr, ptr]
        o0_path = compile_c_qnn_shared_library(source, directory / "qnn_O0.so", optimization="-O0")
        o0_lib = ctypes.CDLL(str(o0_path.resolve()))
        o0_lib.qnn_forward_fixed.argtypes = [ptr, ptr]
        o0_lib.qnn_encode_bytes.argtypes = [byte_ptr, ctypes.c_int, ctypes.c_int, ptr]
        o0_hidden = ctypes.CDLL(str(compile_c_qnn_shared_library(
            prefix, directory / "hidden_O0.so", optimization="-O0").resolve()))
        o0_hidden.qnn_forward_fixed.argtypes = [ptr, ptr]
        optimization_mismatches = {"-O0": 0, "-O2": 0}
        correct = mismatch = 0
        with (directory / "device_vectors.jsonl").open("x") as vectors, (directory / "predictions.csv").open("x", newline="") as predictions:
            writer = csv.DictWriter(predictions, fieldnames=["image_id", "label", "float_prediction",
                                    "quantized_weights_float_prediction", "python_prediction", "c_prediction"])
            writer.writeheader()
            for row_index, row in enumerate(rows):
                image = np.ascontiguousarray(load_crop(study["dataset_root"], row))
                expected_input = encoder.encode(image)
                input_int = np.zeros_like(expected_input)
                status = lib.qnn_encode_bytes(image.ctypes.data_as(byte_ptr), image.shape[0], image.shape[1], input_int.ctypes.data_as(ptr))
                if status or not np.array_equal(expected_input, input_int):
                    raise AssertionError("Host C preprocessing mismatch")
                hidden, expected = cnn.integer_reference(input_int.reshape(geom.input_shape), specs, specs[0].fractional_bits)
                hidden_c, logits = np.zeros_like(hidden), np.zeros_like(expected)
                hidden_lib.qnn_forward_fixed(input_int.ctypes.data_as(ptr), hidden_c.ctypes.data_as(ptr))
                lib.qnn_forward_fixed(input_int.ctypes.data_as(ptr), logits.ctypes.data_as(ptr))
                mismatch_o2 = not np.array_equal(hidden_c, hidden) or not np.array_equal(logits, expected)
                input_o0, hidden_o0, logits_o0 = np.zeros_like(input_int), np.zeros_like(hidden), np.zeros_like(expected)
                status_o0 = o0_lib.qnn_encode_bytes(image.ctypes.data_as(byte_ptr), image.shape[0], image.shape[1], input_o0.ctypes.data_as(ptr))
                o0_hidden.qnn_forward_fixed(input_o0.ctypes.data_as(ptr), hidden_o0.ctypes.data_as(ptr))
                o0_lib.qnn_forward_fixed(input_o0.ctypes.data_as(ptr), logits_o0.ctypes.data_as(ptr))
                mismatch_o0 = bool(status_o0 or not np.array_equal(input_o0, expected_input)
                                   or not np.array_equal(hidden_o0, hidden) or not np.array_equal(logits_o0, expected))
                optimization_mismatches["-O0"] += int(mismatch_o0)
                optimization_mismatches["-O2"] += int(mismatch_o2)
                mismatch += int(mismatch_o0 or mismatch_o2)
                correct += int(np.argmax(logits) == row["class_id"])
                float_values, qfloat_values = source_logits[row_index], quantized_float_logits[row_index]
                real_c = logits.astype(np.float64) / specs[-1].scale_factor
                real_python = expected.astype(np.float64) / specs[-1].scale_factor
                c_vs_float.add(float_values, real_c, row["class_id"])
                qfloat_vs_float.add(float_values, qfloat_values, row["class_id"])
                python_vs_float.add(float_values, real_python, row["class_id"])
                c_vs_qfloat.add(qfloat_values, real_c, row["class_id"])
                writer.writerow({"image_id": row["id"], "label": row["class_id"],
                                 "float_prediction": int(float_values.argmax()),
                                 "quantized_weights_float_prediction": int(qfloat_values.argmax()),
                                 "python_prediction": int(expected.argmax()), "c_prediction": int(logits.argmax())})
                vectors.write(json.dumps({"image_id": row["id"], "image_sha256": row["sha256"],
                              "shape": list(image.shape), "encoded_input": input_int.tolist(),
                              "hidden": hidden.tolist(), "logits": expected.tolist(), "label": row["class_id"]}) + "\n")
        results.append({"method": name, "n_test_images": len(rows), "host_c_accuracy": correct / len(rows),
                        "qif": [asdict(spec) for spec in specs],
                        "logit_error_units": "real_logits_integer_outputs_divided_by_output_scale",
                        "python_c_exact_layer_rate": 1 - mismatch / len(rows),
                        "source_sha256": sha256(source), "library_sha256": sha256(library_path),
                        "actual_library_size_bytes": library_path.stat().st_size,
                        "actual_parameter_array_bytes": sum(l.weights_int.nbytes + l.biases_int.nbytes for l in network.layers),
                        "nominal_packed_storage": "NOT_IMPLEMENTED", "mismatches": mismatch,
                        "optimization_mismatches": optimization_mismatches,
                        "O0_library_sha256": sha256(o0_path),
                        "c_vs_float": c_vs_float.summary(), "python_vs_float": python_vs_float.summary(),
                        "quantized_weights_float_vs_float": qfloat_vs_float.summary(),
                        "c_vs_quantized_weights_float": c_vs_qfloat.summary(),
                        "accuracy_retention_accepted": c_vs_float.summary()["accuracy_drop_percentage_points"] <= max_accuracy_drop_pp,
                        "certification_scope": (region["run_id"] if region.get("byte_crop_property_verified")
                                                else "UNVERIFIED_FIXED_ARTIFACT")
                                                if name == "preqbmc_selected" else "UNCERTIFIED_BASELINE"})
        print(f"{name}: {len(rows)} test images, C accuracy={correct / len(rows):.6f}, "
              f"parity mismatches={mismatch}", flush=True)
    test_identity = [{k: r[k] for k in ("id", "sha256", "class_id")} for r in rows]
    report = {"source_float32_test_accuracy": results[0]["c_vs_float"]["source_accuracy"], "methods": results,
              "original_study_source_accuracy": study.get("source_float32_test_accuracy"),
              "source_execution": "reconstructed_keras_float32_conv_relu_flatten_dense",
              "model_sha256": study["model_sha256"],
              "test_manifest_sha256": hashlib.sha256(json.dumps(test_identity, sort_keys=True).encode()).hexdigest(),
              "evaluation_population": "all_frozen_test_records_no_source_or_certificate_filter",
              "accuracy_max_drop_percentage_points": max_accuracy_drop_pp,
              "quality_status": "MEASURED", "quality_claim": "empirical_not_formal",
              "evaluator_sha256": sha256(Path(__file__)),
              "formal_status_modified": False, "android_transfer_verified": False,
              "compile_flags": ["-shared", "-fPIC", "-O2"], "android_parity": "NOT_MEASURED",
              "additional_parity_compile_flags": ["-shared", "-fPIC", "-O0"],
              "compiler_correctness": "TRUST_ASSUMPTION_NOT_PROVED_BY_PARITY",
              "android_performance": "NOT_MEASURED", "power": "NOT_MEASURED",
              "all_host_parity_passed": all(r["mismatches"] == 0 for r in results)}
    report["selected_quality_accepted"] = results[-1]["accuracy_retention_accepted"] and results[-1]["mismatches"] == 0
    write_new_json(output / "host_quality.json", report)
    with (output / "table_host_quality.csv").open("x", newline="") as handle:
        fields = ["method", "n_images", "source_accuracy", "candidate_accuracy", "accuracy_drop_percentage_points",
                  "prediction_mismatch_rate", "max_abs_logit_error", "mean_abs_logit_error",
                  "python_c_exact_layer_rate", "accuracy_retention_accepted"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, **result["c_vs_float"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--region", type=Path, help="region_summary.json; certification is not required")
    identity.add_argument("--artifact", type=Path, help="Frozen artifact.json, independently of region outcomes")
    parser.add_argument("--max-accuracy-drop-pp", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    if args.artifact:
        artifact, _ = load_artifact(args.artifact, study)
        region = {**artifact, "run_id": "frozen_artifact", "byte_crop_property_verified": False}
        region_dir = args.artifact.parent
    else:
        region, region_dir = json.loads(args.region.read_text()), args.region.parent
    report = evaluate(study, region, region_dir, args.output, max_accuracy_drop_pp=args.max_accuracy_drop_pp)
    print(args.output / "host_quality.json")
    print("Selected empirical quality: " + ("ACCEPTED" if report["selected_quality_accepted"] else "NOT_ACCEPTED"))
    raise SystemExit(0 if report["all_host_parity_passed"] and report["selected_quality_accepted"] else 1)


if __name__ == "__main__":
    main()
