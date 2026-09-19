"""Full-test host accuracy and layer parity; not Android measurements."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source, compile_c_qnn_shared_library
from backends.fixed_point import FixedPointNetwork, LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from models.restricted_conv import RestrictedCNN
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.run_ssv_cnn_gate import write_new_json


def evaluate(study, region, region_dir, output):
    geom = validate_config(study["config"])
    if not region["byte_crop_property_verified"]:
        raise ValueError("A selected byte-crop certificate is required")
    if sha256(study["model_path"]) != study["model_sha256"] or region["model_sha256"] != study["model_sha256"]:
        raise ValueError("Model identity mismatch")
    if sha256(region_dir / "qnn.c") != region["generated_source_sha256"]:
        raise ValueError("Certified source identity mismatch")
    rows = [r for r in study["records"] if r["split"] == "test"]
    if not rows:
        raise ValueError("No test images")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(study["model_path"], allow_pickle=False) as weights:
        cnn = RestrictedCNN(geom, **dict(weights), max_affine_entries=study["config"]["model"]["max_affine_entries"])
    variants = {"ordinary_uniform_Q16_F8": [LayerQuantizationSpec(16, 7, 8)] * 2,
                "preqbmc_selected": [LayerQuantizationSpec(**s) for s in region["qif"]]}
    results = []
    for name, specs in variants.items():
        directory = output / name
        directory.mkdir()
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
        with (directory / "device_vectors.jsonl").open("x") as vectors:
            for row in rows:
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
                vectors.write(json.dumps({"image_id": row["id"], "image_sha256": row["sha256"],
                              "shape": list(image.shape), "encoded_input": input_int.tolist(),
                              "hidden": hidden.tolist(), "logits": expected.tolist(), "label": row["class_id"]}) + "\n")
        results.append({"method": name, "n_test_images": len(rows), "host_c_accuracy": correct / len(rows),
                        "python_c_exact_layer_rate": 1 - mismatch / len(rows),
                        "source_sha256": sha256(source), "library_sha256": sha256(library_path),
                        "actual_library_size_bytes": library_path.stat().st_size,
                        "actual_parameter_array_bytes": sum(l.weights_int.nbytes + l.biases_int.nbytes for l in network.layers),
                        "nominal_packed_storage": "NOT_IMPLEMENTED", "mismatches": mismatch,
                        "optimization_mismatches": optimization_mismatches,
                        "O0_library_sha256": sha256(o0_path),
                        "certification_scope": region["run_id"] if name == "preqbmc_selected" else "UNCERTIFIED_BASELINE"})
    report = {"source_float32_test_accuracy": study["source_float32_test_accuracy"], "methods": results,
              "compile_flags": ["-shared", "-fPIC", "-O2"], "android_parity": "NOT_MEASURED",
              "additional_parity_compile_flags": ["-shared", "-fPIC", "-O0"],
              "compiler_correctness": "TRUST_ASSUMPTION_NOT_PROVED_BY_PARITY",
              "android_performance": "NOT_MEASURED", "power": "NOT_MEASURED",
              "all_host_parity_passed": all(r["mismatches"] == 0 for r in results)}
    write_new_json(output / "host_quality.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--region", type=Path, required=True, help="region_summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.study.read_text()), json.loads(args.region.read_text()), args.region.parent, args.output)
    raise SystemExit(0 if report["all_host_parity_passed"] else 1)


if __name__ == "__main__":
    main()
