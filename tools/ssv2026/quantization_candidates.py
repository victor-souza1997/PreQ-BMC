#!/usr/bin/env python3
"""Isolated validation-only int16/int8 candidates; never modify frozen src/.

Run from the repository root. No TensorFlow, training, test-set reads, certificate
reuse, or implicit ESBMC execution. See docs/ssv2026/int8_int16_candidates.md.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
# Keep small pilots from starting framework/BLAS worker pools.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

# These package initializers eagerly import training/dataset frameworks. Load
# their real submodules without executing framework-only initializers. This
# affects only this standalone process and never patches repository sources.
for package in ("models", "datasets"):
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(ROOT / "src" / package)]
        sys.modules[package] = module

import numpy as np

from backends.conv_fixed_point import (
    QuantizedConv2D, forward_conv_fixed_point_single, generate_conv_qnn_source,
    quantize_restricted_sequential,
)
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN
from verification.arith_kernel import render_arith_kernel
from verification.conv_interval_fast import _im2col

DEFAULT_CONFIG = ROOT / "experiments/sign_conv_native_validation3652_box_proof.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))
            if "tests" not in p.parts}


def load_source(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if config.get("schema") != "ssv_conv_native_pilot_v1" or config.get("split") != "validation":
        raise ValueError("Require the validation-only conv-native configuration")
    search_path = ROOT / config["source_search_output"] / "search_summary.json"
    search = read_json(search_path)
    if search.get("status") != "SELECTED":
        raise ValueError("No selected frozen source model")
    selected = search["selected_candidate_summary"]
    params = Path(selected["folded_model_path"])
    study_path = Path(search["base_study_path"])
    if sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Folded-model hash mismatch")
    if sha256(study_path) != search["base_study_sha256"]:
        raise ValueError("Dataset-study hash mismatch")
    geometries = tuple(ConvGeometry(
        tuple(g["input_shape"]), tuple(g["kernel_shape"]), tuple(g["strides"]), g["padding"]
    ) for g in selected["geometries"])
    with np.load(params, allow_pickle=False) as arrays:
        model = RestrictedSequentialCNN(
            geometries,
            tuple(arrays[f"conv_{i}_kernel"].copy() for i in range(len(geometries))),
            tuple(arrays[f"conv_{i}_bias"].copy() for i in range(len(geometries))),
            tuple(arrays[f"dense_{i}_kernel"].copy() for i in range(len(selected["dense_hidden"]) + 1)),
            tuple(arrays[f"dense_{i}_bias"].copy() for i in range(len(selected["dense_hidden"]) + 1)),
            max_affine_entries=max(selected["affine_entries_per_layer"]),
        )
    study = read_json(study_path)
    rows = [r for r in study["records"] if r.get("split") == "validation"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Missing/duplicate validation rows")
    # Never decode a held-out test image or consume its evaluation report.
    return config, model, study, rows, {
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "search_path": str(search_path), "search_sha256": sha256(search_path),
        "folded_model_path": str(params), "folded_model_sha256": sha256(params),
        "study_path": str(study_path), "study_sha256": sha256(study_path),
    }


def make_network(model, bits, fractional_bits, resize_mode):
    spec = LayerQuantizationSpec(bits, bits - fractional_bits - 1, fractional_bits)
    encoder = ByteImageEncoder(model.input_shape, fractional_bits, bits, resize_mode)
    network = quantize_restricted_sequential(model, [spec] * (
        len(model.conv_kernels) + len(model.dense_kernels)),
        input_total_bits=bits, input_fractional_bits=fractional_bits)
    return network, encoder


def compact_source(source, network, bits):
    """Narrow only parameter arrays and internal buffers; retain the int64 ABI.

    Legal forward inputs must fit the declared signed input format. Encoder
    outputs always do. MAC operands are explicitly widened by mac_i128 before
    multiplication. Casts occur only after clamping (or on legal input).
    """
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for layer in network.layers:
        weights = layer.kernel_int if isinstance(layer, QuantizedConv2D) else layer.weights_int
        if any(np.any(a < low) or np.any(a > high) for a in (weights, layer.bias_int)):
            raise ValueError("A parameter does not fit compact storage")
    source, count = re.subn(
        r"static const int64_t (LAYER_\d+_(?:KERNEL|WEIGHTS|BIAS)\[)",
        rf"static const int{bits}_t \1", source)
    if count != 2 * len(network.layers):
        raise ValueError("Generator parameter layout changed")
    source, count = re.subn(r"int64_t (buffer_[01]\[\d+\])", rf"int{bits}_t \1", source)
    if count != 2:
        raise ValueError("Generator buffer layout changed")
    source, count = re.subn(
        r"(buffer_[01]\[[^\n]+\] = )\(int64_t\)value;", rf"\1(int{bits}_t)value;", source)
    if count != len(network.layers):
        raise ValueError("Generator output cast layout changed")
    old = "buffer_0[i] = input[i];"
    if source.count(old) != 1:
        raise ValueError("Generator input layout changed")
    return source.replace(old, f"buffer_0[i] = (int{bits}_t)input[i];")


def compile_library(source, output):
    command = ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared",
               "-fPIC", str(source), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command


def build(args):
    config, model, study, rows, provenance = load_source(args.config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = source_hashes()
    for bits, frac in ((16, 8), (8, args.int8_fractional_bits)):
        folder = out / f"int{bits}_f{frac}"
        folder.mkdir()
        network, encoder = make_network(model, bits, frac, config["resize_mode"])
        wide = generate_conv_qnn_source(network, encoder_source=encoder.render_c())
        if bits == 16 and hashlib.sha256(wide.encode()).hexdigest() != config[
            "expected_deployment_source_sha256"
        ]:
            raise ValueError("int16 reference is not the pinned Q7.8 deployment")
        compact = compact_source(wide, network, bits)
        (folder / "qnn_reference.c").write_text(wide)
        (folder / "qnn_compact.c").write_text(compact)
        commands = [compile_library(folder / f"qnn_{kind}.c", folder / f"qnn_{kind}.so")
                    for kind in ("reference", "compact")]
        arrays = {}
        parameter_count = 0
        for i, layer in enumerate(network.layers):
            weights = layer.kernel_int if isinstance(layer, QuantizedConv2D) else layer.weights_int
            arrays[f"layer_{i}_weights"] = weights.astype(f"int{bits}")
            arrays[f"layer_{i}_bias"] = layer.bias_int.astype(f"int{bits}")
            parameter_count += weights.size + layer.bias_int.size
        np.savez(folder / "quantized_parameters.npz", **arrays)
        floats = (*model.conv_kernels, *model.conv_biases, *model.dense_kernels, *model.dense_biases)
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        clipped = sum(int(np.count_nonzero((a < lo / (1 << frac)) |
                                          (a > hi / (1 << frac)))) for a in floats)
        maximum = max(math.prod(network.input_shape), *(l.output_size for l in network.layers))
        manifest = {
            "schema": "ssv_quantization_candidate_v1", "bits": bits, "fractional_bits": frac,
            "qif": [asdict(l.spec) for l in network.layers], "resize_mode": config["resize_mode"],
            "input_shape": list(network.input_shape), "classes": network.output_size,
            "provenance": provenance, "src_hashes": pins, "tool_sha256": sha256(__file__),
            "files": {p.name: sha256(p) for p in sorted(folder.iterdir())},
            "compile_commands": commands, "parameter_values": parameter_count,
            "reference_parameter_bytes": int(parameter_count * 8),
            "compact_parameter_bytes": int(parameter_count * bits // 8),
            "reference_internal_buffer_bytes": int(2 * maximum * 8),
            "compact_internal_buffer_bytes": int(2 * maximum * bits // 8),
            "clipped_float_parameters": clipped,
            "accumulator_absolute_envelopes": list(network.accumulator_envelopes),
            "arithmetic": "signed global Q/I/F, half-away rounding, saturation, __int128 MAC",
            "input_precondition": "qnn_forward_fixed inputs fit the signed input format; use qnn_encode_bytes",
            "semantics_preserving_relative_to_original": bits == 16,
            "validation_images_available": len(rows), "quality_status": "NOT_RUN",
            "proof_status": "NOT_RUN", "certified": False,
            "crown_v2_compatible": False,
            "generic_interval_pipeline_compatible": True,
            "trust_boundary": "Python model loading, quantization, C transformation and composition remain trusted",
        }
        write_json(folder / "manifest.json", manifest)
        print(json.dumps({"candidate": str(folder), "bits": bits, "fractional_bits": frac,
                          "clipped_float_parameters": clipped}), flush=True)
    if source_hashes() != pins:
        raise RuntimeError("Frozen sources changed while building")


def load_candidate(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema") != "ssv_quantization_candidate_v1":
        raise ValueError("Unsupported candidate schema")
    if manifest["src_hashes"] != source_hashes():
        raise ValueError("Pipeline sources changed; rebuild candidate under new pins")
    for name, digest in manifest["files"].items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Candidate file hash mismatch: {name}")
    config, model, study, rows, provenance = load_source(manifest["provenance"]["config_path"])
    if provenance != manifest["provenance"]:
        raise ValueError("Candidate provenance changed")
    network, encoder = make_network(model, manifest["bits"], manifest["fractional_bits"],
                                    manifest["resize_mode"])
    wide = generate_conv_qnn_source(network, encoder_source=encoder.render_c())
    if wide != (folder / "qnn_reference.c").read_text() or compact_source(
        wide, network, manifest["bits"]
    ) != (folder / "qnn_compact.c").read_text():
        raise ValueError("Candidate code is not the declared deterministic export")
    with np.load(folder / "quantized_parameters.npz", allow_pickle=False) as saved:
        expected = {}
        for i, layer in enumerate(network.layers):
            expected[f"layer_{i}_weights"] = (layer.kernel_int if isinstance(
                layer, QuantizedConv2D) else layer.weights_int)
            expected[f"layer_{i}_bias"] = layer.bias_int
        if set(saved.files) != set(expected) or any(not np.array_equal(saved[k], v)
                                                   for k, v in expected.items()):
            raise ValueError("Serialized quantized parameters differ")
    return folder, manifest, network, encoder, study, rows


class CNetwork:
    def __init__(self, path, network):
        self.network = network
        self.lib = ctypes.CDLL(str(path))
        self.ptr = ctypes.POINTER(ctypes.c_int64)
        self.lib.qnn_forward_fixed.argtypes = [self.ptr, self.ptr]
        self.lib.qnn_forward_fixed.restype = None
        self.lib.qnn_encode_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
                                             ctypes.c_int, self.ptr]
        self.lib.qnn_encode_bytes.restype = ctypes.c_int

    def encode(self, image):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        encoded = np.empty(math.prod(self.network.input_shape), dtype=np.int64)
        rc = self.lib.qnn_encode_bytes(image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                      image.shape[0], image.shape[1], encoded.ctypes.data_as(self.ptr))
        if rc != 0:
            raise RuntimeError("C encoder rejected a valid crop")
        return encoded

    def forward(self, encoded):
        encoded = np.ascontiguousarray(encoded, dtype=np.int64)
        output = np.empty(self.network.output_size, dtype=np.int64)
        self.lib.qnn_forward_fixed(encoded.ctypes.data_as(self.ptr), output.ctypes.data_as(self.ptr))
        return output


def numpy_reference(network, encoded):
    """Independent vectorized integer reference; enforce int64 MAC headroom."""
    if max(network.accumulator_envelopes) >= (1 << 62):
        raise OverflowError("Reference MAC headroom exceeded")
    value = np.asarray(encoded, dtype=np.int64).reshape(network.input_shape)
    for layer in network.layers:
        if isinstance(layer, QuantizedConv2D):
            acc = _im2col(value, layer) @ layer.kernel_int.reshape(-1, layer.geometry.kernel_shape[-1])
            shape = layer.geometry.output_shape
        else:
            acc = value.reshape(-1) @ layer.weights_int.T
            shape = (layer.output_size,)
        den = 1 << layer.input_fractional_bits
        rounded = np.where(acc < 0, -((np.abs(acc) + den // 2) // den),
                           (np.abs(acc) + den // 2) // den)
        value = np.clip(rounded + layer.bias_int, -(1 << (layer.spec.total_bits - 1)),
                        (1 << (layer.spec.total_bits - 1)) - 1)
        if layer.apply_relu:
            value = np.maximum(value, 0)
        value = value.reshape(shape)
    return value.reshape(-1)


def sanitizer_check(folder, network, manifest, out):
    bits = manifest["bits"]
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    source = f'''#include <assert.h>
#include "{folder / 'qnn_compact.c'}"
int main(void) {{
    int64_t input[{math.prod(network.input_shape)}], output[{network.output_size}];
    for (int trial = 0; trial < 12; ++trial) {{
        for (int i = 0; i < {math.prod(network.input_shape)}; ++i)
            input[i] = trial == 0 ? {low} : trial == 1 ? {high} :
                       {low} + ((i * 37 + trial * 131) % {1 << bits});
        qnn_forward_fixed(input, output);
        for (int i = 0; i < {network.output_size}; ++i)
            assert(output[i] >= {low} && output[i] <= {high});
    }}
    return 0;
}}
'''
    path = out / "sanitizer_driver.c"
    path.write_text(source)
    command = ["gcc", "-std=c11", "-O1", "-fsanitize=undefined,signed-integer-overflow",
               "-fno-sanitize-recover=all", str(path), "-o", str(out / "sanitizer_driver")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(out / "sanitizer_driver")], capture_output=True, text=True, timeout=60)
    (out / "sanitizer.stdout.log").write_text(result.stdout)
    (out / "sanitizer.stderr.log").write_text(result.stderr)
    return {"command": command, "return_code": result.returncode,
            "passed": result.returncode == 0, "measurement_not_proof": True}


def validate(args):
    folder, manifest, network, encoder, study, all_rows = load_candidate(args.artifact)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = all_rows if args.limit == 0 else all_rows[:args.limit]
    compact = CNetwork(folder / "qnn_compact.so", network)
    reference = CNetwork(folder / "qnn_reference.so", network)
    correct = mismatches = encoder_mismatches = 0
    class_total = [0] * network.output_size
    class_correct = [0] * network.output_size
    parity_count = 0
    bigint_count = 0
    start = time.monotonic()
    rng = np.random.default_rng(20260927)
    edge_images = [np.full((32, 32, 3), v, np.uint8) for v in (0, 1, 127, 128, 254, 255)]
    edge_images += [rng.integers(0, 256, size=shape, dtype=np.uint8)
                    for shape in ((1, 1, 3), (17, 39, 3), (33, 31, 3))]
    for image in edge_images:
        encoded = encoder.encode(image)
        if not np.array_equal(compact.encode(image), encoded):
            encoder_mismatches += 1
        actual = compact.forward(encoded)
        if not np.array_equal(actual, reference.forward(encoded)) or not np.array_equal(
            actual, numpy_reference(network, encoded)
        ):
            mismatches += 1
        parity_count += 1
    with (out / "predictions.jsonl").open("x") as stream:
        for index, row in enumerate(rows):
            image = load_crop(study["dataset_root"], row)
            encoded = encoder.encode(image)
            if not np.array_equal(compact.encode(image), encoded):
                encoder_mismatches += 1
            actual = compact.forward(encoded)
            if not np.array_equal(actual, reference.forward(encoded)):
                mismatches += 1
            if index < args.reference_images:
                if not np.array_equal(actual, numpy_reference(network, encoded)):
                    mismatches += 1
                parity_count += 1
            if index < args.bigint_images:
                if not np.array_equal(actual, forward_conv_fixed_point_single(network, encoded)):
                    mismatches += 1
                bigint_count += 1
            predicted, label = int(np.argmax(actual)), int(row["class_id"])
            correct += int(predicted == label)
            class_total[label] += 1
            class_correct[label] += int(predicted == label)
            stream.write(json.dumps({"id": row["id"], "sha256": row["sha256"],
                                      "label": label, "prediction": predicted}) + "\n")
            if (index + 1) % 500 == 0:
                print(f"{folder.name}: {index + 1}/{len(rows)} validation images", flush=True)
    sanitizer = sanitizer_check(folder, network, manifest, out)
    accuracy = correct / len(rows)
    complete = len(rows) == len(all_rows)
    integrity_ok = mismatches == encoder_mismatches == 0 and sanitizer["passed"]
    report = {
        "schema": "ssv_quantization_validation_v1", "claim": "MEASURED", "split": "validation",
        "artifact": str(folder), "manifest_sha256": sha256(folder / "manifest.json"),
        "n_images": len(rows), "all_validation_images": len(all_rows), "full_validation": complete,
        "correct": correct, "accuracy": accuracy, "minimum_accuracy_gate": args.minimum_accuracy,
        "accuracy_gate_passed": complete and accuracy >= args.minimum_accuracy,
        "c_reference_compact_mismatches": mismatches,
        "python_c_encoder_mismatches": encoder_mismatches,
        "numpy_reference_images": parity_count, "python_bigint_reference_images": bigint_count,
        "per_class_accuracy": [c / n if n else None for c, n in zip(class_correct, class_total)],
        "class_counts": class_total, "sanitizer": sanitizer,
        "integrity_checks_passed": integrity_ok,
        "candidate_quality_eligible": complete and accuracy >= args.minimum_accuracy and integrity_ok,
        "elapsed_seconds": time.monotonic() - start,
        "proof_status": "NOT_RUN", "certified": False,
        "predictions_sha256": sha256(out / "predictions.jsonl"),
    }
    write_json(out / "validation_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if integrity_ok else 2


def storage_lemma(bits, frac):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return f'''#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
int64_t nondet_int64_t(void);
__int128 nondet_i128(void);
#define QNN_ASSERT(c,m) __ESBMC_assert((c),(m))
{render_arith_kernel()}
int main(void) {{
    int64_t x = nondet_int64_t(), w = nondet_int64_t();
    __ESBMC_assume(x >= {lo} && x <= {hi});
    __ESBMC_assume(w >= {lo} && w <= {hi});
    int{bits}_t sx = (int{bits}_t)x, sw = (int{bits}_t)w;
    __ESBMC_assert((int64_t)sx == x && (int64_t)sw == w, "legal storage cast identity");
    /* Unchanged mac_i128 multiplies these widened operands. Prove their
       equality directly; storage conversion introduces no new multiplier. */
    __ESBMC_assert((__int128)sw == (__int128)w && (__int128)sx == (__int128)x,
                   "widened operand identity");
    /* Any pre-clamp value: this covers all rounding/bias results in Q{bits}/F{frac},
       without asking a storage lemma to reprove nonlinear division. */
    __int128 v = clamp_to_signed_range_i128(nondet_i128(), {bits});
    int{bits}_t s = (int{bits}_t)v;
    __ESBMC_assert((__int128)s == v, "clamped storage cast identity");
    return 0;
}}
'''


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("Cannot measure available memory")


def classify_result(text, return_code, peak_memory):
    from verification.esbmc import ESBMCRunner
    status = "TIMEOUT" if "timed out" in text.lower() else ESBMCRunner._classify_status(
        text, return_code, peak_memory_bytes=peak_memory, memlimit="2g")
    # Never accept a success marker from a subsequently aborted process.
    return "UNKNOWN" if status == "VERIFIED" and return_code != 0 else status


def run_obligation(path, timeout, min_available):
    from verification.esbmc import ESBMCConfig, ESBMCRunner
    from verification.esbmc_install import resolve_esbmc_executable
    runner = ESBMCRunner(ESBMCConfig(timeout_seconds=timeout, memlimit="2g"))
    command = list(runner.build_command(path, runner.infer_unwind(path.read_text()), "paper-z3"))
    command.append("--overflow-check")
    available = available_gib()
    if available < min_available:
        return {"status": "NOT_RUN_LOW_MEMORY", "available_gib": available, "command": command}
    if not resolve_esbmc_executable():
        return {"status": "NOT_RUN_ESBMC_MISSING", "command": command}
    start = time.monotonic()
    status = None
    peak_memory = 0
    with Path(str(path) + ".stdout.log").open("x") as stdout, Path(str(path) + ".stderr.log").open("x") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        while proc.poll() is None:
            memory, _, _ = ESBMCRunner._process_tree_resources(proc.pid)
            peak_memory = max(peak_memory, memory or 0)
            if available_gib() < 6 or time.monotonic() - start > timeout + 5:
                status = "UNKNOWN_LOW_MEMORY" if available_gib() < 6 else "TIMEOUT"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        proc.wait()
    text = Path(str(path) + ".stdout.log").read_text() + Path(str(path) + ".stderr.log").read_text()
    if status is None:
        status = classify_result(text, proc.returncode, peak_memory)
    return {"status": status, "return_code": proc.returncode, "command": command,
            "elapsed_seconds": time.monotonic() - start, "available_gib_at_launch": available,
            "source_sha256": sha256(path), "peak_process_tree_rss_bytes": peak_memory}


def direct_harness(folder, network, byte_low, byte_high, target):
    """Verify the actual compact C on a raw resized byte box, not a surrogate."""
    arr = lambda values: "{" + ",".join(str(int(v)) for v in values.flat) + "}"
    count = math.prod(network.input_shape)
    maximum = max(count, *(layer.output_size for layer in network.layers))
    h, w, _ = network.input_shape
    return f'''#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
void __ESBMC_assume(_Bool);
uint8_t nondet_uchar(void);
#define QNN_ASSERT(c,m) __ESBMC_assert((c),(m))
#include "{folder / 'qnn_compact.c'}"
#define INVARIANT_SIZE {maximum}
int main(void) {{
    static const uint8_t low[{count}] = {arr(byte_low)};
    static const uint8_t high[{count}] = {arr(byte_high)};
    uint8_t raw[{count}];
    int64_t input[{count}], output[{network.output_size}];
    for (int i = 0; i < {count}; ++i) {{
        raw[i] = nondet_uchar();
        __ESBMC_assume(raw[i] >= low[i] && raw[i] <= high[i]);
    }}
    __ESBMC_assert(qnn_encode_bytes(raw, {h}, {w}, input) == 0, "encoder accepts resized crop");
    qnn_forward_fixed(input, output);
    for (int c = 0; c < {network.output_size}; ++c)
        if (c != {target}) __ESBMC_assert(output[{target}] > output[c], "strict class margin");
    return 0;
}}
'''


def verify(args):
    folder, manifest, network, encoder, study, rows = load_candidate(args.artifact)
    if args.complete:
        quality = read_json(args.validation_report)
        if (not quality.get("candidate_quality_eligible")
                or quality.get("manifest_sha256") != sha256(folder / "manifest.json")
                or quality.get("split") != "validation"
                or quality.get("n_images") != len(rows)
                or quality.get("accuracy", 0) < 0.90):
            raise ValueError("Complete run requires eligible full validation at >=90% for this artifact")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    matches = [r for r in rows if r["id"] == args.sample_id]
    if len(matches) != 1:
        raise ValueError("Sample must identify exactly one frozen validation image")
    row = matches[0]
    image = load_crop(study["dataset_root"], row)
    low, high = encoder.box(image, 1)
    center = encoder.encode(image)
    from verification.conv_contracts import IntegerInvariant, propagate_interval
    from verification.conv_proof import render_byte_encoding_contract
    from verification.interval_lemmas import render_conv_interval_certificate
    invariant = IntegerInvariant.from_arrays(low, high, network.input_shape, "exact_byte_box")
    certificate = propagate_interval(network.layers[0], invariant, layer_index=0)
    storage = storage_lemma(manifest["bits"], manifest["fractional_bits"])
    sources = {
        "storage_cast.c": (storage, "VERIFIED"),
        "storage_negative_control.c": (storage.replace(
            '(__int128)s == v, "clamped storage cast identity"',
            '(__int128)s == v + 1, "clamped storage cast identity"'), "FAILED"),
        "first_conv_interval.c": (render_conv_interval_certificate(
            network.layers[0], certificate, range(min(4, network.layers[0].output_size))), "VERIFIED"),
    }
    byte_low, byte_high = encoder.byte_box(image, 1)
    # Symbolic monotonicity covers every byte; replay a small endpoint sample.
    encoding_invariant = IntegerInvariant.from_arrays(low[:4], high[:4], (4,))
    sources["byte_encoder.c"] = (render_byte_encoding_contract(
        byte_low.reshape(-1)[:4], byte_high.reshape(-1)[:4], encoding_invariant,
        fractional_bits=manifest["fractional_bits"], total_bits=manifest["bits"]), "VERIFIED")
    records = []
    for name, (source, expected) in sources.items():
        path = out / name
        path.write_text(source)
        result = run_obligation(path, args.timeout, args.min_available_gib) if args.run_esbmc else {
            "status": "NOT_RUN", "source_sha256": sha256(path)}
        records.append({"name": name, "expected": expected, **result})
        print(json.dumps(records[-1]), flush=True)
    center_logits = CNetwork(folder / "qnn_compact.so", network).forward(center)
    smoke = {
        "schema": "ssv_quantization_proof_smoke_v1", "artifact": str(folder),
        "manifest_sha256": sha256(folder / "manifest.json"), "sample_id": row["id"],
        "sample_sha256": row["sha256"], "split": "validation", "epsilon_raw_bytes": 1,
        "center_prediction": int(np.argmax(center_logits)), "target": int(row["class_id"]),
        "records": records, "smoke_passed": all(r["status"] == r["expected"] for r in records),
        "network_status": "PARTIAL_NOT_CERTIFIED", "certified": False,
        "trust_boundary": "storage lemma is local, not a proof of the Python export transformation",
    }
    write_json(out / "proof_smoke_summary.json", smoke)
    if args.direct:
        path = out / "direct_compact.c"
        path.write_text(direct_harness(folder, network, byte_low, byte_high, row["class_id"]))
        direct = run_obligation(path, args.timeout, args.min_available_gib) if args.run_esbmc else {"status": "NOT_RUN"}
        write_json(out / "direct_compact_summary.json", {
            "artifact": str(folder), "manifest_sha256": sha256(folder / "manifest.json"),
            "compact_source_sha256": manifest["files"]["qnn_compact.c"],
            "sample_id": row["id"], "split": "validation", "epsilon_raw_bytes": 1,
            "scope": "raw_uint8_box_on_resized_crop; decoder and original-crop resize binding outside scope",
            "result": direct, "certified_for_stated_scope": direct["status"] == "VERIFIED",
            "paper_test_result": False,
        })
        print(json.dumps({"direct_compact": direct}), flush=True)
    if args.complete:
        if not args.run_esbmc or not smoke["smoke_passed"]:
            raise ValueError("Complete run requires successful ESBMC smoke checks")
        if int(np.argmax(center_logits)) != row["class_id"]:
            raise ValueError("Center misclassified; no robustness certification attempted")
        if available_gib() < args.min_available_gib:
            raise RuntimeError("Insufficient memory for complete pipeline")
        from verification.conv_proof import ConvProofConfig, ConvProofCoordinator
        from verification.esbmc import ESBMCConfig, ESBMCResult, ESBMCRunner
        class GuardedOverflowRunner(ESBMCRunner):
            def build_command(self, *positional, **keywords):
                return (*super().build_command(*positional, **keywords), "--overflow-check")

            def run_file(self, c_file, profile=None, **keywords):
                if available_gib() < args.min_available_gib:
                    return ESBMCResult("UNKNOWN", (), "", "Low-memory launch guard", -1)
                result = super().run_file(c_file, profile, **keywords)
                if result.status == "VERIFIED" and result.return_code != 0:
                    return replace(result, status="UNKNOWN")
                return result

            def _wait_with_peak_memory(self, process, timeout_seconds):
                start, peak = time.monotonic(), 0
                while process.poll() is None:
                    memory, _, _ = self._process_tree_resources(process.pid)
                    peak = max(peak, memory or 0)
                    if available_gib() < 6 or time.monotonic() - start > args.timeout + 5:
                        self._terminate_process_tree(process)
                        return (-1, peak, "linux_procfs_process_tree_rss", None, None, "unavailable", True)
                    time.sleep(0.1)
                return (process.returncode, peak, "linux_procfs_process_tree_rss", None, None, "unavailable", False)
        coordinator = ConvProofCoordinator(
            network, out / "generic_pipeline", encoder_source=encoder.render_c(),
            esbmc=ESBMCConfig(timeout_seconds=args.timeout, memlimit="2g"),
            config=ConvProofConfig(block_size=4, proof_mode="proof_carrying_interval",
                                   exact_margin_refinement=True, fail_fast=True,
                                   max_blocks_per_layer=None, jobs=1, min_available_gib=8),
        )
        coordinator.runner = GuardedOverflowRunner(coordinator.runner.config)
        coordinator.store.runner = coordinator.runner
        coordinator.store.checker_identity["extra_flags"] = ["--overflow-check"]
        if coordinator.deployment_source_sha256 != manifest["files"]["qnn_reference.c"]:
            raise ValueError("Generic pipeline deployment hash differs from candidate reference")
        byte_low, byte_high = encoder.byte_box(image, 1)
        formal = coordinator.verify(invariant, input_witness=center, target_class=row["class_id"],
                                    input_byte_low=byte_low.reshape(-1), input_byte_high=byte_high.reshape(-1))
        write_json(out / "complete_pipeline_summary.json", {
            "reference_deployment_result": formal,
            "compact_deployment_status": "NOT_CERTIFIED_EXPORT_BINDING_REQUIRED",
            "compact_certified": False,
            "note": "A complete reference proof does not automatically certify transformed compact C.",
        })
        if formal["final_status"] != "VERIFIED":
            return 2
    print(json.dumps(smoke, indent=2), flush=True)
    if args.direct and args.run_esbmc and direct["status"] != "VERIFIED":
        return 2
    return 0 if not args.run_esbmc or smoke["smoke_passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("build")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--int8-fractional-bits", type=int, choices=range(8), default=4)
    p.set_defaults(func=build)
    p = sub.add_parser("validate")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0: entire validation split; otherwise smoke only")
    p.add_argument("--reference-images", type=int, default=3)
    p.add_argument("--bigint-images", type=int, default=1)
    p.add_argument("--minimum-accuracy", type=float, default=0.90)
    p.set_defaults(func=validate)
    p = sub.add_parser("verify")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-id", default="Final_Training/Images/00012/00005_00022.ppm")
    p.add_argument("--run-esbmc", action="store_true")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--min-available-gib", type=float, default=8)
    p.add_argument("--complete", action="store_true", help="Expensive full reference pipeline, no automatic compact certificate")
    p.add_argument("--direct", action="store_true", help="Bounded monolithic check of actual compact C; may exhaust resources")
    p.add_argument("--validation-report", type=Path)
    p.set_defaults(func=verify)
    args = parser.parse_args()
    if getattr(args, "limit", 0) < 0 or getattr(args, "reference_images", 0) < 0 or getattr(args, "bigint_images", 0) < 0:
        parser.error("Image counts must be nonnegative")
    if hasattr(args, "minimum_accuracy") and not 0.90 <= args.minimum_accuracy <= 1:
        parser.error("SSV accuracy gate must be in [0.90,1]")
    if hasattr(args, "timeout") and (args.timeout <= 0 or args.min_available_gib < 8):
        parser.error("Timeout must be positive and launch memory guard at least 8 GiB")
    if getattr(args, "complete", False) and args.validation_report is None:
        parser.error("--complete requires --validation-report")
    result = args.func(args)
    return result or 0


if __name__ == "__main__":
    raise SystemExit(main())
