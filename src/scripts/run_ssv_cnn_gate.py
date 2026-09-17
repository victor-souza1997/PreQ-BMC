"""Small, real solver feasibility experiment; never reports a GTSRB certificate."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source, compile_c_qnn_shared_library
from backends.fixed_point import FixedPointNetwork, LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from models.restricted_conv import ConvGeometry, RestrictedCNN
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.invariants import exact_layer_interval


def tiny_model():
    # A nonconstant Conv/ReLU classifier with a large local margin, two outputs.
    return RestrictedCNN(ConvGeometry((2, 2, 1), (2, 2, 1, 1), padding="SAME"),
                         np.ones((2, 2, 1, 1)), np.zeros(1),
                         np.array([[1.0, -1.0]] * 4), np.array([1.0, 0.0]))


def write_new_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def classification_harness(network, low, high, target):
    # Verify the exported implementation, with no unproved layer invariants.
    low = np.asarray(low, dtype=np.int64)
    high = np.asarray(high, dtype=np.int64)
    if low.shape != high.shape or low.shape != (network.layers[0].weights_int.shape[1],) or np.any(low > high):
        raise ValueError("Invalid input box")
    classes = len(network.layers[-1].biases_int)
    if not 0 <= target < classes:
        raise ValueError("Invalid target")
    source = generate_c_qnn_source(network)
    arr = lambda xs: "{" + ",".join(str(int(x)) for x in xs) + "}"
    return "#define QNN_VERIFY_WITH_ESBMC\n" + source + f"""
extern int64_t nondet_int64_t(void);
extern void __ESBMC_assume(_Bool);
static const int INPUT_SIZE = {len(low)};
int main(void) {{
    const int64_t low[{len(low)}] = {arr(low)};
    const int64_t high[{len(low)}] = {arr(high)};
    int64_t input[{len(low)}], output[{classes}];
    for (int i = 0; i < INPUT_SIZE; ++i) {{
        input[i] = nondet_int64_t();
        __ESBMC_assume(input[i] >= low[i] && input[i] <= high[i]);
    }}
    qnn_forward_fixed(input, output);
    int selected = 0;
    for (int j = 1; j < {classes}; ++j)
        if (output[j] > output[selected]) selected = j;
    __ESBMC_assert(selected == {target}, "target class, lowest-index tie break");
    return 0;
}}
"""


def run_gate(output: Path, timeout=60, derived=True):
    output.mkdir(parents=True, exist_ok=False)
    cnn = tiny_model()
    specs = [LayerQuantizationSpec(10, 1, 8)] * 2
    encoder = ByteImageEncoder(cnn.geometry.input_shape, 8, 10)
    network = cnn.quantized(specs, input_total_bits=10)
    center = np.ones(cnn.geometry.input_shape, dtype=np.uint8)
    low, high = encoder.box(center, 1)
    lo, hi, frac = low, high, 8
    arithmetic = []
    for layer in network.layers:
        bounds = exact_layer_interval(layer, input_low=lo, input_high=hi,
                                      input_fractional_bits=frac, total_bits=layer.spec.total_bits,
                                      apply_relu=not layer.is_output_layer)
        arithmetic.append(bounds.arithmetic_safety)
        lo, hi, frac = bounds.output_low, bounds.output_high, layer.spec.fractional_bits
    source = output / "qnn.c"
    source.write_text(generate_c_qnn_source(network) + encoder.render_c(), encoding="utf-8")
    libraries = []
    for count in (1, 2):
        prefix = FixedPointNetwork(8, 10, network.layers[:count])
        path = output / f"prefix_{count}.c"
        path.write_text(generate_c_qnn_source(prefix) + encoder.render_c(), encoding="utf-8")
        so = compile_c_qnn_shared_library(path, output / f"prefix_{count}.so")
        lib = ctypes.CDLL(str(so.resolve()))
        lib.qnn_forward_fixed.argtypes = [ctypes.POINTER(ctypes.c_int64)] * 2
        libraries.append(lib)
    cases = 0
    parity_vectors = []
    for vector in itertools.product(range(3), repeat=4):
        encoded = np.array(vector, dtype=np.int64)
        hidden, logits = cnn.integer_reference(encoded.reshape(cnn.geometry.input_shape), specs)
        for lib, expected in zip(libraries, (hidden, logits)):
            actual = np.zeros(len(expected), dtype=np.int64)
            ptr = ctypes.POINTER(ctypes.c_int64)
            lib.qnn_forward_fixed(encoded.ctypes.data_as(ptr), actual.ctypes.data_as(ptr))
            if not np.array_equal(actual, expected):
                raise AssertionError("Direct CNN / lowered C layer parity failed")
        if int(np.argmax(logits)) != 0:
            raise AssertionError("Expected true toy property")
        parity_vectors.append({"raw_hwc_bytes": list(vector), "encoded_input": encoded.tolist(),
                               "hidden": hidden.tolist(), "logits": logits.tolist()})
        cases += 1
    write_new_json(output / "parity_vectors.json", {"shape": list(cnn.geometry.input_shape), "vectors": parity_vectors})
    sanitizer = output / "sanitizer.c"
    sanitizer.write_text(source.read_text() + """
int main(void) {
    int64_t input[4], output[2];
    uint8_t raw[4];
    for (int code = 0; code < 81; ++code) {
        int value = code;
        for (int j = 0; j < 4; ++j) { raw[j] = value % 3; value /= 3; }
        if (qnn_encode_bytes(raw, 2, 2, input)) return 1;
        qnn_forward_fixed(input, output);
        if (output[0] <= output[1]) return 2;
    }
    return 0;
}
""")
    executable = output / "sanitizer"
    subprocess.run(["gcc", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    str(sanitizer), "-o", str(executable)], check=True)
    subprocess.run([str(executable.resolve())], check=True)
    config = ESBMCConfig(timeout_seconds=timeout, default_profile="paper-z3")
    runner = ESBMCRunner(config)
    proofs = []
    for name, target in (("true", 0), ("false", 1)):
        path = output / f"{name}.c"
        path.write_text(classification_harness(network, low, high, target), encoding="utf-8")
        result = runner.run_file(path, extract_counterexample=True)
        record = asdict(result)
        record.update(property=name, harness=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        proofs.append(record)
    false_witness = proofs[1]["counterexample_inputs"]
    witness_replayed = False
    if false_witness is not None and len(false_witness) == 4:
        witness = np.asarray(false_witness, dtype=np.int64)
        if np.all(low <= witness) and np.all(witness <= high):
            _, values = cnn.integer_reference(witness.reshape(cnn.geometry.input_shape), specs)
            witness_replayed = int(np.argmax(values)) != 1
    derived_results = []
    if derived and [p["status"] for p in proofs] == ["VERIFIED", "FAILED"]:
        from synthesis.preqbmc import GPEncoding, QuadapterConfig
        from synthesis.forward import forward_dnn
        for beta in (0, 1, 2):
            model = cnn.as_deep_model()
            cfg = QuadapterConfig(bit_lb=8, bit_ub=8, preimg_mode="milp", verify_mode="esbmc",
                                  sample_id=0, eps=1 / 256, output_dir=output / f"derived_beta{beta}",
                                  esbmc=config, solver="cbc", esbmc_layer_block_size=beta,
                                  error_budget_mode="derived", enforce_contract_chaining=True,
                                  tighten_verified_bounds=True, margin_cuts=False, e2e_fallback=False)
            synth = GPEncoding([4, 4, 2], model, cfg, 0, low / 256, high / 256)
            forward_dnn(center.reshape(-1).astype(np.float32) / 256, synth)
            result = synth.run(low / 256, high / 256)
            # A0 is re-checked from the actual selected candidate, not a report label.
            bridge = False
            if result.success:
                scale = 1 << result.fractional_bits[0]
                a, b = synth._layer_input_bounds_int(synth.dense_layers[0], synth.input_layer, scale)
                exact_encoder = ByteImageEncoder(cnn.geometry.input_shape, result.fractional_bits[0], result.total_bits[0])
                expected_low, expected_high = exact_encoder.box(center, 1)
                bridge = bool(np.all(a <= expected_low) and np.all(expected_high <= b))
            derived_results.append({"beta": beta, **result.to_dict(), "input_bridge_checked": bridge,
                                    "source_region": synth.source_region_summary(),
                                    "preimage_provenance": synth.preimage_provenance_summary(),
                                    "chaining": synth.chaining_summary(), "calls": synth.esbmc_call_records})
    shared = bool(derived_results) and all(
        r["total_bits"] == [s.total_bits for s in specs]
        and r["fractional_bits"] == [s.fractional_bits for s in specs] for r in derived_results)
    passed = ([p["status"] for p in proofs] == ["VERIFIED", "FAILED"] and witness_replayed and
              (not derived or (all(r["success"] and r["input_bridge_checked"] and r["chaining"]["all_ok"]
                                   for r in derived_results) and len(derived_results) == 3 and shared)))
    report = {"study": "enumerable_cnn_feasibility_only", "gate_passed": passed,
              "python": platform.python_version(), "esbmc_version": subprocess.run(
                  [config.executable, "--version"], text=True, stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, check=True).stdout.strip(),
              "exhaustive_inputs": cases, "python_host_layer_parity": True,
              "false_counterexample_replayed": witness_replayed, "host_ubsan": "PASSED",
              "shared_qif_across_block_variants": shared,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "input_domain": "uint8 HWC crop; nearest-neighbor; /256; epsilon=1 byte",
              "arithmetic_safety": arithmetic, "qif": [asdict(s) for s in specs],
              "proofs": proofs, "derived_runs": derived_results,
              "android_parity": "NOT_MEASURED", "power": "NOT_MEASURED",
              "gtsrb_accuracy": None, "gtsrb_certified_regions": 0}
    write_new_json(output / "gate_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-derived", action="store_true", help="Run only the non-preimage arithmetic sanity test")
    args = parser.parse_args()
    report = run_gate(args.output, args.timeout, not args.skip_derived)
    print(json.dumps({"gate_passed": report["gate_passed"], "report": str(args.output / "gate_summary.json")}))
    raise SystemExit(0 if report["gate_passed"] else 1)


if __name__ == "__main__":
    main()
