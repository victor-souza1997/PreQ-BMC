"""Reproduce the input-box boundary gap using existing harness and C generators.

This is an isolated diagnostic, not a full GPEncoding synthesis experiment.
Run from the repository root with PYTHONPATH=src. All generated files are temporary.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from backends.c_qnn_generator import CompiledCQNN, compile_c_qnn_shared_library, write_c_qnn_source
from backends.fixed_point import LayerQuantizationSpec, build_fixed_point_network
from utils.fixed_point import quantize_interval_int_bounds, quantize_int
from verification.c_templates import render_network_end_to_end_program
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc_install import resolve_esbmc_executable


class Dense:
    def __init__(self, weights, biases):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.biases = np.asarray(biases, dtype=np.float32)
        self.units = len(biases)

    def get_weights(self):
        return self.weights, self.biases


def main() -> None:
    sample = np.asarray([0.5], dtype=np.float32)
    epsilon = 0.437500000001
    actual_lower = float(sample[0]) - epsilon
    pipeline_lower = np.clip(sample - epsilon, 0, 1)
    pipeline_upper = np.clip(sample + epsilon, 0, 1)
    low, high = quantize_interval_int_bounds(pipeline_lower, pipeline_upper, 6, 3)
    witness = np.asarray([actual_lower], dtype=np.float64)
    integer_witness = quantize_int(witness, 6, 3)
    model = SimpleNamespace(dense_layers=[Dense([[1]], [0]), Dense([[2, 0]], [0, 0.1])])
    spec = LayerQuantizationSpec(total_bits=6, integer_bits=2, fractional_bits=3)
    network = build_fixed_point_network(model, [spec, spec])
    layers = [
        {
            "input_size": 1, "output_size": 1,
            "total_bits": 6, "fractional_bits": 3, "input_fractional_bits": 3,
            "weights_c_int": "{{8}}", "biases_c_int": "{0}",
            "invariant_low_c_int": "{0}", "invariant_high_c_int": "{31}",
        },
        {
            "input_size": 1, "output_size": 2,
            "total_bits": 6, "fractional_bits": 3, "input_fractional_bits": 3,
            "weights_c_int": "{{16},{0}}", "biases_c_int": "{0,1}",
            "invariant_low_c_int": "{0,0}", "invariant_high_c_int": "{31,31}",
        },
    ]
    evidence = {
        "scope": "Input-domain bridge and generated-kernel diagnostic; not a full synthesis run",
        "sample": float(sample[0]), "epsilon": epsilon,
        "requested_lower_float64": actual_lower,
        "pipeline_lower_float32": float(pipeline_lower[0]),
        "pipeline_integer_box": [low.tolist(), high.tolist()],
        "witness_int": integer_witness.tolist(),
        "witness_in_requested_region": abs(actual_lower - float(sample[0])) <= epsilon,
        "witness_in_verified_box": bool(np.all(integer_witness >= low) and np.all(integer_witness <= high)),
        "float_network_minimum_margin": 2 * actual_lower - float(model.dense_layers[1].biases[1]),
    }
    with tempfile.TemporaryDirectory(prefix="preqbmc-domain-audit-") as tmp:
        directory = Path(tmp)
        source = write_c_qnn_source(network, directory / "network.c")
        library = compile_c_qnn_shared_library(source, directory / "network.so")
        compiled = CompiledCQNN(network, library)
        output = compiled.forward(witness)
        evidence["compiled_c_witness_output"] = output.tolist()
        evidence["compiled_c_property_holds_at_witness"] = bool(output[0] > output[1])
        executable = resolve_esbmc_executable()
        evidence["esbmc_executable"] = executable
        if executable:
            runner = ESBMCRunner(ESBMCConfig(executable=str(executable), timeout_seconds=10, default_profile="paper-z3"))
            for label, lower in (("rounded_box", int(low[0])), ("covering_box", int(integer_witness[0]))):
                harness = directory / f"{label}.c"
                harness.write_text(render_network_end_to_end_program(
                    input_size=1, input_bounds_low_c_int=f"{{{lower}}}",
                    input_bounds_high_c_int=f"{{{int(high[0])}}}", layers=layers,
                    target_label=0, inject_invariants=False,
                ), encoding="utf-8")
                result = runner.run_file(harness, extract_counterexample=True)
                evidence[label] = {"status": result.status, "inputs_int": result.counterexample_inputs}
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
