"""Run a bounded pilot or complete proof-carrying convolution campaign.

A positive max_blocks_per_layer creates a non-certifying feasibility pilot.
A null limit executes the complete ledger and may certify only when every
required ESBMC obligation is VERIFIED.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from backends.conv_fixed_point import (
    forward_conv_fixed_point_single,
    quantize_restricted_sequential,
)
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from scripts.evaluate_ssv_deep_source_model import restricted_from_npz
from scripts.run_ssv_cnn_gate import write_new_json
from verification.conv_contracts import IntegerInvariant
from verification.conv_proof import ConvProofConfig, ConvProofCoordinator
from verification.esbmc import ESBMCConfig


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parents[1] / path).resolve()


def validate_config(config: dict):
    if config.get("schema") != "ssv_conv_native_pilot_v1":
        raise ValueError("Expected ssv_conv_native_pilot_v1")
    qif = config.get("qif")
    if not isinstance(qif, list) or len(qif) < 2:
        raise ValueError("Need one Q/I/F entry per affine stage")
    specs = [LayerQuantizationSpec(**row) for row in qif]
    proof = config.get("proof", {})
    if proof.get("proof_mode", "explicit_transition") not in {
        "explicit_transition", "proof_carrying_interval"
    }:
        raise ValueError("Unsupported proof.proof_mode")
    if type(proof.get("exact_margin_refinement", False)) is not bool:
        raise ValueError("proof.exact_margin_refinement must be boolean")
    if proof.get("cache_directory") is not None and not isinstance(
            proof.get("cache_directory"), str):
        raise ValueError("proof.cache_directory must be a path string or null")
    if type(proof.get("block_size")) is not int or proof["block_size"] <= 0:
        raise ValueError("proof.block_size must be positive")
    max_blocks = proof.get("max_blocks_per_layer")
    if max_blocks is not None and (type(max_blocks) is not int or max_blocks <= 0):
        raise ValueError("proof.max_blocks_per_layer must be null or a positive integer")
    if type(proof.get("jobs", 1)) is not int or proof.get("jobs", 1) <= 0:
        raise ValueError("proof.jobs must be positive")
    if type(proof.get("min_available_gib", 6.0)) not in {int, float} \
            or proof.get("min_available_gib", 6.0) < 0:
        raise ValueError("proof.min_available_gib must be nonnegative")
    sample_split = config.get("split", "test")
    if sample_split not in {"test", "validation"}:
        raise ValueError("split must be test or validation")
    if type(config.get("epsilon_raw_bytes")) not in {int, float} \
            or config["epsilon_raw_bytes"] < 0:
        raise ValueError("epsilon_raw_bytes must be nonnegative")
    return specs


def run(config_path: Path, output: Path):
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    specs = validate_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")

    search_root = _resolve(config_path, config["source_search_output"])
    search = _load_json(search_root / "search_summary.json")
    if search.get("status") != "SELECTED":
        raise ValueError("Source-model search has no selected candidate")
    selected = search["selected_candidate_summary"]
    params = Path(selected["folded_model_path"])
    if sha256(params) != selected["folded_model_sha256"]:
        raise ValueError("Selected folded-model identity changed")
    if len(specs) != len(selected["geometries"]) + len(selected["dense_hidden"]) + 1:
        raise ValueError("Q/I/F count does not match selected model stages")
    test_report = _load_json(_resolve(config_path, config["test_evaluation"]))
    if (test_report.get("status") != "COMPLETED"
            or test_report.get("folded_model_sha256") != selected["folded_model_sha256"]):
        raise ValueError("Held-out test evidence does not identify the selected model")
    base_path = Path(search["base_study_path"])
    base = _load_json(base_path)
    if sha256(base_path) != search["base_study_sha256"]:
        raise ValueError("Frozen dataset study changed")
    sample_split = config.get("split", "test")
    matches = [
        row for row in base["records"]
        if row["id"] == config["sample_id"] and row.get("split") == sample_split
    ]
    if len(matches) != 1:
        raise ValueError(
            "sample_id must identify exactly one frozen image in the configured split"
        )
    sample = matches[0]
    image = load_crop(base["dataset_root"], sample)
    if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() == sample.get("sha256"):
        image_identity = "decoded_pixels"
    else:
        image_identity = "frozen_record_file_hash"

    model = restricted_from_npz(selected, params)
    input_format = config["input_format"]
    encoder = ByteImageEncoder(
        model.input_shape,
        fractional_bits=input_format["fractional_bits"],
        total_bits=input_format["total_bits"],
        resize_mode=config["resize_mode"],
    )
    byte_low, byte_high = encoder.byte_box(image, config["epsilon_raw_bytes"])
    low, high = encoder.encode(byte_low), encoder.encode(byte_high)
    center = encoder.encode(image)
    input_invariant = IntegerInvariant.from_arrays(low, high, model.input_shape, "exact_byte_box_image")
    network = quantize_restricted_sequential(
        model,
        specs,
        input_fractional_bits=encoder.fractional_bits,
        input_total_bits=encoder.total_bits,
    )
    center_logits, center_trace = forward_conv_fixed_point_single(
        network, center, return_trace=True
    )
    prediction = int(np.argmax(center_logits))
    target = int(sample["class_id"])
    if prediction != target:
        output.mkdir(parents=True)
        blocked = {
            "schema": config["schema"],
            "final_status": "CENTER_MISCLASSIFIED",
            "certified": False,
            "sample_id": sample["id"],
            "sample_split": sample_split,
            "target_class": target,
            "quantized_prediction": prediction,
            "qif": [asdict(spec) for spec in specs],
            "proof_status": "NOT_RUN",
        }
        write_new_json(output / "pilot_summary.json", blocked)
        return blocked

    proof = config["proof"]
    coordinator = ConvProofCoordinator(
        network,
        output / "proof",
        esbmc=ESBMCConfig(
            timeout_seconds=proof["timeout_seconds"],
            memlimit=proof["memlimit"],
            default_profile=proof["profile"],
        ),
        cache_directory=(
            _resolve(config_path, proof["cache_directory"])
            if proof.get("cache_directory") else None
        ),
        encoder_source=encoder.render_c(),
        config=ConvProofConfig(
            block_size=proof["block_size"],
            proof_mode=proof.get("proof_mode", "explicit_transition"),
            exact_margin_refinement=proof.get("exact_margin_refinement", False),
            profile=proof["profile"],
            fail_fast=proof.get("fail_fast", True),
            max_blocks_per_layer=proof.get("max_blocks_per_layer"),
            jobs=proof.get("jobs", 1),
            min_available_gib=float(proof.get("min_available_gib", 6.0)),
        ),
    )
    formal = coordinator.verify(
        input_invariant, input_witness=center, target_class=target,
        input_byte_low=byte_low.reshape(-1), input_byte_high=byte_high.reshape(-1),
    )
    summary = {
        "schema": config["schema"],
        "final_status": formal["final_status"],
        "certified": formal["certified"],
        "reason": (
            "bounded feasibility pilots cannot establish a network certificate"
            if proof.get("max_blocks_per_layer") is not None
            else "complete proof ledger determines certification"
        ),
        "sample_id": sample["id"],
        "sample_split": sample_split,
        "sample_sha256": sample["sha256"],
        "sample_identity_kind": image_identity,
        "target_class": target,
        "quantized_prediction": prediction,
        "epsilon_raw_bytes": config["epsilon_raw_bytes"],
        "qif": [asdict(spec) for spec in specs],
        "center_trace_shapes": [list(value.shape) for value in center_trace],
        "source_test_accuracy": test_report["test_accuracy_folded_affine"],
        "source_local_robustness": {
            "status": "NOT_CHECKED",
            "method": None,
            "eligible_for_transfer": False,
            "note": (
                "the convolution-native path currently proves the integer QNN directly; "
                "it does not establish the float-source local property"
            ),
        },
        "float_to_integer_transfer_claim": False,
        "proof_summary": str(output / "proof" / "proof_summary.json"),
        "formal": formal,
    }
    write_new_json(output / "pilot_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = _load_json(args.config)
    if args.check_config:
        specs = validate_config(config)
        print(json.dumps({"configuration_valid": True, "qif": [asdict(s) for s in specs],
                          "certificate_claim": "NOT_RUN"}, indent=2))
        return
    result = run(args.config, args.output)
    print(json.dumps({key: result[key] for key in (
        "final_status", "certified", "sample_id", "quantized_prediction"
    )}, indent=2))


if __name__ == "__main__":
    main()
