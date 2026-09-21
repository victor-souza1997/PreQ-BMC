"""Run frozen restricted-CNN regions through existing MILP preimages and ESBMC."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
import os
# Força o sistema a olhar para a primeira GPU física
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import load_crop, sha256
from models.restricted_conv import ConvGeometry, RestrictedCNN
from models.ssv_artifact import load_artifact
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.run_ssv_cnn_gate import write_new_json
from verification.esbmc import ESBMCConfig


@dataclass
class RegionContext:
    synthesizer: Any
    cnn: RestrictedCNN
    geometry: ConvGeometry
    image: np.ndarray
    low: np.ndarray
    high: np.ndarray


def build_region_context(study, run, output):
    """Construct the same source box and derived-preimage encoder for all GTSRB paths."""
    from synthesis.preqbmc import GPEncoding, QuadapterConfig
    from synthesis.forward import forward_dnn
    geom = validate_config(study["config"])
    if sha256(study["model_path"]) != study["model_sha256"]:
        raise ValueError("Source model identity mismatch")
    with np.load(study["model_path"], allow_pickle=False) as weights:
        cnn = RestrictedCNN(geom, **dict(weights), max_affine_entries=study["config"]["model"]["max_affine_entries"])
    image = load_crop(study["dataset_root"], run["sample"])
    # At F=8, Q=16 the /256 encoder equals raw bytes, with no clamp loss.
    byte_encoder = ByteImageEncoder(geom.input_shape)
    byte_low, byte_high = byte_encoder.box(image, run["epsilon"])
    low, high = byte_low / 256, byte_high / 256  # exactly representable dyadics
    model = cnn.as_deep_model()
    center = byte_encoder.resize(image).reshape(-1).astype(np.float32) / 256
    target = int(run["sample"]["class_id"])
    if int(np.argmax(np.asarray(model(center[None, :])))) != target:
        raise ValueError("Frozen clean prediction changed")
    cfg = QuadapterConfig(
        bit_lb=run["bit_lb"], bit_ub=run["bit_ub"], preimg_mode="milp", verify_mode="esbmc",
        sample_id=0, eps=float(run["epsilon"]) / 256, output_dir=output / "layers", solver="cbc",
        esbmc=ESBMCConfig(timeout_seconds=run["timeout_seconds"], memlimit=run["memlimit"], default_profile=run["profile"]),
        esbmc_layer_block_size=run["block_size"], error_budget_mode="derived", harness_scope="layer",
        tighten_verified_bounds=True, enforce_contract_chaining=True, unsound_contract_tolerance=False,
        margin_cuts=run["margin_cuts"], e2e_fallback=False, cegar_max_rounds=run["cegar_max_rounds"],
        blockwise_run_all_blocks_on_failure=run["blockwise_run_all_blocks_on_failure"],
        source_verification=run.get("source_verification", "deeppoly"),
        source_milp_timeout_seconds=float(run.get("source_milp_timeout_seconds", 30.0)),
        output_refinement=run.get("output_refinement", "none"))
    synth = GPEncoding([len(center), int(np.prod(geom.output_shape)), 43], model, cfg, target, low, high)
    forward_dnn(center, synth)
    return RegionContext(synth, cnn, geom, image, low, high)


def run_region(study, run, output, *, fixed_artifact=None):
    artifact, fixed_qif = (None, None) if fixed_artifact is None else load_artifact(fixed_artifact, study)
    if run.get("verification_mode") == "fixed_qif_check" and artifact is None:
        raise ValueError("Fixed-check campaign requires its frozen artifact")
    if fixed_qif is not None and run.get("fixed_qif", artifact["qif"]) != artifact["qif"]:
        raise ValueError("Run Q/I/F differs from frozen artifact")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    context = build_region_context(study, run, output)
    synth, cnn, geom, image = context.synthesizer, context.cnn, context.geometry, context.image
    low, high = context.low, context.high
    result = synth.run(low, high, fixed_qif=fixed_qif)
    bridge, source_hash, qif = False, None, fixed_qif or []
    if result.success:
        qif = [LayerQuantizationSpec(q, i, f) for q, i, f in
               zip(result.total_bits, result.integer_bits, result.fractional_bits)]
        if fixed_qif is not None and qif != fixed_qif:
            raise ValueError("Fixed checking changed the requested formats")
        enc = ByteImageEncoder(geom.input_shape, qif[0].fractional_bits, qif[0].total_bits)
        e_low, e_high = enc.box(image, run["epsilon"])
        a_low, a_high = synth._layer_input_bounds_int(synth.dense_layers[0], synth.input_layer, 1 << qif[0].fractional_bits)
        bridge = bool(np.all(a_low <= e_low) and np.all(e_high <= a_high))
        network = cnn.quantized(qif, input_fractional_bits=qif[0].fractional_bits, input_total_bits=qif[0].total_bits)
        path = output / "qnn.c"
        path.write_text(generate_c_qnn_source(network) + enc.render_c(), encoding="utf-8")
        source_hash = sha256(path)
        if artifact is not None and source_hash != artifact["generated_source_sha256"]:
            raise ValueError("Verified export differs from the frozen artifact")
        write_new_json(output / "input_bridge.json", {"checked": bridge, "E_low": e_low.tolist(),
                       "E_high": e_high.tolist(), "A0_low": a_low.tolist(), "A0_high": a_high.tolist(),
                       "encoder": asdict(enc), "domain": "uint8 raw cropped RGB images"})
    chaining = synth.chaining_summary()
    encoded = bool(result.success and chaining["all_ok"])
    bridged = bool(encoded and bridge)
    report = {"schema": "ssv_region_v1", "run_id": run["run_id"], "sample": run["sample"],
              "epsilon": run["epsilon"], "block_size": run["block_size"], "margin_cuts": run["margin_cuts"],
              **result.to_dict(), "source_region": synth.source_region_summary(),
              "encoded_contract_verified": encoded, "input_bridge_checked": bridge,
              "byte_crop_property_verified": bridged,
              "guarantee_level": "byte-crop-integer-C" if bridged else "unknown",
              "final_status": result.final_status if not encoded or bridge else "INPUT_BRIDGE_INCONCLUSIVE",
              "android_transfer_verified": False, "android_parity": "NOT_MEASURED",
              "deployment_quality_status": "NOT_MEASURED", "power_status": "NOT_MEASURED",
              "source_float32_test_accuracy": study["source_float32_test_accuracy"],
              "qif": [asdict(spec) for spec in qif], "generated_source_sha256": source_hash,
              "verification_mode": "fixed_qif_check" if artifact else "region_synthesis",
              "fixed_artifact_sha256": sha256(fixed_artifact) if artifact else None,
              "fixed_artifact_source_sha256": artifact["generated_source_sha256"] if artifact else None,
              "fixed_qif_checks": getattr(synth, "fixed_qif_records", []),
              "model_sha256": study["model_sha256"], "preimage": synth.preimage_provenance_summary(),
              "chaining": chaining, "vacuity": synth.vacuity_summary(),
              "cegar": synth.cegar_summary(), "calls": synth.esbmc_call_records,
              "output_refinement": run.get("output_refinement", "none"),
              "affine_residual": synth.affine_residual_summary(),
              "composition_path": synth.composition_path,
              "total_runtime_seconds": time.monotonic() - start}
    write_new_json(output / "region_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], help="Exact frozen run ID; repeat to run a subset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixed-artifact", type=Path, help="Frozen artifact.json; disables bit-width synthesis")
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    validate_config(study["config"])
    fixed_artifact = args.fixed_artifact or study.get("fixed_artifact")
    if study.get("fixed_artifact_sha256") and (
            fixed_artifact is None or sha256(fixed_artifact) != study["fixed_artifact_sha256"]):
        parser.error("Frozen campaign artifact identity mismatch")
    if fixed_artifact is not None:
        load_artifact(fixed_artifact, study)
    requested_ids = set(args.only)
    runs = [r for r in study["runs"] if not requested_ids or r["run_id"] in requested_ids]
    if not runs:
        parser.error("No matching frozen region")
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output / "study_identity.json", {"study": str(args.study.resolve()), "sha256": sha256(args.study),
                   "fixed_artifact_sha256": sha256(fixed_artifact) if fixed_artifact else None})
    for run in runs:
        destination = args.output / run["run_id"]
        try:
            report = run_region(study, run, destination, fixed_artifact=fixed_artifact)
            print(run["run_id"], report["final_status"], flush=True)
        except Exception as exc:
            # A runner error is NOT an ESBMC failure or a source counterexample.
            destination.mkdir(parents=True, exist_ok=True)
            write_new_json(destination / "runner_error.json", {"run_id": run["run_id"], "status": "RUNNER_ERROR",
                           "error_type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
