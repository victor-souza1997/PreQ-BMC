"""Freeze host-validated integer vectors for Android/AAOS device replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct

from datasets.gtsrb_study import sha256
from scripts.run_ssv_cnn_gate import write_new_json


MAGIC = b"PQNNV001"


def _int64_vector(values, size, name):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"Invalid {name} dimension")
    result = []
    for value in values:
        if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
            raise ValueError(f"{name} must contain signed int64 values")
        result.append(value)
    return result


def prepare(host_quality_path: Path, output: Path, certificates=()):
    host_quality_path, output = Path(host_quality_path), Path(output)
    report_path = host_quality_path / "host_quality.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("quality_status") != "MEASURED"
            or not report.get("all_host_parity_passed")
            or report.get("android_parity") != "NOT_MEASURED"):
        raise ValueError("Need a completed host-parity report that has not been relabelled as Android evidence")
    methods = [row for row in report.get("methods", []) if row.get("method") == "preqbmc_selected"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one selected PreQ-BMC implementation")
    method = methods[0]
    method_dir = host_quality_path / "preqbmc_selected"
    source = method_dir / "qnn.c"
    vectors = method_dir / "device_vectors.jsonl"
    if sha256(source) != method.get("source_sha256"):
        raise ValueError("Generated C identity differs from the host-parity report")
    certificate_records = []
    for path in map(Path, certificates):
        certificate = json.loads(path.read_text(encoding="utf-8"))
        if (certificate.get("final_status") != "VERIFIED"
                or not certificate.get("byte_crop_property_verified")
                or certificate.get("generated_source_sha256") != method["source_sha256"]
                or certificate.get("model_sha256") != report.get("model_sha256")):
            raise ValueError(f"Certificate does not verify this exact model and C artifact: {path}")
        certificate_records.append({
            "run_id": certificate.get("run_id"),
            "sample_id": certificate.get("sample", {}).get("id"),
            "epsilon_raw_bytes": certificate.get("epsilon"),
            "guarantee_level": certificate.get("guarantee_level"),
            "region_summary_sha256": sha256(path),
            "region_summary_path": str(path.resolve()),
        })
    if len({row["run_id"] for row in certificate_records}) != len(certificate_records):
        raise ValueError("Duplicate certificate identity")
    rows = []
    with vectors.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid device vector at line {line_number}") from exc
            if not isinstance(row.get("image_id"), str) or not isinstance(row.get("image_sha256"), str):
                raise ValueError("Every device vector needs immutable image identity")
            rows.append(row)
    if not rows or len(rows) != method.get("n_test_images"):
        raise ValueError("Device-vector denominator differs from the host report")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate device-vector image identity")
    input_dim, output_dim = len(rows[0].get("encoded_input", [])), len(rows[0].get("logits", []))
    if input_dim <= 0 or output_dim < 2:
        raise ValueError("Invalid device-vector dimensions")
    output.mkdir(parents=True, exist_ok=False)
    copied_source = output / "qnn.c"
    shutil.copyfile(source, copied_source)
    corpus = output / "vectors.bin"
    identities = []
    with corpus.open("xb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<III", len(rows), input_dim, output_dim))
        for row in rows:
            encoded = _int64_vector(row.get("encoded_input"), input_dim, "encoded input")
            logits = _int64_vector(row.get("logits"), output_dim, "logits")
            label = row.get("label")
            if type(label) is not int or not 0 <= label < output_dim:
                raise ValueError("Invalid vector label")
            handle.write(struct.pack("<i", label))
            handle.write(struct.pack(f"<{input_dim}q", *encoded))
            handle.write(struct.pack(f"<{output_dim}q", *logits))
            identities.append({"image_id": row["image_id"], "image_sha256": row["image_sha256"]})
    manifest = {
        "schema": "preqbmc_android_replay_bundle_v1",
        "purpose": "Android/AAOS integer-C parity and inference-only benchmarking",
        "formal_claim": "NONE_ADDED_BY_THIS_BUNDLE",
        "formal_artifact_status": "REGION_CERTIFICATES_BOUND" if certificate_records else "NO_CERTIFICATE_BOUND",
        "bound_region_certificates": certificate_records,
        "android_status": "NOT_MEASURED",
        "source_model_sha256": report.get("model_sha256"),
        "qif": method.get("qif"),
        "qnn_source": "qnn.c",
        "qnn_source_sha256": sha256(copied_source),
        "vectors": "vectors.bin",
        "vectors_sha256": sha256(corpus),
        "source_vectors_sha256": sha256(vectors),
        "host_quality_report_sha256": sha256(report_path),
        "n_test_images": len(rows),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "test_accuracy_host_c": method["host_c_accuracy"],
        "host_python_c_exact_layer_rate": method["python_c_exact_layer_rate"],
        "images": identities,
        "benchmark_scope": "qnn_forward_fixed only; byte decoding, resize, JNI and UI excluded",
        "required_device_evidence": [
            "automotive feature/build identity", "arm64 ABI and CPU affinity", "NDK and compiler versions",
            "qnn source and binary hashes", "all-vector parity", "independent latency trials",
            "sampled process memory", "synchronized external power traces"
        ],
    }
    write_new_json(output / "bundle.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-quality", type=Path, required=True,
                        help="Directory containing host_quality.json and preqbmc_selected/")
    parser.add_argument("--output", type=Path, required=True, help="New bundle directory")
    parser.add_argument("--certificate", type=Path, action="append", default=[],
                        help="VERIFIED region_summary.json for this exact C artifact; repeat as needed")
    args = parser.parse_args()
    prepare(args.host_quality, args.output, args.certificate)
    print(args.output / "bundle.json")


if __name__ == "__main__":
    main()
