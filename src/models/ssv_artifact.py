"""Identity checks for one immutable traffic-sign integer program."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from datasets.gtsrb_study import sha256
from models.restricted_conv import RestrictedCNN
from scripts.prepare_ssv_gtsrb import validate_config


def parse_qif(rows):
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Expected convolution and output layer Q/I/F entries")
    specs = []
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"total_bits", "integer_bits", "fractional_bits"}
                or any(type(v) is not int for v in row.values())):
            raise ValueError("Q/I/F fields must be explicit integers")
        spec = LayerQuantizationSpec(**row)
        if not 8 <= spec.fractional_bits <= 16 or spec.total_bits > 62:
            raise ValueError("Traffic-sign artifacts require 8 <= F <= 16 and Q <= 62")
        specs.append(spec)
    return specs


def render_artifact(study, specs):
    geom = validate_config(study["config"])
    if sha256(study["model_path"]) != study["model_sha256"]:
        raise ValueError("Source model identity mismatch")
    with np.load(study["model_path"], allow_pickle=False) as weights:
        cnn = RestrictedCNN(geom, **dict(weights),
                            max_affine_entries=study["config"]["model"]["max_affine_entries"])
    enc = ByteImageEncoder(geom.input_shape, specs[0].fractional_bits, specs[0].total_bits)
    network = cnn.quantized(specs, input_fractional_bits=enc.fractional_bits,
                            input_total_bits=enc.total_bits)
    source = generate_c_qnn_source(network) + enc.render_c()
    metadata = asdict(enc)
    metadata["output_shape"] = list(enc.output_shape)
    return source, metadata


def freeze_artifact(study, rows, output):
    specs = parse_qif(rows)
    source, encoder = render_artifact(study, specs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "qnn.c").write_text(source, encoding="utf-8")
    manifest = {"schema": "ssv_fixed_artifact_v1", "verification_mode": "fixed_qif_check",
                "model_sha256": study["model_sha256"], "qif": [asdict(s) for s in specs],
                "encoder": encoder, "tie_break": "lowest_class_index", "source": "qnn.c",
                "generated_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "proof_status": "NOT_RUN", "android_status": "NOT_MEASURED"}
    (output / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_artifact(path, study):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "ssv_fixed_artifact_v1"
            or manifest.get("verification_mode") != "fixed_qif_check"
            or manifest.get("model_sha256") != study["model_sha256"]
            or manifest.get("tie_break") != "lowest_class_index"
            or manifest.get("source") != "qnn.c"):
        raise ValueError("Fixed artifact identity or semantics mismatch")
    specs = parse_qif(manifest["qif"])
    source, encoder = render_artifact(study, specs)
    expected = manifest["generated_source_sha256"]
    if (manifest["encoder"] != encoder
            or hashlib.sha256(source.encode("utf-8")).hexdigest() != expected
            or sha256(path.parent / "qnn.c") != expected):
        raise ValueError("Fixed artifact changed: model, encoder, Q/I/F or C generator drift")
    return manifest, specs
