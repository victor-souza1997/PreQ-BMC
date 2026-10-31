"""Independent coverage check for v2 shard checks.

The renderer decides which harnesses to emit, so counting its manifest cannot
detect an obligation it never emitted. This module derives the assertions a
complete check must contain from the deployed CSR rows and the validated chain
steps, reads the assertion labels from the harness files ESBMC actually
verified, and requires the two to match exactly. It never imports the renderer.

What it does not establish: that each assertion's C expression encodes the
obligation its label names. The harness translation stays in the trusted base.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
from typing import Any

from verification.crown_chain_v2 import CertificateV2, _sparse

# Labels of the shared lemmas; each is required once per shard that uses the
# corresponding step kind.
ROUNDING_LEMMA = "deployed rounding error is within half a step"
RELU_LEMMAS = (
    "ReLU is nonnegative after clamp",
    "without upper saturation ReLU dominates raw",
    "lower fact bounds ReLU-minus-raw",
    "upper fact bounds ReLU output",
    "two-sided class excludes clamp saturation",
    "two-sided class is piecewise linear ReLU",
    "positive coefficient preserves nonnegative order gap",
)
_LABEL = re.compile(r'"([^"]*)"\);\s*$')
_COEFFICIENT_ENVELOPE = re.compile(r"coefficient > 0 && coefficient <= (\d+)\)")
_RAW_ENVELOPE = re.compile(r"raw >= -(\d+) && raw <= (\d+)\)")


def required_assertions(certificate: CertificateV2) -> set[str]:
    """Assertion labels a complete check of the loaded chains must contain."""

    required: set[str] = set()
    uses_affine = uses_relu = False
    for chain in certificate.chains:
        chain_id = chain["certificate_id"]
        steps = chain["_normalized_steps"]
        for step in steps:
            step_id = step["step_id"]
            if step["kind"] == "affine":
                uses_affine = True
                csr = certificate.layers[int(step["layer_index"])]
                columns = {
                    csr["indices"][position]
                    for row in step["_input_coefficients"]
                    for position in range(csr["indptr"][row], csr["indptr"][row + 1])
                }
                for column in columns:
                    required.update(
                        f"{step_id} {check} {column}" for check in ("g", "floor", "residual")
                    )
                required.update(
                    f"{step_id} {check}"
                    for check in ("bias dot", "residual dot", "rounding term", "output constant")
                )
                if int(step["layer_index"]) > 0:
                    required.update(
                        f"{step_id} residual fact {column} in Q16" for column in step["_lower_values"]
                    )
            elif step["kind"] == "relu":
                uses_relu = True
                for class_name, indices in step["coordinate_classes"].items():
                    for index in indices:
                        required.add(f"{step_id} {class_name} instance {index}")
                        if class_name == "lower_upper":
                            required.add(f"{step_id} lower_upper breakpoint {index}")
                required.add(f"{step_id} constant sum")
            else:
                required.add(f"{step_id} concretize")
        for left in steps[:-1]:
            required.add(f"{chain_id} coefficient link {left['step_id']}")
            required.add(f"{chain_id} constant link {left['step_id']}")
        required.add(f"{chain_id} rounded bound")
    if uses_affine:
        required.add(ROUNDING_LEMMA)
    if uses_relu:
        required.update(RELU_LEMMAS)
    return required


def _relu_instances(certificate: CertificateV2):
    for chain in certificate.chains:
        for step in chain["_normalized_steps"]:
            if step["kind"] != "relu":
                continue
            rows = certificate.layers[int(step["layer_index"])]["rows"]
            output = _sparse(step["output_coefficients"], "output_coefficients", rows)
            for index, coefficient in step["_input_coefficients"].items():
                lower, upper = step["_fact_values"][index]
                yield step["step_id"], index, coefficient, output.get(index, 0), lower, upper


def verify_shard_coverage(certificate: CertificateV2, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Require one VERIFIED, unchanged assertion for every required obligation."""

    required = required_assertions(certificate)
    seen: Counter[str] = Counter()
    lemma_sources: list[str] = []
    for record in records:
        path = Path(record["harness"])
        source = path.read_text(encoding="utf-8")
        if record["status"] != "VERIFIED":
            raise ValueError(f"{path} is not VERIFIED")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != record["harness_sha256"]:
            raise ValueError(f"{path} changed after it was verified")
        for line in source.splitlines():
            if "__ESBMC_assert(" not in line or line.lstrip().startswith("void"):
                continue
            match = _LABEL.search(line)
            if not match:
                raise ValueError(f"{path} has an assertion without a parseable label")
            seen[match.group(1)] += 1
        if record.get("chain_id") == "__shared_lemmas__":
            lemma_sources.append(source)
    missing = required - set(seen)
    unexpected = set(seen) - required
    duplicated = sorted(label for label, count in seen.items() if count > 1)
    if missing or unexpected or duplicated:
        raise ValueError(
            f"Coverage mismatch: {len(missing)} missing (e.g. {sorted(missing)[:3]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:3]}), "
            f"{len(duplicated)} duplicated (e.g. {duplicated[:3]})"
        )
    instances = list(_relu_instances(certificate))
    if instances:
        # The shared ReLU lemmas quantify over a finite envelope; every
        # instance they discharge must lie inside it.
        coefficient_limits = [int(value) for text in lemma_sources for value in _COEFFICIENT_ENVELOPE.findall(text)]
        raw_limits = [int(low) for text in lemma_sources for low, high in _RAW_ENVELOPE.findall(text) if low == high]
        if not coefficient_limits or not raw_limits:
            raise ValueError("Shared ReLU lemmas state no envelope")
        coefficient_limit, raw_limit = min(coefficient_limits), min(raw_limits)
        for step_id, index, a, a_out, lower, upper in instances:
            if abs(a) > coefficient_limit or abs(a_out) > coefficient_limit or any(
                value is not None and abs(value) > raw_limit for value in (lower, upper)
            ):
                raise ValueError(f"{step_id} instance {index} lies outside the shared lemma envelope")
    return {
        "status": "COMPLETE",
        "required_assertions": len(required),
        "verified_harnesses": len(records),
        "relu_instances_in_envelope": len(instances),
    }
