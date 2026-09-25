"""Corrupt v2 certificate data after Python validation and require ESBMC to reject it.

Each case copies one validated chain, changes a number the Python validator
already accepted (as a validator bug would), renders it through the production
renderer, and requires ESBMC to return FAILED on the named assertion.  The
unmutated rendering of the same harness must be VERIFIED, so each FAILED is
attributable to the corruption.  This tests that ESBMC, not Python, decides.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from verification.crown_chain_v2 import (
    Q_HIGH,
    CertificateV2,
    _render_affine,
    _render_chain_close,
    _render_concretize,
    _render_relu,
    load_certificate_v2,
    relu_envelope,
    render_rounding_lemma,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner


Rendered = list[tuple[str, str, dict[str, Any]]]


def _pick_chain(certificate: CertificateV2) -> dict[str, Any]:
    """First chain with a residual-fact affine step and a ReLU step covering every class."""

    for chain in certificate.chains:
        steps = chain["_normalized_steps"]
        affine = [s for s in steps if _residual_affine(s)]
        relu = [s for s in steps if s["kind"] == "relu"]
        if affine and relu and all(relu[0]["coordinate_classes"][name] for name in (
            "bound_free", "nosat_upper", "lower", "upper", "lower_upper"
        )):
            return chain
    raise ValueError("No chain exercises every obligation class")


def _residual_affine(step: dict[str, Any]) -> bool:
    """An affine step past the input layer whose floor residual is not identically zero."""

    return step["kind"] == "affine" and int(step["layer_index"]) > 0 and any(step["_residual"].values())


def _step(chain: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "affine":
        return next(s for s in chain["_normalized_steps"] if _residual_affine(s))
    return next(s for s in chain["_normalized_steps"] if s["kind"] == kind)


def _first(step: dict[str, Any], class_name: str, predicate: Callable[[int], bool] = lambda _: True) -> int:
    return next(index for index in step["coordinate_classes"][class_name] if predicate(index))


def _set_sparse(payload: dict[str, Any], index: int, value: int) -> None:
    entries = dict(zip(payload["indices"], payload["values"]))
    entries[index] = value
    entries = {k: v for k, v in sorted(entries.items()) if v}
    payload["indices"], payload["values"] = list(entries), list(entries.values())


def _cases(certificate: CertificateV2, chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Each case: name, mutate(chain) -> needle naming the harness, expected failing assertion text."""

    coefficient_limit, raw_limit = relu_envelope(certificate)
    affine = _step(chain, "affine")
    relu = _step(chain, "relu")
    constants = dict(zip(relu["relu_constants"]["indices"], relu["relu_constants"]["values"]))
    # Prefer a cited fact; otherwise inject one (a validator accepting a bogus citation).
    residual_column = next(
        (c for c in sorted(affine["_lower_values"]) if affine["_residual"].get(c)),
        next(c for c in sorted(affine["_residual"]) if affine["_residual"][c]),
    )
    output_column = affine["output_coefficients"]["indices"][0]

    def affine_residual_q16(ch):
        step = _step(ch, "affine")
        step["_lower_values"][residual_column] = Q_HIGH + 1
        # Keep every other assertion consistent so only the Q16 fact check can fail.
        step["_residual_dot"] = sum(v * step["_lower_values"].get(i, 0) for i, v in step["_residual"].items())
        l1 = sum(abs(v) for v in step["_input_coefficients"].values())
        step["output_constant"] = (
            int(step["input_constant"]) + step["_bias_dot"] + (step["_residual_dot"] - 128 * l1) // 256
        )

    def affine_rounding_term(ch):
        step = _step(ch, "affine")
        step["rounding_term"] = int(step["rounding_term"]) - 256
        step["output_constant"] = int(step["output_constant"]) + 1

    def relu_constant(class_name, delta):
        index = _first(relu, class_name)

        def mutate(ch):
            step = _step(ch, "relu")
            _set_sparse(step["relu_constants"], index, constants.get(index, 0) + delta)
        return index, mutate

    nosat_index = _first(relu, "nosat_upper")

    def nosat_fact(ch):
        step = _step(ch, "relu")
        step["_fact_values"][nosat_index] = (None, Q_HIGH + 1)

    two_sided = _first(relu, "lower_upper", lambda i: relu["_fact_values"][i][1] > 0)

    def two_sided_fact(ch):
        step = _step(ch, "relu")
        lower, _ = step["_fact_values"][two_sided]
        upper = Q_HIGH + 1
        a, a_out = step["_input_coefficients"][two_sided], _output(step).get(two_sided, 0)
        points = [lower, upper] + ([0] if lower < 0 < upper else [])
        step["_fact_values"][two_sided] = (lower, upper)
        _set_sparse(step["relu_constants"], two_sided, min(a * max(p, 0) - a_out * p for p in points))

    def relu_sum(ch):
        step = _step(ch, "relu")
        step["output_constant"] = int(step["output_constant"]) + 1

    def concretize(ch):
        step = _step(ch, "concretize")
        step["scaled_bound"] = int(step["scaled_bound"]) + 1

    def bound(ch):
        ch["bound"] = int(ch["bound"]) + 1

    def link(ch):
        ch["_normalized_steps"][1]["input_constant"] = int(ch["_normalized_steps"][1]["input_constant"]) + 1

    def output_coefficient(ch):
        step = _step(ch, "affine")
        _set_sparse(step["output_coefficients"], output_column, step["output_coefficients"]["values"][0] + 1)

    lower_index, lower_mutate = relu_constant("lower", -1)
    upper_index, upper_mutate = relu_constant("upper", -1)
    lu_index, lu_mutate = relu_constant("lower_upper", 1)
    return [
        {"name": "affine_output_coefficient", "kind": "affine", "mutate": output_coefficient,
         "needle": f' floor {output_column}"', "assertion": f"floor {output_column}"},
        {"name": "affine_residual_fact_outside_q16", "kind": "affine_close", "mutate": affine_residual_q16,
         "needle": "residual fact", "assertion": f"residual fact {residual_column} in Q16"},
        {"name": "affine_rounding_term", "kind": "affine_close", "mutate": affine_rounding_term,
         "needle": "rounding term", "assertion": "rounding term"},
        {"name": "relu_lower_constant", "kind": "relu", "mutate": lower_mutate,
         "needle": f"lower instance {lower_index}\"", "assertion": f"lower instance {lower_index}"},
        {"name": "relu_upper_constant", "kind": "relu", "mutate": upper_mutate,
         "needle": f"upper instance {upper_index}\"", "assertion": f"upper instance {upper_index}"},
        {"name": "relu_lower_upper_constant", "kind": "relu", "mutate": lu_mutate,
         "needle": f"lower_upper instance {lu_index}\"", "assertion": f"lower_upper instance {lu_index}"},
        {"name": "relu_nosat_fact_saturates", "kind": "relu", "mutate": nosat_fact,
         "needle": f"nosat_upper instance {nosat_index}\"", "assertion": f"nosat_upper instance {nosat_index}"},
        {"name": "relu_lower_upper_fact_outside_q16", "kind": "relu", "mutate": two_sided_fact,
         "needle": f"lower_upper instance {two_sided}\"", "assertion": f"lower_upper instance {two_sided}"},
        {"name": "relu_coefficient_outside_lemma_envelope", "kind": "relu", "mutate": None,
         "envelope": (abs(relu["_input_coefficients"][lower_index]) - 1, raw_limit),
         # The first instance whose |a| exceeds the shrunken envelope fails, whichever class it is.
         "needle": f"lower instance {lower_index}\"", "assertion": " instance "},
        {"name": "relu_constant_sum", "kind": "relu_close", "mutate": relu_sum,
         "needle": "constant sum", "assertion": "constant sum"},
        {"name": "concretize_scaled_bound", "kind": "concretize", "mutate": concretize,
         "needle": "concretize", "assertion": "concretize"},
        {"name": "chain_bound_rounding", "kind": "chain_close", "mutate": bound,
         "needle": "rounded bound", "assertion": "rounded bound"},
        {"name": "chain_constant_link", "kind": "chain_close", "mutate": link,
         "needle": "constant link", "assertion": "constant link"},
        {"name": "_envelope", "coefficient_limit": coefficient_limit, "raw_limit": raw_limit},
    ]


def _output(step: dict[str, Any]) -> dict[int, int]:
    return dict(zip(step["output_coefficients"]["indices"], step["output_coefficients"]["values"]))


def _render(certificate: CertificateV2, chain: dict[str, Any], kind: str, needle: str,
            envelope: tuple[int, int]) -> tuple[str, str]:
    rendered: Rendered = []
    if kind in {"affine", "affine_close"}:
        rendered = _render_affine(certificate, chain, _step(chain, "affine"), 25_000)
    elif kind in {"relu", "relu_close"}:
        rendered = _render_relu(certificate, chain, _step(chain, "relu"), 512, *envelope)
    elif kind == "concretize":
        rendered = [_render_concretize(certificate, chain, _step(chain, "concretize"))]
    else:
        rendered = [_render_chain_close(certificate, chain)]
    wanted = {"affine": "affine_chunk", "affine_close": "affine_close", "relu": "relu_chunk",
              "relu_close": "relu_close"}.get(kind, kind)
    matches = [(name, source) for name, source, meta in rendered if meta["kind"] == wanted and needle in source]
    if len(matches) != 1:
        raise ValueError(f"{kind}: {len(matches)} harnesses match {needle!r}")
    return matches[0]


def run_regression(certificate_dir: Path, deployment_c: Path, output: Path, *, layer: int,
                   timeout: int, memlimit: str, profile: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    certificate = load_certificate_v2(certificate_dir, selected_layers=[layer], deployment_c=deployment_c)
    chain = _pick_chain(certificate)
    cases = _cases(certificate, chain)
    envelope_record = cases.pop()
    envelope = (envelope_record["coefficient_limit"], envelope_record["raw_limit"])
    runner = ESBMCRunner(ESBMCConfig(timeout_seconds=timeout, memlimit=memlimit, default_profile=profile))
    controls: dict[str, dict[str, Any]] = {}
    results = []

    def run(path: Path, source: str) -> Any:
        path.write_text(source, encoding="utf-8")
        return runner.run_file(path, profile=profile)

    def _violated(path: Path, source: str) -> str:
        """Attribution only: the paper profile uses --result-only, so re-run a copy to name the property."""

        copy_path = path.with_suffix(".attribution.c")
        copy_path.write_text(source, encoding="utf-8")
        result = runner.run_file(copy_path, profile="safety")
        text = result.stdout + "\n" + result.stderr  # ESBMC 7.11 prints the property on stderr
        if "Violated property:" not in text:
            return ""
        block = text.split("Violated property:")[-1].strip().splitlines()
        return " ".join(line.strip() for line in block[:3])

    for ordinal, case in enumerate(cases):
        _, clean = _render(certificate, chain, case["kind"], case["needle"], envelope)
        digest = hashlib.sha256(clean.encode()).hexdigest()
        if digest not in controls:
            result = run(output / f"control_{len(controls):02d}.c", clean)
            controls[digest] = {"status": result.status, "harness": str(output / f"control_{len(controls):02d}.c")}
        mutated = copy.deepcopy(chain)
        if case["mutate"] is not None:
            case["mutate"](mutated)
        _, corrupted = _render(certificate, mutated, case["kind"], case["needle"], case.get("envelope", envelope))
        if corrupted == clean:
            raise ValueError(f"{case['name']}: corruption did not change the rendered harness")
        path = output / f"{ordinal:02d}_{case['name']}.c"
        result = run(path, corrupted)
        violated_text = _violated(path, corrupted) if result.status == "FAILED" else ""
        results.append({
            "case": case["name"],
            "corrupted_harness": str(path),
            "control_status": controls[digest]["status"],
            "status": result.status,
            "expected_assertion": case["assertion"],
            "violated_property": violated_text,
            "named_assertion_violated": case["assertion"] in violated_text,
            "elapsed_seconds": result.elapsed_seconds,
        })
        print(json.dumps(results[-1]), flush=True)

    # The rounding lemma must be tight: shrinking its bound to 127 must fail.
    source = render_rounding_lemma()[1]
    lemma = run(output / "control_rounding_lemma.c", source)
    tightened = source.replace("error >= -128 && error <= 128", "error >= -127 && error <= 127")
    tight_path = output / f"{len(cases):02d}_rounding_lemma_tightened.c"
    tight = run(tight_path, tightened)
    results.append({
        "case": "rounding_lemma_tightened", "control_status": lemma.status, "status": tight.status,
        "expected_assertion": "deployed rounding error",
        "named_assertion_violated": "deployed rounding error" in _violated(tight_path, tightened),
        "elapsed_seconds": tight.elapsed_seconds,
    })
    passed = all(
        row["control_status"] == "VERIFIED" and row["status"] == "FAILED" and row["named_assertion_violated"]
        for row in results
    )
    summary = {
        "schema": "crown_chain_v2_data_corruption_regression",
        "claim": "negative_test_not_certificate",
        "status": "PASSED" if passed else "FAILED",
        "certificate_dir": str(certificate_dir),
        "layer": layer,
        "chain": chain["certificate_id"],
        "expected": "control VERIFIED, corruption FAILED on the named assertion",
        "cases": results,
    }
    (output / "data_corruption_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate_dir", type=Path)
    parser.add_argument("--deployment-c", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memlimit", default="2g")
    parser.add_argument("--profile", default="paper-z3")
    args = parser.parse_args()
    summary = run_regression(
        args.certificate_dir, args.deployment_c, args.output, layer=args.layer,
        timeout=args.timeout, memlimit=args.memlimit, profile=args.profile,
    )
    print(json.dumps({"status": summary["status"], "cases": len(summary["cases"])}))


if __name__ == "__main__":
    main()
