"""Check that one-value CROWN certificate corruptions are rejected by ESBMC."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

from verification.crown_chain_certificate import (
    load_crown_chain_certificate,
    render_affine_step,
    render_relu_step,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner


Mutation = Callable[[dict[str, Any]], tuple[str, str]]


def _first_chain(data: dict[str, Any]) -> dict[str, Any]:
    return data["chains"][0]


def _mutate_affine_output(data: dict[str, Any]) -> tuple[str, str]:
    step = _first_chain(data)["steps"][0]
    step["output_coefficients"]["values"][0] += 1
    return step["step_id"], "affine_chunk"


def _mutate_relu_constant(data: dict[str, Any]) -> tuple[str, str]:
    step = _first_chain(data)["steps"][1]
    if not step["relu_constants"]["values"]:
        raise ValueError("Pilot ReLU step has no exported constant")
    step["relu_constants"]["values"][0] += 1
    return step["step_id"], "relu_chunk"


def _mutate_affine_constant(data: dict[str, Any]) -> tuple[str, str]:
    step = _first_chain(data)["steps"][0]
    step["output_constant"] += 1
    return step["step_id"], "affine_close"


def _mutate_bound_lower(data: dict[str, Any]) -> tuple[str, str]:
    step = _first_chain(data)["steps"][1]
    bound_id = step["bound_ids"][0]
    record = next(
        item for item in data["bound_records"] if item["bound_id"] == bound_id
    )
    record["lower"] += 1
    data["bounds_by_id"][bound_id] = record
    return step["step_id"], "relu_chunk"


def _render_selected(
    data: dict[str, Any],
    step_id: str,
    kind: str,
) -> tuple[str, str]:
    chain = _first_chain(data)
    step = next(item for item in chain["steps"] if item["step_id"] == step_id)
    if step["kind"] == "affine":
        rendered = render_affine_step(data, chain, step)
        return next(
            (name, source)
            for name, source, metadata in rendered
            if metadata["kind"] == kind
        )
    rendered = render_relu_step(data, chain, step)
    return next(
        (name, source)
        for name, source, metadata in rendered
        if metadata["kind"] == kind
    )


def run_regressions(
    certificate_path: Path,
    output: Path,
    *,
    timeout: int = 30,
    memlimit: str = "4g",
    profile: str = "paper-z3",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    original = load_crown_chain_certificate(certificate_path)
    runner = ESBMCRunner(ESBMCConfig(
        timeout_seconds=timeout,
        memlimit=memlimit,
        default_profile=profile,
    ))
    cases: tuple[tuple[str, Mutation], ...] = (
        ("affine_a_out", _mutate_affine_output),
        ("relu_d_i", _mutate_relu_constant),
        ("affine_output_constant", _mutate_affine_constant),
        ("bound_record_lower", _mutate_bound_lower),
    )
    records = []
    for case_name, mutate in cases:
        clean_data = deepcopy(original)
        mutated_data = deepcopy(original)
        step_id, kind = mutate(mutated_data)
        clean_name, clean_source = _render_selected(clean_data, step_id, kind)
        corrupt_name, corrupt_source = _render_selected(
            mutated_data, step_id, kind
        )
        case_root = output / case_name
        case_root.mkdir()
        clean_path = case_root / f"clean_{clean_name}"
        corrupt_path = case_root / f"corrupt_{corrupt_name}"
        clean_path.write_text(clean_source, encoding="utf-8")
        corrupt_path.write_text(corrupt_source, encoding="utf-8")
        clean = runner.run_file(clean_path, profile=profile)
        corrupt = runner.run_file(corrupt_path, profile=profile)
        records.append({
            "case": case_name,
            "step_id": step_id,
            "harness_kind": kind,
            "clean": {
                key: value for key, value in asdict(clean).items()
                if key not in {"stdout", "stderr"}
            },
            "corrupt": {
                key: value for key, value in asdict(corrupt).items()
                if key not in {"stdout", "stderr"}
            },
            "passed": clean.status == "VERIFIED" and corrupt.status == "FAILED",
        })
    summary = {
        "schema": "crown_chain_corruption_regression_v1",
        "claim": "non_vacuity_regression_not_network_certificate",
        "certificate": str(certificate_path),
        "passed": all(record["passed"] for record in records),
        "records": records,
    }
    (output / "corruption_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--memlimit", default="4g")
    parser.add_argument("--profile", default="paper-z3")
    args = parser.parse_args()
    summary = run_regressions(
        args.certificate,
        args.output,
        timeout=args.timeout,
        memlimit=args.memlimit,
        profile=args.profile,
    )
    print(json.dumps({
        "passed": summary["passed"],
        "cases": [
            {
                "case": record["case"],
                "clean": record["clean"]["status"],
                "corrupt": record["corrupt"]["status"],
            }
            for record in summary["records"]
        ],
    }))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
