"""Corrupt one assertion per v2 obligation class and require ESBMC failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from verification.esbmc import ESBMCConfig, ESBMCRunner


_RELU_CLASSES = ("bound_free", "nosat_upper", "lower", "upper", "lower_upper")


def _corrupt(source: str, needle: str | None) -> str:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("  __ESBMC_assert(") or (needle and needle not in line):
            continue
        match = re.search(r', (".*"\);)$', line)
        if match is None:
            continue
        lines[index] = f"  __ESBMC_assert(0, {match.group(1)}"
        return "\n".join(lines) + "\n"
    raise ValueError(f"No assertion found for corruption needle {needle!r}")


def run_regression(
    check_dir: Path,
    output: Path,
    *,
    timeout: int = 30,
    memlimit: str = "1g",
    profile: str = "paper-z3",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((check_dir / "harnesses" / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    requested: list[tuple[str, dict[str, Any], str | None]] = []
    for kind in ("affine_chunk", "affine_close", "concretize", "chain_close"):
        requested.append((kind, next(record for record in records if record["kind"] == kind), None))
    relu = next(record for record in records if record["kind"] == "relu_chunk" and all(
        record["class_counts"].get(name, 0) for name in _RELU_CLASSES
    ))
    requested.extend((f"relu_{name}", relu, f" {name} instance ") for name in _RELU_CLASSES)
    for name in _RELU_CLASSES:
        kind = f"relu_lemma_{name}"
        requested.append((kind, next(record for record in records if record["kind"] == kind), None))
    requested.append((
        "relu_lemma_positive_product",
        next(record for record in records if record["kind"] == "relu_lemma_positive_product"),
        None,
    ))
    runner = ESBMCRunner(ESBMCConfig(
        timeout_seconds=timeout, memlimit=memlimit, default_profile=profile
    ))
    results = []
    for ordinal, (name, record, needle) in enumerate(requested):
        original = Path(record["harness"])
        corrupted = output / f"{ordinal:02d}_{name}.c"
        corrupted.write_text(_corrupt(original.read_text(encoding="utf-8"), needle), encoding="utf-8")
        result = runner.run_file(corrupted, profile=profile)
        results.append({
            "obligation_class": name,
            "source_harness": str(original),
            "corrupted_harness": str(corrupted),
            "status": result.status,
            "elapsed_seconds": result.elapsed_seconds,
            "stdout_log_path": result.stdout_log_path,
            "stderr_log_path": result.stderr_log_path,
        })
    status = "PASSED" if all(row["status"] == "FAILED" for row in results) else "FAILED"
    summary = {
        "schema": "crown_chain_v2_corruption_regression",
        "claim": "negative_test_not_certificate",
        "status": status,
        "expected_status_per_case": "FAILED",
        "cases": results,
    }
    (output / "corruption_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--memlimit", default="1g")
    parser.add_argument("--profile", default="paper-z3")
    args = parser.parse_args()
    summary = run_regression(
        args.check_dir, args.output,
        timeout=args.timeout, memlimit=args.memlimit, profile=args.profile,
    )
    print(json.dumps({"status": summary["status"], "cases": len(summary["cases"])}))


if __name__ == "__main__":
    main()
