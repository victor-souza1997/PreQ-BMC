"""Generate and sequentially benchmark ESBMC affine-certificate harnesses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verification.conv_affine_certificate import (
    load_affine_certificate,
    render_affine_rounding_error_lemma,
    render_bounded_relu_hull_lemma,
    render_unrolled_recomputed_affine_block,
    render_symbolic_affine_block,
    slice_affine_certificate,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner


def _result(result, source: Path, certificate: Path, kind: str) -> dict:
    return {
        "certificate": str(certificate),
        "harness_kind": kind,
        "harness": str(source),
        "source_bytes": source.stat().st_size,
        "status": result.status,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_memory_bytes": result.peak_memory_bytes,
        "peak_memory_mib": (
            result.peak_memory_bytes / (1024 * 1024)
            if result.peak_memory_bytes is not None else None
        ),
        "cpu_time_seconds": result.cpu_time_seconds,
        "average_cpu_utilization_percent": result.average_cpu_utilization_percent,
        "command": list(result.command),
        "return_code": result.return_code,
        "stdout_log_path": result.stdout_log_path,
        "stderr_log_path": result.stderr_log_path,
        "resource_control": result.resource_control,
    }


def benchmark(paths: list[Path], output: Path, *, timeout: int, memlimit: str,
              profile: str, options: tuple[str, ...],
              neurons_per_harness: int | None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    runner = ESBMCRunner(ESBMCConfig(
        timeout_seconds=timeout, memlimit=memlimit, default_profile=profile,
    ))
    records = []
    for certificate_path in paths:
        certificate = load_affine_certificate(certificate_path)
        stem = certificate_path.stem
        sources = []
        if "rounding" in options:
            sources.append(("rounding_error_lemma", render_affine_rounding_error_lemma(certificate)))
        if "relu" in options:
            sources.append(("bounded_relu_hull_lemma", render_bounded_relu_hull_lemma(certificate)))
        step = neurons_per_harness or len(certificate["block"])
        if step <= 0:
            raise ValueError("neurons_per_harness must be positive")
        for start in range(0, len(certificate["block"]), step):
            end = min(start + step, len(certificate["block"]))
            selected = slice_affine_certificate(certificate, range(start, end))
            suffix = f"_n{start}_{end}"
            if "option_b" in options:
                sources.append((f"option_b_recomputation{suffix}", render_unrolled_recomputed_affine_block(selected)))
            if "option_a" in options:
                sources.append((f"option_a_symbolic{suffix}", render_symbolic_affine_block(selected)))
        for kind, source in sources:
            source_path = output / f"{stem}_{kind}.c"
            source_path.write_text(source, encoding="utf-8")
            result = runner.run_file(source_path, profile=profile)
            record = _result(result, source_path, certificate_path, kind)
            records.append(record)
            print(json.dumps({key: record[key] for key in (
                "certificate", "harness_kind", "status", "elapsed_seconds",
                "peak_memory_mib", "source_bytes",
            )}), flush=True)
    summary = {
        "schema": "affine_certificate_esbmc_benchmark_v1",
        "claim": "checker_cost_measurement_not_network_certificate",
        "execution": "sequential",
        "timeout_seconds": timeout,
        "memlimit": memlimit,
        "profile": profile,
        "options": list(options),
        "neurons_per_harness": neurons_per_harness,
        "records": records,
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificates", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memlimit", default="4g")
    parser.add_argument("--profile", default="paper-z3")
    parser.add_argument(
        "--options", nargs="+",
        choices=("relu", "rounding", "option_a", "option_b"),
        default=("relu", "rounding", "option_b", "option_a"),
    )
    args = parser.parse_args()
    parser.add_argument("--neurons-per-harness", type=int)
    benchmark(args.certificates, args.output, timeout=args.timeout,
              memlimit=args.memlimit, profile=args.profile,
              options=tuple(args.options),
              neurons_per_harness=args.neurons_per_harness)


if __name__ == "__main__":
    main()
