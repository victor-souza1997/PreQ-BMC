from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthesis.pipeline import RobustnessPipelineConfig, run_robustness_pipeline
from utils.logging_utils import configure_logging


def _parse_valid_labels(raw_value: str | None) -> tuple[int, ...] | None:
    if not raw_value:
        return None
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


def _optional_threshold(raw_value: float) -> float | None:
    return None if raw_value < 0 else float(raw_value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Quadapter robustness synthesis pipeline.")
    parser.add_argument(
        "--dataset",
        default="mnist",
        help=(
            "Base dataset name or benchmark alias. Examples: mnist, fashion-mnist, iris, seeds, "
            "iris_4x2, seeds_4x1."
        ),
    )
    parser.add_argument("--arch", default="1blk_100")
    parser.add_argument("--sample-id", "--sample_id", dest="sample_id", type=int, default=0)
    parser.add_argument("--bit-lb", "--bit_lb", dest="bit_lb", type=int, default=1)
    parser.add_argument("--bit-ub", "--bit_ub", dest="bit_ub", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--output-dir", "--outputPath", dest="output_dir", default="output")
    parser.add_argument("--if-relax", "--ifRelax", dest="if_relax", type=int, default=0)
    parser.add_argument("--preimage-mode", "--preimg_mode", dest="preimg_mode", default="milp", choices=["milp", "abstr", "comp"])
    parser.add_argument("--verify-mode", "--verify_mode", dest="verify_mode", default="milp", choices=["milp", "esbmc"])
    parser.add_argument(
        "--esbmc-layer-block-size",
        "--esbmc_layer_block_size",
        dest="esbmc_layer_block_size",
        type=int,
        default=10,
        help="Split hidden affine ESBMC verification into blocks of N output neurons. Use 0 for full-layer verification.",
    )
    parser.add_argument(
        "--blockwise-fail-fast",
        dest="blockwise_fail_fast",
        action="store_true",
        default=True,
        help="Reject a shared-QIF block-wise candidate as soon as one block fails.",
    )
    parser.add_argument(
        "--no-blockwise-fail-fast",
        dest="blockwise_fail_fast",
        action="store_false",
        help="Do not fail fast on the first failed block.",
    )
    parser.add_argument(
        "--blockwise-run-all-blocks-on-failure",
        dest="blockwise_run_all_blocks_on_failure",
        action="store_true",
        default=False,
        help="Run every block for a failing candidate for diagnostics.",
    )
    parser.add_argument(
        "--no-blockwise-run-all-blocks-on-failure",
        dest="blockwise_run_all_blocks_on_failure",
        action="store_false",
        help="Skip remaining blocks for a candidate after the first block failure.",
    )
    parser.add_argument("--esbmc-jobs", "--esbmc_jobs", dest="esbmc_jobs", type=int, default=1)
    parser.add_argument("--esbmc-memlimit", "--esbmc_memlimit", dest="esbmc_memlimit", default="6g")
    parser.add_argument(
        "--esbmc-profile",
        "--esbmc_profile",
        dest="esbmc_profile",
        default="paper-fast",
        choices=["paper-fast", "debug", "fast", "preimage", "safety", "overflow"],
    )
    parser.add_argument("--esbmc-timeout", "--esbmc_timeout", dest="esbmc_timeout_seconds", type=int, default=900)
    parser.add_argument("--gurobi-threads", "--gurobi_threads", dest="gurobi_threads", type=int, default=4)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--valid-labels", default=None, help="Comma-separated valid output labels for the output property.")
    parser.add_argument("--compare-split", default="test", choices=["train", "test"])
    parser.add_argument("--compare-limit", type=int, default=100, help="Number of samples to compare for the QNN-vs-Keras report. Use 0 for all.")
    parser.add_argument(
        "--enable-diagnostics",
        dest="enable_diagnostics",
        action="store_true",
        default=True,
        help="Include fixed-point saturation and semantic-gap diagnostics in the QNN-vs-Keras report.",
    )
    parser.add_argument(
        "--disable-diagnostics",
        dest="enable_diagnostics",
        action="store_false",
        help="Skip detailed fixed-point diagnostics in the QNN-vs-Keras report.",
    )
    parser.add_argument(
        "--formal-saturation-check",
        dest="formal_saturation_check",
        action="store_true",
        default=True,
        help="Require ESBMC no-saturation verification for fixed-point affine layers.",
    )
    parser.add_argument(
        "--no-formal-saturation-check",
        dest="formal_saturation_check",
        action="store_false",
        help="Disable ESBMC no-saturation verification as an acceptance criterion.",
    )
    parser.add_argument(
        "--empirical-saturation-check",
        dest="empirical_saturation_check",
        action="store_true",
        default=True,
        help="Use Python fixed-point saturation diagnostics as a deployment-quality acceptance criterion.",
    )
    parser.add_argument(
        "--no-empirical-saturation-check",
        dest="empirical_saturation_check",
        action="store_false",
        help="Do not reject/refine candidates based on empirical saturation diagnostics.",
    )
    parser.add_argument(
        "--accuracy-drop-threshold",
        type=float,
        default=0.05,
        help="Reject deployment candidates when Keras accuracy minus Python fixed-point accuracy exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=0.01,
        help="Reject deployment candidates when any layer saturation rate exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--mismatch-threshold",
        type=float,
        default=0.05,
        help="Reject deployment candidates when Python fixed-point mismatch rate versus quantized Keras exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument(
        "--max-quality-refinement-steps",
        type=int,
        default=10,
        help="Maximum deployment-quality bit refinement steps. Use 0 to keep the previous accept-after-synthesis behavior.",
    )
    parser.add_argument(
        "--export-paper-tables",
        dest="export_paper_tables",
        action="store_true",
        default=True,
        help="Export paper-ready experiment summary CSV tables.",
    )
    parser.add_argument(
        "--no-export-paper-tables",
        dest="export_paper_tables",
        action="store_false",
        help="Skip paper-ready CSV table export.",
    )
    parser.add_argument(
        "--baseline-results-json",
        type=Path,
        default=None,
        help="Optional JSON or CSV file with external Quadapter/CEG4N baseline results.",
    )
    parser.add_argument("--skip-c-backend", action="store_true", help="Skip gcc compilation and C shared-library execution.")
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument(
        "--no-gurobi",
        "--no_gurobi",
        action="store_true",
        help="Load cached preimage bounds instead of calling Gurobi. Requires --verify-mode esbmc.",
    )
    parser.add_argument(
        "--save-preimage-cache",
        "--save_preimage_cache",
        action="store_true",
        help="After computing the preimage with Gurobi, save it for later --no-gurobi runs.",
    )
    parser.add_argument(
        "--preimage-cache-dir",
        "--preimage_cache_dir",
        dest="preimage_cache_dir",
        default=None,
        help="Directory containing/exporting preimage cache entries. Defaults to OUTPUT_DIR/preimage_cache.",
    )
    parser.add_argument(
        "--preimage-cache-key",
        "--preimage_cache_key",
        dest="preimage_cache_key",
        default=None,
        help="Optional explicit preimage cache key. By default it is derived from dataset/arch/sample/eps/model.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(Path(args.output_dir) / "logs" / "robustness_pipeline.log", getattr(logging, args.log_level))
    config = RobustnessPipelineConfig(
        dataset=args.dataset,
        arch=args.arch,
        sample_id=args.sample_id,
        eps=args.eps,
        bit_lb=args.bit_lb,
        bit_ub=args.bit_ub,
        preimg_mode=args.preimg_mode,
        verify_mode=args.verify_mode,
        output_dir=Path(args.output_dir),
        if_relax=bool(args.if_relax),
        target_label=args.target_label,
        valid_labels=_parse_valid_labels(args.valid_labels),
        compare_split=args.compare_split,
        compare_limit=None if args.compare_limit == 0 else args.compare_limit,
        compile_c_backend=not args.skip_c_backend,
        compiler=args.compiler,
        enable_diagnostics=args.enable_diagnostics,
        formal_saturation_check=args.formal_saturation_check,
        empirical_saturation_check=args.empirical_saturation_check,
        accuracy_drop_threshold=_optional_threshold(args.accuracy_drop_threshold),
        saturation_threshold=_optional_threshold(args.saturation_threshold),
        mismatch_threshold=_optional_threshold(args.mismatch_threshold),
        max_quality_refinement_steps=max(0, int(args.max_quality_refinement_steps)),
        no_gurobi=args.no_gurobi,
        save_preimage_cache=args.save_preimage_cache,
        preimage_cache_dir=Path(args.preimage_cache_dir) if args.preimage_cache_dir is not None else None,
        preimage_cache_key=args.preimage_cache_key,
        esbmc_layer_block_size=max(0, int(args.esbmc_layer_block_size)),
        blockwise_fail_fast=bool(args.blockwise_fail_fast),
        blockwise_run_all_blocks_on_failure=bool(args.blockwise_run_all_blocks_on_failure),
        esbmc_jobs=max(1, int(args.esbmc_jobs)),
        esbmc_memlimit=str(args.esbmc_memlimit),
        esbmc_profile=args.esbmc_profile,
        esbmc_timeout_seconds=max(1, int(args.esbmc_timeout_seconds)),
        gurobi_threads=max(1, int(args.gurobi_threads)),
        export_paper_tables=args.export_paper_tables,
        baseline_results_json=args.baseline_results_json,
    )

    summary = run_robustness_pipeline(Path(__file__).resolve().parent.parent, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
