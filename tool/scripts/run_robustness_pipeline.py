from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from synthesis.pipeline import RobustnessPipelineConfig, run_robustness_pipeline
from utils.logging_utils import configure_logging


def _parse_valid_labels(raw_value: str | None) -> tuple[int, ...] | None:
    if not raw_value:
        return None
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Quadapter robustness synthesis pipeline.")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashion-mnist", "iris", "seeds", "mnist64", "mnist_onnx"])
    parser.add_argument("--arch", default="1blk_100")
    parser.add_argument("--sample-id", "--sample_id", dest="sample_id", type=int, default=0)
    parser.add_argument("--bit-lb", "--bit_lb", dest="bit_lb", type=int, default=1)
    parser.add_argument("--bit-ub", "--bit_ub", dest="bit_ub", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--output-dir", "--outputPath", dest="output_dir", default="output")
    parser.add_argument("--if-relax", "--ifRelax", dest="if_relax", type=int, default=0)
    parser.add_argument("--preimage-mode", "--preimg_mode", dest="preimg_mode", default="milp", choices=["milp", "abstr", "comp"])
    parser.add_argument("--verify-mode", "--verify_mode", dest="verify_mode", default="milp", choices=["milp", "esbmc"])
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--valid-labels", default=None, help="Comma-separated valid output labels for the output property.")
    parser.add_argument("--compare-split", default="test", choices=["train", "test"])
    parser.add_argument("--compare-limit", type=int, default=100, help="Number of samples to compare for the QNN-vs-Keras report. Use 0 for all.")
    parser.add_argument("--skip-c-backend", action="store_true", help="Skip gcc compilation and C shared-library execution.")
    parser.add_argument("--compiler", default="gcc")
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
    )

    summary = run_robustness_pipeline(Path(__file__).resolve().parent.parent, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
