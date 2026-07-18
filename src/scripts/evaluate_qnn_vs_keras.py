from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backends.fixed_point import LayerQuantizationSpec, build_fixed_point_network, clone_quantized_keras_model
from datasets.loaders import load_dataset
from models.loading import (
    build_and_load_deep_model,
    infer_dense_architecture_from_h5,
    normalize_dataset_selection,
    parse_architecture,
    resolve_weight_path,
)
from synthesis.pipeline import compare_qnn_to_keras
from utils.logging_utils import configure_logging


def _load_quantization_specs(config_path: Path) -> tuple[str, str, list[LayerQuantizationSpec]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    arch = payload["arch"]
    synthesis = payload["synthesis"]
    if not synthesis.get("success"):
        raise ValueError(f"Quantization config at {config_path} does not contain a successful synthesis result.")
    specs = [
        LayerQuantizationSpec(
            total_bits=synthesis["total_bits"][index],
            integer_bits=synthesis["integer_bits"][index],
            fractional_bits=synthesis["fractional_bits"][index],
        )
        for index in range(len(synthesis["total_bits"]))
    ]
    return dataset, arch, specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the generated fixed-point QNN against the quantized Keras model.")
    parser.add_argument("--quant-config", required=True, type=Path, help="Path to the quantization_config.json file produced by the pipeline.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate. Use 0 for the full split.")
    parser.add_argument("--skip-c-backend", action="store_true")
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(Path(args.output_dir) / "logs" / "evaluate_qnn_vs_keras.log", getattr(logging, args.log_level))

    dataset_name, arch, layer_specs = _load_quantization_specs(args.quant_config)
    repo_root = Path(__file__).resolve().parent.parent
    selection = normalize_dataset_selection(dataset_name)
    dataset = load_dataset(selection.base_name)
    weights_path = resolve_weight_path(repo_root, dataset_name, arch)
    inferred_arch = infer_dense_architecture_from_h5(weights_path)
    input_dim = inferred_arch[0] if inferred_arch else dataset.input_dim
    layer_units = inferred_arch[1:] if inferred_arch else parse_architecture(arch, dataset.num_classes)

    model = build_and_load_deep_model(
        input_dim=input_dim,
        layer_units=layer_units,
        weights_path=weights_path,
        input_scale=dataset.input_scale,
    )
    quantized_model = clone_quantized_keras_model(model, layer_specs)
    fixed_point_network = build_fixed_point_network(model, layer_specs)

    comparison = compare_qnn_to_keras(
        dataset=dataset,
        quantized_model=quantized_model,
        fixed_point_network=fixed_point_network,
        split=args.split,
        limit=None if args.limit == 0 else args.limit,
        output_dir=Path(args.output_dir),
        compile_c_backend=not args.skip_c_backend,
        compiler=args.compiler,
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
