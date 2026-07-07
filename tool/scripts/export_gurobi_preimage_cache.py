from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loaders import load_dataset
from models.loading import (
    build_and_load_deep_model,
    infer_dense_architecture_from_h5,
    normalize_dataset_selection,
    parse_architecture,
    resolve_weight_path,
)
from synthesis.forward import forward_dnn
from synthesis.preimage_cache import build_preimage_cache_identity
from synthesis.preqbmc import GPEncoding, QuadapterConfig
from utils.logging_utils import configure_logging, get_logger
from verification.properties import ClassificationProperty

LOGGER = get_logger(__name__)


def _parse_csv(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _parse_int_csv(raw_value: str | None) -> list[int]:
    values = _parse_csv(raw_value)
    return [int(value) for value in values] if values else []


def _arch_from_alias_arch(raw_arch: str) -> str:
    if "blk" in raw_arch or "_" in raw_arch:
        return raw_arch
    if "x" not in raw_arch:
        return raw_arch
    width_raw, depth_raw = raw_arch.split("x", maxsplit=1)
    width = int(width_raw)
    depth = int(depth_raw)
    return f"{depth}blk_" + "_".join(str(width) for _ in range(depth))


def _arch_from_weight_stem(dataset: str, stem: str, weights_path: Path) -> str | None:
    prefix = f"{dataset}_"
    if stem.startswith(prefix):
        return _arch_from_alias_arch(stem.removeprefix(prefix))
    if stem != dataset:
        return None

    inferred_arch = infer_dense_architecture_from_h5(weights_path)
    if len(inferred_arch) >= 2:
        hidden = inferred_arch[1:-1]
        if hidden:
            return f"{len(hidden)}blk_" + "_".join(str(width) for width in hidden)
        return f"1blk_{inferred_arch[-1]}"
    return None


def discover_jobs(repo_root: Path, selected_datasets: set[str] | None) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    benchmark_root = repo_root / "benchmark"
    explicit_aliases: set[str] = set()
    selected_base_datasets: set[str] | None = None
    if selected_datasets is not None:
        selected_base_datasets = set()
        for selected in selected_datasets:
            selection = normalize_dataset_selection(selected)
            if selection.benchmark_name is not None:
                explicit_aliases.add(selection.benchmark_name)
            else:
                selected_base_datasets.add(selection.base_name)

    for weights_path in sorted(benchmark_root.glob("*/*_weight.h5")):
        dataset = weights_path.parent.name
        benchmark_alias = weights_path.name.removesuffix("_weight.h5")
        arch = _arch_from_weight_stem(dataset, benchmark_alias, weights_path)
        if arch is None:
            LOGGER.warning("Skipping %s because no pipeline architecture string could be inferred.", weights_path)
            continue
        if selected_datasets is not None:
            if benchmark_alias in explicit_aliases:
                jobs.append((benchmark_alias, arch))
                continue
            if selected_base_datasets is not None and dataset in selected_base_datasets:
                jobs.append((dataset, arch))
                continue
            continue
        jobs.append((dataset, arch))
    return jobs


def _predict_logits(model: Any, features: np.ndarray) -> np.ndarray:
    logits = model(np.asarray(features, dtype=np.float32), training=False)
    return np.asarray(logits.numpy(), dtype=np.float64)


def export_one(
    *,
    repo_root: Path,
    dataset_name: str,
    arch: str,
    sample_id: int,
    eps: float,
    bit_lb: int,
    bit_ub: int,
    preimg_mode: str,
    if_relax: bool,
    cache_dir: Path,
    target_label: int | None,
    valid_labels: tuple[int, ...] | None,
) -> Path:
    selection = normalize_dataset_selection(dataset_name)
    dataset = load_dataset(selection.base_name)
    weights_path = resolve_weight_path(repo_root, dataset_name, arch)
    inferred_arch = infer_dense_architecture_from_h5(weights_path)
    input_dim = dataset.input_dim
    num_classes = dataset.num_classes
    layer_units = parse_architecture(arch, num_classes)
    if inferred_arch:
        input_dim = inferred_arch[0]
        layer_units = inferred_arch[1:]
        num_classes = layer_units[-1]

    model = build_and_load_deep_model(
        input_dim=input_dim,
        layer_units=layer_units,
        weights_path=weights_path,
        input_scale=dataset.input_scale,
    )

    sample = dataset.x_test[sample_id]
    sample_label = int(dataset.y_test[sample_id])
    sample_logits = _predict_logits(model, np.expand_dims(sample, axis=0))[0]
    predicted_label = int(np.argmax(sample_logits))
    if predicted_label != sample_label:
        raise ValueError(
            f"{dataset_name}/{arch} sample {sample_id} is misclassified by the reference model "
            f"(pred={predicted_label}, label={sample_label})."
        )

    x_low = np.clip(sample - eps, dataset.clip_low, dataset.clip_high)
    x_high = np.clip(sample + eps, dataset.clip_low, dataset.clip_high)
    property_spec = ClassificationProperty(
        target_label=target_label if target_label is not None else predicted_label,
        valid_labels=valid_labels,
    )
    property_spec.validate(num_classes)

    cache_key, cache_metadata = build_preimage_cache_identity(
        dataset=dataset_name,
        arch=arch,
        sample_id=sample_id,
        eps=eps,
        preimg_mode=preimg_mode,
        if_relax=if_relax,
        target_label=int(property_spec.target_label if property_spec.target_label is not None else predicted_label),
        valid_labels=property_spec.valid_labels,
        weights_path=weights_path,
    )

    config = QuadapterConfig(
        bit_lb=bit_lb,
        bit_ub=bit_ub,
        preimg_mode=preimg_mode,
        verify_mode="esbmc",
        sample_id=sample_id,
        eps=eps,
        output_dir=cache_dir / "_scratch",
        if_relax=if_relax,
        preimage_cache_dir=cache_dir,
        preimage_cache_key=cache_key,
        preimage_cache_metadata=cache_metadata,
        solver="gurobi",
    )
    synthesizer = GPEncoding(
        arch=[input_dim] + layer_units,
        model=model,
        config=config,
        original_prediction=predicted_label,
        x_low_real=np.asarray(x_low / dataset.input_scale, dtype=np.float32),
        x_high_real=np.asarray(x_high / dataset.input_scale, dtype=np.float32),
        property_spec=property_spec,
    )

    forward_dnn(np.asarray(sample / dataset.input_scale, dtype=np.float32), synthesizer)
    x_low_real = np.asarray(x_low / dataset.input_scale, dtype=np.float32)
    x_high_real = np.asarray(x_high / dataset.input_scale, dtype=np.float32)
    synthesizer.assert_input_box(x_low_real, x_high_real)
    synthesizer.symbolic_propagate()

    other_max = max(
        value
        for index, value in enumerate(synthesizer.output_layer.ub)
        if index != synthesizer.targetCls
    )
    if synthesizer.output_layer.lb[synthesizer.targetCls] < other_max:
        raise ValueError(f"The original DNN property does not hold for {dataset_name}/{arch} sample {sample_id}.")

    synthesizer.backward_preimage_computation()
    return synthesizer.save_cached_preimage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Gurobi preimage caches for later --no-gurobi runs.")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated base datasets or benchmark aliases, e.g. iris or iris_15x2. Default: all benchmark folders.",
    )
    parser.add_argument("--archs", default=None, help="Comma-separated architectures. Default: all discovered for each dataset.")
    parser.add_argument("--sample-ids", "--sample_ids", default="0", help="Comma-separated test sample ids. Default: 0.")
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--bit-lb", "--bit_lb", dest="bit_lb", type=int, default=1)
    parser.add_argument("--bit-ub", "--bit_ub", dest="bit_ub", type=int, default=16)
    parser.add_argument("--preimage-mode", "--preimg_mode", dest="preimg_mode", default="milp", choices=["milp", "abstr", "comp"])
    parser.add_argument("--if-relax", "--ifRelax", dest="if_relax", type=int, default=0)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--valid-labels", default=None, help="Comma-separated valid output labels.")
    parser.add_argument("--cache-dir", "--preimage-cache-dir", dest="cache_dir", default="output/preimage_cache")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--continue-on-error", action="store_true", help="Continue exporting other jobs if one job fails.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = Path(args.cache_dir)
    configure_logging(cache_dir / "export_preimage_cache.log", getattr(logging, args.log_level))

    selected_datasets = set(_parse_csv(args.datasets)) if args.datasets else None
    selected_archs = {_arch_from_alias_arch(value) for value in _parse_csv(args.archs)} if args.archs else None
    sample_ids = _parse_int_csv(args.sample_ids) or [0]
    valid_labels = tuple(_parse_int_csv(args.valid_labels)) if args.valid_labels else None

    jobs = discover_jobs(repo_root, selected_datasets)
    if selected_archs is not None:
        filtered_jobs = [(dataset, arch) for dataset, arch in jobs if arch in selected_archs]
        has_explicit_alias = False
        if selected_datasets is not None:
            has_explicit_alias = any(
                normalize_dataset_selection(dataset).benchmark_name is not None for dataset in selected_datasets
            )
        if filtered_jobs or not has_explicit_alias:
            jobs = filtered_jobs
        else:
            LOGGER.warning(
                "Ignoring --archs=%s because benchmark aliases already identify the architecture.",
                ",".join(sorted(selected_archs)),
            )
    if not jobs:
        raise ValueError("No benchmark jobs matched the requested dataset/architecture filters.")

    failures: list[str] = []
    for dataset, arch in jobs:
        for sample_id in sample_ids:
            label = f"{dataset}/{arch}/sample{sample_id}"
            try:
                path = export_one(
                    repo_root=repo_root,
                    dataset_name=dataset,
                    arch=arch,
                    sample_id=sample_id,
                    eps=args.eps,
                    bit_lb=args.bit_lb,
                    bit_ub=args.bit_ub,
                    preimg_mode=args.preimg_mode,
                    if_relax=bool(args.if_relax),
                    cache_dir=cache_dir,
                    target_label=args.target_label,
                    valid_labels=valid_labels,
                )
                LOGGER.info("Exported %s to %s", label, path)
                print(f"exported {label}: {path}")
            except Exception as exc:
                failures.append(f"{label}: {exc}")
                LOGGER.exception("Failed to export %s", label)
                if not args.continue_on_error:
                    raise
                print(f"failed {label}: {exc}")

    if failures:
        raise SystemExit("Some preimage cache exports failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
