from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from .deep_model import DeepModel

SUPPORTED_DATASETS = ("mnist", "fashion-mnist", "iris", "seeds", "mnist64", "mnist_onnx")


@dataclass(frozen=True)
class DatasetSelection:
    """Normalized dataset request for benchmark aliases such as `iris_4x2`."""

    requested_name: str
    base_name: str
    benchmark_name: str | None = None


def infer_dense_architecture_from_h5(weight_file: str | Path) -> list[int]:
    """Infer `[input_dim, hidden..., output_dim]` from a Keras HDF5 weights file."""

    weight_path = Path(weight_file)
    if not weight_path.exists() or weight_path.suffix.lower() != ".h5":
        return []

    kernel_shapes: list[tuple[int, int]] = []
    with h5py.File(weight_path, "r") as handle:
        layer_names = handle.attrs.get("layer_names", [])
        if isinstance(layer_names, np.ndarray):
            layer_names = layer_names.tolist()
        for raw_name in layer_names:
            layer_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else raw_name
            if layer_name not in handle:
                continue
            layer_group = handle[layer_name]
            weight_names = layer_group.attrs.get("weight_names", [])
            if isinstance(weight_names, np.ndarray):
                weight_names = weight_names.tolist()
            for raw_weight_name in weight_names:
                weight_name = raw_weight_name.decode("utf-8") if isinstance(raw_weight_name, bytes) else raw_weight_name
                dataset_key = weight_name.split("/", maxsplit=1)[-1]
                if not dataset_key.endswith("kernel:0"):
                    continue
                if dataset_key in layer_group:
                    dataset = layer_group[dataset_key]
                elif weight_name in layer_group:
                    dataset = layer_group[weight_name]
                else:
                    continue
                kernel_shapes.append(tuple(int(dim) for dim in dataset.shape))

    if not kernel_shapes:
        return []

    architecture = [kernel_shapes[0][0]]
    architecture.extend(shape[1] for shape in kernel_shapes)
    return architecture


def parse_architecture(arch: str, num_classes: int) -> list[int]:
    """Parse an architecture string such as `2blk_100_50` into layer widths."""

    parts = arch.split("_")
    if not parts:
        raise ValueError("Architecture cannot be empty.")

    widths = [int(part) for part in parts[1:] if part]
    if not widths:
        raise ValueError(f"Unable to parse layer widths from architecture '{arch}'.")
    if widths[-1] != num_classes:
        widths.append(num_classes)
    return widths


def normalize_dataset_selection(dataset_name: str) -> DatasetSelection:
    """Map benchmark aliases like `iris_4x2` to their base dataset."""

    if dataset_name in SUPPORTED_DATASETS:
        return DatasetSelection(
            requested_name=dataset_name,
            base_name=dataset_name,
            benchmark_name=None,
        )

    for base_name in sorted(SUPPORTED_DATASETS, key=len, reverse=True):
        prefix = f"{base_name}_"
        if dataset_name.startswith(prefix):
            return DatasetSelection(
                requested_name=dataset_name,
                base_name=base_name,
                benchmark_name=dataset_name,
            )

    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. "
        f"Use a base dataset ({supported}) or a benchmark alias like 'iris_4x2' or 'seeds_4x1'."
    )


def list_available_benchmarks(root_dir: Path, base_name: str) -> list[str]:
    """List available benchmark weight stems for a base dataset directory."""

    dataset_dir = root_dir / "benchmark" / base_name
    if not dataset_dir.exists():
        return []
    names: list[str] = []
    for path in sorted(dataset_dir.glob("*_weight.h5")):
        stem = path.name.removesuffix("_weight.h5")
        names.append(stem)
    return names


def resolve_weight_path(root_dir: Path, dataset_name: str, arch: str) -> Path:
    """Resolve the benchmark weights path for the requested dataset and architecture."""

    selection = normalize_dataset_selection(dataset_name)
    if selection.benchmark_name is not None:
        benchmark_path = root_dir / "benchmark" / selection.base_name / f"{selection.benchmark_name}_weight.h5"
        if benchmark_path.exists():
            return benchmark_path
        available = ", ".join(list_available_benchmarks(root_dir, selection.base_name))
        raise FileNotFoundError(
            f"Benchmark '{selection.benchmark_name}' was not found under benchmark/{selection.base_name}. "
            f"Available benchmarks: {available or '(none)'}"
        )

    dataset_dir = root_dir / "benchmark" / selection.base_name
    if not dataset_dir.exists():
        dataset_dir = root_dir / "benchmark" / selection.base_name.split("_")[0]
    if selection.base_name in {"iris", "seeds", "mnist64", "mnist_onnx"}:
        arch_candidate = dataset_dir / f"{selection.base_name}_{arch}_weight.h5"
        if arch_candidate.exists():
            return arch_candidate
        return dataset_dir / f"{selection.base_name}_weight.h5"
    return dataset_dir / f"{selection.base_name}_{arch}_weight.h5"


def build_and_load_deep_model(
    input_dim: int,
    layer_units: list[int],
    weights_path: Path,
    input_scale: float,
) -> DeepModel:
    """Instantiate `DeepModel`, materialize variables, then load weights."""

    model = DeepModel(layer_units, last_layer_signed=True, input_scale=input_scale)
    model.build((None, input_dim))
    _ = model(tf.zeros((1, input_dim), dtype=tf.float32))
    model.load_weights(str(weights_path))
    return model
