"""Model definitions and weight-loading helpers."""

from .deep_model import DeepDense, DeepLayer, DeepModel
from .loading import (
    build_and_load_deep_model,
    infer_dense_architecture_from_h5,
    parse_architecture,
    resolve_weight_path,
)

__all__ = [
    "DeepDense",
    "DeepLayer",
    "DeepModel",
    "build_and_load_deep_model",
    "infer_dense_architecture_from_h5",
    "parse_architecture",
    "resolve_weight_path",
]
