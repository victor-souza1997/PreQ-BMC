from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import tensorflow as tf

from utils.data.iris import load_train_test_data as load_iris_train_test_data
from utils.data.mnist_64 import load_train_test_data_mnist64
from utils.data.seeds import load_train_test_data_seeds

DatasetName = Literal["mnist", "fashion-mnist", "iris", "seeds", "mnist64", "mnist_onnx"]


@dataclass(frozen=True)
class DatasetBundle:
    """Materialized dataset tensors plus metadata needed by the pipeline."""

    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_scale: float
    clip_low: float
    clip_high: float

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[-1]) if self.x_train.ndim > 1 else int(self.x_train.size)

    @property
    def num_classes(self) -> int:
        return int(np.max(self.y_train)) + 1


def _ensure_numpy(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    return x, y


def _flatten_if_image(x: np.ndarray) -> np.ndarray:
    if x.ndim > 2:
        return x.reshape((x.shape[0], int(np.prod(x.shape[1:])))).astype(np.float32)
    return x.astype(np.float32)


def load_dataset(name: str) -> DatasetBundle:
    """Load one of the supported benchmark datasets with consistent preprocessing."""

    if name == "fashion-mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        input_scale = 255.0
    elif name in {"mnist", "mnist_onnx"}:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        input_scale = 255.0
    elif name == "mnist64":
        (x_train, x_test), (y_train, y_test) = load_train_test_data_mnist64()
        input_scale = 255.0
    elif name == "iris":
        (x_train, x_test), (y_train, y_test) = load_iris_train_test_data()
        input_scale = 1.0
    elif name == "seeds":
        (x_train, x_test), (y_train, y_test) = load_train_test_data_seeds()
        input_scale = 1.0
    else:
        raise ValueError(f"Unsupported dataset '{name}'.")

    x_train, y_train = _ensure_numpy(_flatten_if_image(np.asarray(x_train)), np.asarray(y_train))
    x_test, y_test = _ensure_numpy(_flatten_if_image(np.asarray(x_test)), np.asarray(y_test))

    return DatasetBundle(
        name=name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_scale=input_scale,
        clip_low=0.0,
        clip_high=input_scale,
    )


def select_split(dataset: DatasetBundle, split: Literal["train", "test"]) -> tuple[np.ndarray, np.ndarray]:
    """Return a dataset split as `(features, labels)`."""

    if split == "train":
        return dataset.x_train, dataset.y_train
    if split == "test":
        return dataset.x_test, dataset.y_test
    raise ValueError(f"Unsupported split '{split}'.")
