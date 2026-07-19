from __future__ import annotations

from typing import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Layer

if tf.__version__.startswith(("0.", "1.")):
    raise ValueError("TensorFlow 2.x is required.")


def deep_op_activation(x: tf.Tensor, if_output: bool) -> tf.Tensor:
    """Apply the model activation policy."""

    return x if if_output else tf.cast(tf.nn.relu(x), tf.float32)


class DeepModel(tf.keras.Model):
    """Simple dense ReLU stack used throughout the benchmark suite."""

    def __init__(
        self,
        layers: Iterable[int],
        dropout_rate: float = 0.0,
        last_layer_signed: bool = False,
        input_scale: float = 255.0,
    ) -> None:
        super().__init__()
        del last_layer_signed  # The final layer always emits signed logits in this model.
        self.input_scale = input_scale
        self.dropout_rate = float(dropout_rate)
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layers: list[DeepDense] = []

        layer_units = list(layers)
        for index, units in enumerate(layer_units):
            if not isinstance(units, int):
                raise ValueError(f"Unexpected layer width type: {type(units)!r}")
            self.dense_layers.append(
                DeepDense(
                    output_dim=units,
                    signed_output=index == len(layer_units) - 1,
                    if_output_layer=index == len(layer_units) - 1,
                )
            )

    def build(self, input_shape: tf.TensorShape) -> None:
        self._input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs: object) -> tf.Tensor:
        del kwargs
        x = tf.cast(inputs, tf.float32)
        if self.input_scale not in (None, 0):
            x = x / self.input_scale

        for dense_layer in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            x = dense_layer(x)
        return x


class DeepLayer(Layer):
    """Base layer kept for compatibility with the original codebase."""

    def __init__(self, input_bits: int | None = None, quantization_config: object | None = None, **kwargs: object) -> None:
        del input_bits, quantization_config
        super().__init__(**kwargs)


class DeepDense(DeepLayer):
    """Dense layer with ReLU on all hidden layers."""

    def __init__(
        self,
        output_dim: int,
        signed_output: bool = False,
        if_output_layer: bool = False,
        kernel_initializer: tf.keras.initializers.Initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(output_dim)
        self.kernel_initializer = kernel_initializer
        self.signed_output = signed_output
        self.if_output_layer = if_output_layer

    def build(self, input_shape: tf.TensorShape) -> None:
        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[1]), self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer=tf.keras.initializers.Constant(0.25),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        del training
        y = tf.matmul(x, self.kernel) + self.bias
        return deep_op_activation(y, self.if_output_layer)
