from __future__ import annotations

import copy

import numpy as np


def forward_dnn(x: np.ndarray, ilp_model: object) -> np.ndarray:
    """Forward propagate through the floating-point model stored in the synthesizer."""

    model = ilp_model.deep_model
    all_layers = copy.copy(ilp_model.dense_layers)
    all_layers.append(ilp_model.output_layer)
    current = np.asarray(x, dtype=np.float32)

    for index, encoded_layer in enumerate(all_layers):
        tf_layer = model.dense_layers[index]
        weights, bias = tf_layer.get_weights()
        out_values: list[np.float32] = []
        before_relu: list[np.float32] = []
        is_output_layer = index == len(ilp_model.dense_layers)

        for out_index in range(encoded_layer.layer_size):
            weight_row = np.float32(weights[:, out_index])
            bias_value = np.float32(bias[out_index])
            accumulator = np.float32(np.array(weight_row * current).sum() + bias_value)
            before_relu.append(accumulator)
            if not is_output_layer and accumulator < 0:
                accumulator = np.float32(0)
            out_values.append(accumulator)

        encoded_layer.set_realVal(before_relu)
        current = np.asarray(out_values, dtype=np.float32)

    return current


def forward_dnn_multi(x_set: np.ndarray, ilp_model: object) -> None:
    """Multi-sample version kept for the backdoor flow."""

    model = ilp_model.deep_model
    all_layers = copy.copy(ilp_model.dense_layers)
    all_layers.append(ilp_model.output_layer)

    for original_x in x_set:
        current = np.asarray(original_x, dtype=np.float32)
        for index, encoded_layer in enumerate(all_layers):
            tf_layer = model.dense_layers[index]
            weights, bias = tf_layer.get_weights()
            out_values: list[np.float32] = []
            before_relu: list[np.float32] = []
            is_output_layer = index == len(ilp_model.dense_layers)

            for out_index in range(encoded_layer.layer_size):
                weight_row = np.float32(weights[:, out_index])
                bias_value = np.float32(bias[out_index])
                accumulator = np.float32(np.array(weight_row * current).sum() + bias_value)
                before_relu.append(accumulator)
                if not is_output_layer and accumulator < 0:
                    accumulator = np.float32(0)
                out_values.append(accumulator)

            encoded_layer.set_realVal_multi(before_relu)
            current = np.asarray(out_values, dtype=np.float32)
