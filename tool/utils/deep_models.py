from tensorflow.keras.layers import Layer
# from backward_pkg.utils_quadapter import *
import tensorflow as tf
import numpy as np

if tf.__version__.startswith("1.") or tf.__version__.startswith("0."):
    raise ValueError("Please upgrade to TensorFlow 2.x")


def deep_op_activation(x, if_output):
    # return x
    if if_output:
        return x
    else:
        return tf.cast(tf.nn.relu(x), tf.float32)


class DeepModel(tf.keras.Model):
    def __init__(
            self,
            layers,
            dropout_rate=0.0,
            last_layer_signed=False,
            input_scale=255.0,
    ):

        super(DeepModel, self).__init__()
        self.dense_layers = []
        self.input_scale = input_scale
        for i, l in enumerate(layers):
            if type(l) == int:
                signed = (
                    True if i == len(layers) - 1 else False
                )
                last_layer = (True if i == len(layers) - 1 else False)
                self.dense_layers.append(
                    DeepDense(
                        output_dim=l,
                        signed_output=signed,
                        if_output_layer=last_layer
                    )
                )
            else:
                raise ValueError("Unexpected type {} ({})".format(type(l), str(l)))
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.flatten_layer = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        super(DeepModel, self).build(
            input_shape
        )

    def call(self, inputs, **kwargs):
        x = tf.cast(inputs, tf.float32)
        if self.input_scale not in (None, 0):
            x = x / self.input_scale

        for c in self.dense_layers:
            if self.dropout_rate > 0.0:
                x = self.dropout_layer(x)
            x = c(x)
        return x


class DeepLayer(Layer):
    def __init__(self, input_bits=None, quantization_config=None, **kwargs):
        super(DeepLayer, self).__init__(**kwargs)


class DeepDense(DeepLayer):
    def __init__(
            self,
            output_dim,
            signed_output=False,
            if_output_layer=False,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
            **kwargs
    ):
        self.units = output_dim
        self.kernel_initializer = kernel_initializer

        self.signed_output = signed_output
        self.if_output_layer = if_output_layer
        super(DeepDense, self).__init__(**kwargs)

    def build(self, input_shape):
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

        super(DeepDense, self).build(
            input_shape
        )

    def call(self, x, training=None):
        y = tf.matmul(x, self.kernel) + self.bias
        y = deep_op_activation(y, self.if_output_layer)
        return y
