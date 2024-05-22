import tensorflow as tf
from tensorflow import keras
from keras import Model
import numpy as np
from layers.conv_stack import ConvStack
from layers.dense_stack import DenseStack
from layers.noisy_dense import NoisyDense
from agent_configs import RainbowConfig

from utils import prepare_kernel_initializers

# from noisy_conv2d import NoisyConv2D


class Network(Model):
    def __init__(
        self, config: DuelingDQNConfig, output_size: int, input_shape, *args, **kwargs
    ):
        super().__init__()
        self.config = config

        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = config.dense_layers > 0

        # Convert the config into a list of filters, kernel_sizes, and strides (could put in utils?)
        filters = []
        kernel_sizes = []
        strides = []
        for filter, kernel_size, stride in config.conv_layers:
            filters.append(filter)
            kernel_sizes.append(kernel_size)
            strides.append(stride)

        if self.has_conv_layers:
            self.conv_layers = ConvStack(
                input_shape,
                filters,
                kernel_sizes,
                strides,
                self.config.activation,
                self.config.kernel_initializer,
            )

        widths = [config.width] * config.dense_layers
        if self.has_dense_layers:
            self.dense_layers = DenseStack(
                widths,
                self.config.activation,
                self.config.kernel_initializer,
            )

        self.has_value_hidden_layers = config.value_hidden_layers > 0
        if self.has_value_hidden_layers:
            widths = [config.width] * config.value_hidden_layers
            self.value_hidden_layers = DenseStack(
                widths,
                self.config.activation,
                self.config.kernel_initializer,
            )

        self.value = tf.keras.layers.Dense(
            1,
            kernel_initializer=prepare_kernel_initializers(config.kernel_initializer),
            activation="linear",
            name="HiddenV",
        )

        self.has_advantage_hidden_layers = config.advantage_hidden_layers > 0
        if self.has_advantage_hidden_layers:
            widths = [config.width] * config.advantage_hidden_layers
            self.advantage_hidden_layers = DenseStack(
                widths,
                self.config.activation,
                self.config.kernel_initializer,
            )

        self.advantage = tf.keras.layers.Dense(
            output_size,
            kernel_initializer=prepare_kernel_initializers(config.kernel_initializer),
            activation="linear",
            name="A",
        )

        self.advantage_reduced_mean = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=-1, keepdims=True), name="Ao"
        )

        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        x = self.flatten(x)
        if self.has_dense_layers:
            x = self.dense_layers(x)

        if self.has_value_hidden_layers:
            x = self.value_hidden_layers(x)
        value = self.value(x)

        if self.has_advantage_hidden_layers:
            x = self.advantage_hidden_layers(x)
        advantage = self.advantage(x)
        advantage = self.advantage_reduced_mean(advantage)

        q = self.add([value, advantage])
        return q
