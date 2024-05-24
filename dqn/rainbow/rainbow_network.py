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
        self, config: RainbowConfig, output_size: int, input_shape, *args, **kwargs
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
                self.config.conv_layers_noisy,
            )

        widths = [config.width] * config.dense_layers
        if self.has_dense_layers:
            self.dense_layers = DenseStack(
                widths,
                self.config.activation,
                self.config.kernel_initializer,
                noisy_sigma=self.config.noisy_sigma,
            )

        self.has_value_hidden_layers = config.value_hidden_layers > 0
        if self.has_value_hidden_layers:
            widths = [config.width] * config.value_hidden_layers
            self.value_hidden_layers = DenseStack(
                widths,
                self.config.activation,
                self.config.kernel_initializer,
                noisy_sigma=self.config.noisy_sigma,
            )

        if self.config.noisy_sigma != 0:
            self.value = NoisyDense(
                config.atom_size,
                sigma=config.noisy_sigma,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer, output_layer=True
                ),
                activation="linear",
                name="HiddenV",
            )
        else:
            self.value = tf.keras.layers.Dense(
                config.atom_size,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer, output_layer=True
                ),
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
                noisy_sigma=self.config.noisy_sigma,
            )

        if self.config.noisy_sigma != 0:
            self.advantage = NoisyDense(
                config.atom_size * output_size,
                sigma=config.noisy_sigma,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer, output_layer=True
                ),
                activation="linear",
                name="A",
            )
        else:
            self.advantage = tf.keras.layers.Dense(
                config.atom_size * output_size,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer, output_layer=True
                ),
                activation="linear",
                name="A",
            )

        self.advantage_reshaped = tf.keras.layers.Reshape(
            (output_size, config.atom_size), name="ReshapeAo"
        )
        self.value_reshaped = tf.keras.layers.Reshape(
            (1, config.atom_size), name="ReshapeV"
        )
        self.advantage_reduced_mean = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True), name="Ao"
        )
        self.add = tf.keras.layers.Add()
        # self.softmax = tf.keras.activations.softmax(self.add, axis=-1)
        # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
        self.clip_qs = tf.keras.layers.Lambda(
            lambda q: tf.clip_by_value(q, 1e-3, 1), name="ClippedQ"
        )
        self.outputs = tf.keras.layers.Lambda(
            lambda q: tf.reduce_sum(q * config.support, axis=2), name="Q"
        )

        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=False):
        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        x = self.flatten(x)
        if self.has_dense_layers:
            x = self.dense_layers(x)

        if self.has_value_hidden_layers:
            value = self.value_hidden_layers(x)
        else:
            value = x
        value = self.value(value)
        value = self.value_reshaped(value)

        if self.has_advantage_hidden_layers:
            advantage = self.advantage_hidden_layers(x)
        else:
            advantage = x
        advantage = self.advantage(advantage)
        advantage = self.advantage_reshaped(advantage)
        advantage = self.advantage_reduced_mean(advantage)

        q = self.add([value, advantage])
        q = tf.keras.activations.softmax(q, axis=-1)
        # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
        # q = self.clip_qs(q)
        # q = self.outputs(q)
        # print(q.shape)
        return q

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_conv_layers:
                self.conv_layers.reset_noise()
            if self.has_dense_layers:
                self.dense_layers.reset_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.reset_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.reset_noise()
            self.value.reset_noise()
            self.advantage.reset_noise()
