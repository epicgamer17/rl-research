import tensorflow as tf
from tensorflow import keras
from keras import Model
import numpy as np
from layers.noisy_dense import NoisyDense
from agent_configs import RainbowConfig

# from noisy_conv2d import NoisyConv2D


class Network(Model):
    def __init__(
        self, config: RainbowConfig, output_size, input_shape, *args, **kwargs
    ):
        super().__init__()
        self.config = config

        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = config.dense_layers > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config.conv_layers):
                if config.conv_layers_noisy:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=self.config.prepare_kernel_initializers(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=self.config.prepare_kernel_initializers(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(
                            tf.keras.layers.Conv2D(
                                filters,
                                kernel_size,
                                strides=strides,
                                kernel_initializer=self.config.prepare_kernel_initializers(),
                                activation=config.activation,
                                input_shape=input_shape,
                                padding="same",
                            )
                        )
                    else:
                        self.conv_layers.append(
                            tf.keras.layers.Conv2D(
                                filters,
                                kernel_size,
                                strides=strides,
                                kernel_initializer=self.config.prepare_kernel_initializers(),
                                activation=config.activation,
                                padding="same",
                            )
                        )
            # self.conv_layers.append(tf.keras.layers.Flatten())

        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config.dense_layers):
                if config.dense_layers_noisy:
                    self.dense_layers.append(
                        NoisyDense(
                            config.width,
                            sigma=config.noisy_sigma,
                            kernel_initializer=self.config.prepare_kernel_initializers(),
                            activation=config.activation,
                        )
                    )
                else:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            config.width,
                            kernel_initializer=self.config.prepare_kernel_initializers(),
                            activation=config.activation,
                        )
                    )

        self.has_value_hidden_layers = config.value_hidden_layers > 0
        if self.has_value_hidden_layers:
            self.value_hidden_layers = []
            for i in range(config.value_hidden_layers):
                self.value_hidden_layers.append(
                    NoisyDense(
                        config.width,
                        sigma=config.noisy_sigma,
                        kernel_initializer=self.config.prepare_kernel_initializers(),
                        activation=config.activation,
                    )
                )

        self.value = NoisyDense(
            config.atom_size,
            sigma=config.noisy_sigma,
            kernel_initializer=self.config.prepare_kernel_initializers(),
            activation="linear",
            name="HiddenV",
        )

        self.has_advantage_hidden_layers = config.advantage_hidden_layers > 0
        if self.has_advantage_hidden_layers:
            self.advantage_hidden_layers = []
            for i in range(config.advantage_hidden_layers):
                self.advantage_hidden_layers.append(
                    NoisyDense(
                        config.width,
                        sigma=config.noisy_sigma,
                        kernel_initializer=self.config.prepare_kernel_initializers(),
                        activation=config.activation,
                    )
                )

        self.advantage = NoisyDense(
            config.atom_size * output_size,
            sigma=config.noisy_sigma,
            kernel_initializer=self.config.prepare_kernel_initializers(),
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
        # print("Input Shape ", x.shape)
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        x = self.flatten(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        if self.has_value_hidden_layers:
            for layer in self.value_hidden_layers:
                x = layer(x)
        # print("Last Dense Layer Shape ", x.shape)
        value = self.value(x)
        # print("Value Shape ", value.shape)
        value = self.value_reshaped(value)
        # print("Reshaped Value Shape ", value.shape)

        if self.has_advantage_hidden_layers:
            for layer in self.advantage_hidden_layers:
                x = layer(x)
        advantage = self.advantage(x)
        advantage = self.advantage_reshaped(advantage)
        advantage = self.advantage_reduced_mean(advantage)
        # print("Reduced Mean Advantage Shape ", advantage.shape)

        q = self.add([value, advantage])
        q = tf.keras.activations.softmax(q, axis=-1)
        # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
        # q = self.clip_qs(q)
        # q = self.outputs(q)
        # print(q.shape)
        return q

    def reset_noise(self):
        if self.has_dense_layers and self.config.conv_layers_noisy:
            for layer in self.conv_layers:
                layer.reset_noise()
        if self.has_dense_layers and self.config.dense_layers_noisy:
            for layer in self.dense_layers:
                layer.reset_noise()
        if self.has_value_hidden_layers:
            for layer in self.value_hidden_layers:
                layer.reset_noise()
        if self.has_advantage_hidden_layers:
            for layer in self.advantage_hidden_layers:
                layer.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()
