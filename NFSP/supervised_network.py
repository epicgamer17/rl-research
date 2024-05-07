import tensorflow as tf
from tensorflow import keras
from keras import Model
import numpy as np
from layers.noisy_dense import NoisyDense


class SupervisedNetwork(Model):
    def __init__(self, config, output_size, input_shape, *args, **kwargs):
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
                    # self.dense_layers.append(
                    #     NoisyDense(
                    #         config.width,
                    #         sigma=config.noisy_sigma,
                    #         kernel_initializer=self.config.prepare_kernel_initializers(),
                    #         activation=config.activation,
                    #     )
                    # )
                    pass
                else:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            config.width,
                            kernel_initializer=self.config.prepare_kernel_initializers(),
                            activation=config.activation,
                        )
                    )
        self.output_layer = tf.keras.layers.Dense(output_size, activation="softmax")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        x = self.flatten(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        return self.output_layer(x)

    def reset_noise(self):
        # if we use noisy layers for this network
        for layer in self.dense_layers:
            layer.reset_noise()
        for layer in self.conv_layers:
            layer.reset_noise()
