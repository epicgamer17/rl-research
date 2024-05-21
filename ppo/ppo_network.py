import tensorflow as tf
from layers.noisy_dense import NoisyDense
import numpy as np
from utils import prepare_kernel_initializers


class Network(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape, discrete):
        super(Network, self).__init__()
        self.actor = ActorNetwork(config, input_shape, output_shape, discrete)
        self.critic = CriticNetwork(config, input_shape)

    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)


class CriticNetwork(tf.keras.Model):
    def __init__(self, config, input_shape):
        super(CriticNetwork, self).__init__()

        self.inputs = tf.keras.layers.Input(shape=input_shape, name="my_input")
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = config.critic_dense_layers > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config.conv_layers):
                if config.conv_layers_noisy:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(
                            tf.keras.layers.Conv2D(
                                filters,
                                kernel_size,
                                strides=strides,
                                kernel_initializer=prepare_kernel_initializers(
                                    config.kernel_initializer
                                ),
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
                                kernel_initializer=prepare_kernel_initializers(
                                    config.kernel_initializer
                                ),
                                activation=config.activation,
                                padding="same",
                            )
                        )
            self.conv_layers.append(tf.keras.layers.Flatten())
        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config.critic_dense_layers):
                if config.critic_dense_layers_noisy:
                    self.dense_layers.append(
                        NoisyDense(
                            config.critic_width,
                            sigma=config.noisy_sigma,
                            kernel_initializer=prepare_kernel_initializers(
                                config.kernel_initializer
                            ),
                            activation=config.activation,
                        )
                    )
                else:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            config.critic_width,
                            kernel_initializer=prepare_kernel_initializers(
                                config.kernel_initializer
                            ),
                            activation=config.activation,
                        )
                    )
        self.value = tf.keras.layers.Dense(
            1,
            kernel_initializer=prepare_kernel_initializers(config.kernel_initializer),
            activation=None,
            name="value",
        )

    def call(self, inputs):
        x = inputs
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        value = self.value(x)
        return value


class ActorNetwork(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape, discrete=True):
        super(ActorNetwork, self).__init__()
        self.config = config
        self.discrete = discrete

        self.inputs = tf.keras.layers.Input(shape=input_shape, name="my_input")
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = config.actor_dense_layers > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config.conv_layers):
                if config.conv_layers_noisy:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(
                            tf.keras.layers.Conv2D(
                                filters,
                                kernel_size,
                                strides=strides,
                                kernel_initializer=prepare_kernel_initializers(
                                    config.kernel_initializer
                                ),
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
                                kernel_initializer=prepare_kernel_initializers(
                                    config.kernel_initializer
                                ),
                                activation=config.activation,
                                padding="same",
                            )
                        )
            self.conv_layers.append(tf.keras.layers.Flatten())
        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config.actor_dense_layers):
                if config.actor_dense_layers_noisy:
                    self.dense_layers.append(
                        NoisyDense(
                            config.actor_width,
                            sigma=config.noisy_sigma,
                            kernel_initializer=prepare_kernel_initializers(
                                config.kernel_initializer
                            ),
                            activation=config.activation,
                        )
                    )
                else:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            config.actor_width,
                            kernel_initializer=prepare_kernel_initializers(
                                config.kernel_initializer
                            ),
                            activation=config.activation,
                        )
                    )
        if self.discrete:
            self.actions = tf.keras.layers.Dense(
                output_shape,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer
                ),
                activation="softmax",
                name="actions",
            )
        else:
            self.mean = tf.keras.layers.Dense(
                output_shape,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer
                ),
                activation="tanh",
                name="mean",
            )
            self.std = tf.keras.layers.Dense(
                output_shape,
                kernel_initializer=prepare_kernel_initializers(
                    config.kernel_initializer
                ),
                activation="softplus",
                name="std",
            )

    def call(self, inputs):
        x = inputs
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        if self.discrete:
            actions = self.actions(x)
            return actions
        else:
            mean = self.mean(x)
            std = self.std(x)
            return mean, std
