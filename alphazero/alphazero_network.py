import tensorflow as tf
import numpy as np

from layers.residual import Residual


class Network(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape):
        super(Network, self).__init__()
        self.config = config

        regularizer = tf.keras.regularizers.L2(config["weight_decay"])

        self.inputs = tf.keras.layers.Conv2D(
            config["num_filters"],
            kernel_size=config["kernel_size"],
            strides=1,
            padding="same",
            input_shape=input_shape,
            activation="relu",
            kernel_regularizer=regularizer,
        )
        self.input_batch_norm = tf.keras.layers.BatchNormalization(
            beta_regularizer=regularizer, gamma_regularizer=regularizer
        )
        self.residuals = [
            Residual(
                config["num_filters"],
                kernel_size=config["kernel_size"],
                regularizer=regularizer,
            )
            for _ in range(config["num_res_blocks"])
        ]
        self.critic_conv_layers = []
        for critic_conv_layer in range(config["critic_conv_layers"]):
            self.critic_conv_layers.append(
                tf.keras.layers.Conv2D(
                    config["critic_conv_filters"],
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=regularizer,
                )
            )
            self.critic_conv_layers.append(
                tf.keras.layers.BatchNormalization(
                    beta_regularizer=regularizer, gamma_regularizer=regularizer
                )
            )
        self.critic_dense_layers = []
        for critic_dense_layer in range(config["critic_dense_layers"]):
            self.critic_dense_layers.append(
                tf.keras.layers.Dense(
                    config["critic_dense_size"],
                    activation="relu",
                    kernel_regularizer=regularizer,
                )
            )
        self.critic = tf.keras.layers.Dense(
            1, activation="tanh", name="critic", kernel_regularizer=regularizer
        )

        self.actor_conv_layers = []
        for actor_conv_layer in range(config["actor_conv_layers"]):
            self.actor_conv_layers.append(
                tf.keras.layers.Conv2D(
                    config["actor_conv_filters"],
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=regularizer,
                )
            )
            self.actor_conv_layers.append(
                tf.keras.layers.BatchNormalization(
                    beta_regularizer=regularizer, gamma_regularizer=regularizer
                )
            )
        self.actor_dense_layers = []
        for actor_dense_layer in range(config["actor_dense_layers"]):
            self.actor_dense_layers.append(
                tf.keras.layers.Dense(
                    config["actor_dense_size"],
                    activation="relu",
                    kernel_regularizer=regularizer,
                )
            )
        self.actor = tf.keras.layers.Dense(
            output_shape,
            activation="softmax",
            name="actor",
            kernel_regularizer=regularizer,
        )
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.inputs(inputs)
        x = self.input_batch_norm(x)
        for residual in self.residuals:
            x = residual(x)
        critic_x = x
        for layer in self.critic_conv_layers:
            critic_x = layer(critic_x)
        critic_x = self.flatten(critic_x)
        for layer in self.critic_dense_layers:
            critic_x = layer(critic_x)
        value = self.critic(critic_x)
        actor_x = x
        for layer in self.actor_conv_layers:
            actor_x = layer(actor_x)
        actor_x = self.flatten(actor_x)
        for layer in self.actor_dense_layers:
            actor_x = layer(actor_x)
        policy = self.actor(actor_x)

        return value, policy
