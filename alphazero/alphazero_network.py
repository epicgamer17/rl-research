import tensorflow as tf
import numpy as np

from layers.residual import Residual


class Network(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape):
        super(Network, self).__init__()
        self.config = config
        self.inputs = tf.keras.layers.Conv2D(
            config["num_filters"],
            kernel_size=config["kernel_size"],
            strides=1,
            padding="same",
            input_shape=input_shape,
            activation="relu",
        )
        self.input_batch_norm = tf.keras.layers.BatchNormalization()
        self.residuals = [
            Residual(config["num_filters"], kernel_size=config["kernel_size"])
            for _ in range(config["num_res_blocks"])
        ]
        self.critic_conv = tf.keras.layers.Conv2D(
            config["critic_hidden_filters"], kernel_size=1, strides=1, padding="same"
        )
        self.critic_batch_norm = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.critic_dense = tf.keras.layers.Dense(
            config["critic_dense_size"], activation="relu"
        )
        self.critic = tf.keras.layers.Dense(1, activation="tanh")

        self.actor_conv = tf.keras.layers.Conv2D(
            config["actor_hidden_filters"], kernel_size=1, strides=1, padding="same"
        )
        self.actor_batch_norm = tf.keras.layers.BatchNormalization()
        self.actor_dense = tf.keras.layers.Dense(
            config["actor_dense_size"], activation="relu"
        )
        self.actor = tf.keras.layers.Dense(output_shape, activation="softmax")

    def call(self, inputs):
        x = self.inputs(inputs)
        print(x.shape)
        x = self.input_batch_norm(x)
        for residual in self.residuals:
            x = residual(x)
        value = self.critic_conv(x)
        value = self.critic_batch_norm(value)
        value = self.flatten(value)
        value = self.critic_dense(value)
        value = self.critic(value)

        policy = self.actor_conv(x)
        policy = self.actor_batch_norm(policy)
        policy = self.flatten(policy)
        policy = self.actor_dense(policy)
        policy = self.actor(policy)

        return value, policy
