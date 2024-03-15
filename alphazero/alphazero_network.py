import tensorflow as tf
import numpy as np

from layers.residual import Residual


class Network(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape):
        super(Network, self).__init__()
        self.config = config
        kernel_initializers = []
        for i in range(
            1
            + config["num_res_blocks"]
            + config["critic_conv_layers"]
            + config["critic_dense_layers"]
            + config["actor_conv_layers"]
            + config["actor_dense_layers"]
            + 2
        ):
            if config["kernel_initializer"] == "glorot_uniform":
                kernel_initializers.append(
                    tf.keras.initializers.glorot_uniform(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "glorot_normal":
                kernel_initializers.append(
                    tf.keras.initializers.glorot_normal(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "he_normal":
                kernel_initializers.append(
                    tf.keras.initializers.he_normal(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "he_uniform":
                kernel_initializers.append(
                    tf.keras.initializers.he_uniform(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "variance_baseline":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "variance_0.1":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=0.1, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "variance_0.3":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=0.3, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "variance_0.8":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=0.8, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "variance_3":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=3, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "variance_5":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=5, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "variance_10":
                kernel_initializers.append(
                    tf.keras.initializers.VarianceScaling(
                        scale=10, seed=np.random.seed()
                    )
                )
            elif config["kernel_initializer"] == "lecun_uniform":
                kernel_initializers.append(
                    tf.keras.initializers.lecun_uniform(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "lecun_normal":
                kernel_initializers.append(
                    tf.keras.initializers.lecun_normal(seed=np.random.seed())
                )
            elif config["kernel_initializer"] == "orthogonal":
                kernel_initializers.append(
                    tf.keras.initializers.orthogonal(seed=np.random.seed())
                )

        self.inputs = tf.keras.layers.Conv2D(
            config["num_filters"],
            kernel_size=config["kernel_size"],
            strides=1,
            padding="same",
            input_shape=input_shape,
            kernel_initializer=kernel_initializers.pop(),
            kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
            data_format="channels_first",
        )
        self.input_batch_norm = tf.keras.layers.BatchNormalization(
            beta_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
            gamma_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
            axis=1,  # AXIS SHOULD BE CHANNEL AXIS I THINK SO CHANGE THIS WHEN BOARD HISTORY IS USED
        )
        self.residuals = [
            Residual(
                config["num_filters"],
                kernel_size=config["kernel_size"],
                kernel_initializer=kernel_initializers.pop(),
                regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
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
                    kernel_initializer=kernel_initializers.pop(),
                    kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    data_format="channels_first",
                )
            )
            self.critic_conv_layers.append(
                tf.keras.layers.BatchNormalization(
                    beta_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    gamma_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    axis=1,  # AXIS SHOULD BE CHANNEL AXIS I THINK SO CHANGE THIS WHEN BOARD HISTORY IS USED
                )
            )
        self.critic_dense_layers = []
        for critic_dense_layer in range(config["critic_dense_layers"]):
            self.critic_dense_layers.append(
                tf.keras.layers.Dense(
                    config["critic_dense_size"],
                    activation="relu",
                    kernel_initializer=kernel_initializers.pop(),
                    kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                )
            )
        self.critic = tf.keras.layers.Dense(
            1,
            activation="tanh",
            kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
            name="critic",
        )

        self.actor_conv_layers = []
        for actor_conv_layer in range(config["actor_conv_layers"]):
            self.actor_conv_layers.append(
                tf.keras.layers.Conv2D(
                    config["actor_conv_filters"],
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_initializer=kernel_initializers.pop(),
                    kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    data_format="channels_first",
                )
            )
            self.actor_conv_layers.append(
                tf.keras.layers.BatchNormalization(
                    beta_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    gamma_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                    axis=1,  # AXIS SHOULD BE CHANNEL AXIS I THINK SO CHANGE THIS WHEN BOARD HISTORY IS USED
                )
            )

        self.actor_dense_layers = []
        for actor_dense_layer in range(config["actor_dense_layers"]):
            self.actor_dense_layers.append(
                tf.keras.layers.Dense(
                    config["actor_dense_size"],
                    activation="relu",
                    kernel_initializer=kernel_initializers.pop(),
                    kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
                )
            )
        self.actor = tf.keras.layers.Dense(
            output_shape,
            name="actor",
            kernel_initializer=kernel_initializers.pop(),
            kernel_regularizer=tf.keras.regularizers.L2(config["weight_decay"]),
        )

        self.flatten = tf.keras.layers.Flatten()
        self.relu = tf.keras.layers.Activation("relu")
        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = self.inputs(inputs)
        x = self.input_batch_norm(x)
        x = self.relu(x)
        for residual in self.residuals:
            x = residual(x)
        critic_x = x
        for layer in self.critic_conv_layers:
            critic_x = layer(critic_x)
            critic_x = self.relu(critic_x)
        critic_x = self.flatten(critic_x)
        for layer in self.critic_dense_layers:
            critic_x = layer(critic_x)
        value = self.critic(critic_x)
        actor_x = x
        for layer in self.actor_conv_layers:
            actor_x = layer(actor_x)
            actor_x = self.relu(actor_x)
        actor_x = self.flatten(actor_x)
        for layer in self.actor_dense_layers:
            actor_x = layer(actor_x)
        policy = self.actor(actor_x)
        # print(policy)
        policy = self.softmax(policy)

        return value, policy
