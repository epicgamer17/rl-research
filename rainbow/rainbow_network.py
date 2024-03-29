import tensorflow as tf
import numpy as np
from layers.noisy_dense import NoisyDense

# from noisy_conv2d import NoisyConv2D


class Network(tf.keras.Model):
    def __init__(self, config, output_size, input_shape, *args, **kwargs):
        super().__init__()
        self.config = config
        kernel_initializers = []
        for i in range(
            len(config["conv_layers"])
            + config["dense_layers"]
            + config["value_hidden_layers"]
            + config["advantage_hidden_layers"]
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

        activation = None
        if config["activation"] == "linear":
            activation = None
        elif config["activation"] == "relu":
            activation = tf.keras.activations.relu
        elif config["activation"] == "relu6":
            activation = tf.keras.activations.relu(max_value=6)
        elif config["activation"] == "sigmoid":
            activation = tf.keras.activations.sigmoid
        elif config["activation"] == "softplus":
            activation = tf.keras.activations.softplus
        elif config["activation"] == "soft_sign":
            activation = tf.keras.activations.softsign
        elif config["activation"] == "silu":
            activation = tf.nn.silu
        elif config["activation"] == "swish":
            activation = tf.nn.swish
        elif config["activation"] == "log_sigmoid":
            activation = tf.math.log_sigmoid
        elif config["activation"] == "hard_sigmoid":
            activation = tf.keras.activations.hard_sigmoid
        elif config["activation"] == "hard_silu":
            activation = tf.keras.activations.hard_silu
        elif config["activation"] == "hard_swish":
            activation = tf.keras.activations.hard_swish
        elif config["activation"] == "hard_tanh":
            activation = tf.keras.activations.hard_tanh
        elif config["activation"] == "elu":
            activation = tf.keras.activations.elu
        elif config["activation"] == "celu":
            activation = tf.keras.activations.celu
        elif config["activation"] == "selu":
            activation = tf.keras.activations.selu
        elif config["activation"] == "gelu":
            activation = tf.nn.gelu
        elif config["activation"] == "glu":
            activation = tf.keras.activations.glu

        self.inputs = tf.keras.layers.Input(shape=input_shape, name="my_input")
        self.has_conv_layers = len(config["conv_layers"]) > 0
        self.has_dense_layers = config["dense_layers"] > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config["conv_layers"]):
                if config["conv_layers_noisy"]:
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
                                kernel_initializer=kernel_initializers.pop(),
                                activation=activation,
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
                                kernel_initializer=kernel_initializers.pop(),
                                activation=activation,
                                padding="same",
                            )
                        )
            self.conv_layers.append(tf.keras.layers.Flatten())

        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config["dense_layers"]):
                if config["dense_layers_noisy"]:
                    self.dense_layers.append(
                        NoisyDense(
                            config["width"],
                            sigma=config["noisy_sigma"],
                            kernel_initializer=kernel_initializers.pop(),
                            activation=activation,
                        )
                    )
                else:
                    self.dense_layers.append(
                        tf.keras.layers.Dense(
                            config["width"],
                            kernel_initializer=kernel_initializers.pop(),
                            activation=activation,
                        )
                    )

        self.has_value_hidden_layers = config["value_hidden_layers"] > 0
        if self.has_value_hidden_layers:
            self.value_hidden_layers = []
            for i in range(config["value_hidden_layers"]):
                self.value_hidden_layers.append(
                    NoisyDense(
                        config["width"],
                        sigma=config["noisy_sigma"],
                        kernel_initializer=kernel_initializers.pop(),
                        activation=activation,
                    )
                )

        self.value = NoisyDense(
            config["atom_size"],
            sigma=config["noisy_sigma"],
            kernel_initializer=kernel_initializers.pop(),
            activation="linear",
            name="HiddenV",
        )

        self.has_advantage_hidden_layers = config["advantage_hidden_layers"] > 0
        if self.has_advantage_hidden_layers:
            self.advantage_hidden_layers = []
            for i in range(config["advantage_hidden_layers"]):
                self.advantage_hidden_layers.append(
                    NoisyDense(
                        config["width"],
                        sigma=config["noisy_sigma"],
                        kernel_initializer=kernel_initializers.pop(),
                        activation=activation,
                    )
                )

        self.advantage = NoisyDense(
            config["atom_size"] * output_size,
            sigma=config["noisy_sigma"],
            kernel_initializer=kernel_initializers.pop(),
            activation="linear",
            name="A",
        )
        self.advantage_reduced_mean = tf.keras.layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True), name="Ao"
        )

        self.advantage_reshaped = tf.keras.layers.Reshape(
            (output_size, config["atom_size"]), name="ReshapeAo"
        )
        self.value_reshaped = tf.keras.layers.Reshape(
            (1, config["atom_size"]), name="ReshapeV"
        )
        self.add = tf.keras.layers.Add()
        # self.softmax = tf.keras.activations.softmax(self.add, axis=-1)
        # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
        self.clip_qs = tf.keras.layers.Lambda(
            lambda q: tf.clip_by_value(q, 1e-3, 1), name="ClippedQ"
        )
        self.outputs = tf.keras.layers.Lambda(
            lambda q: tf.reduce_sum(q * config["support"], axis=2), name="Q"
        )

    def call(self, inputs, training=False):
        x = inputs
        # logging.debug(f"input shape: {x.shape}")
        if self.has_conv_layers:
            for layer in self.conv_layers:
                x = layer(x)
        if self.has_dense_layers:
            for layer in self.dense_layers:
                x = layer(x)
        # logging.debug(f"after dense layers: {x.shape}")
        if self.has_value_hidden_layers:
            for layer in self.value_hidden_layers:
                x = layer(x)
        # logging.debug(f"after value hidden layers: {x.shape}")
        value = self.value(x)
        # logging.debug(f"after value layer: {value.shape}")
        value = self.value_reshaped(value)
        # logging.debug(f"after value reshaped: {value.shape}")

        if self.has_advantage_hidden_layers:
            for layer in self.advantage_hidden_layers:
                x = layer(x)
        # logging.debug(f"after advantage hidden layers: {x.shape}")
        advantage = self.advantage(x)
        # logging.debug(f"after advantage layer: {advantage.shape}")
        advantage = self.advantage_reduced_mean(advantage)
        # logging.debug(f"after advantage reduced mean: {advantage.shape}")
        advantage = self.advantage_reshaped(advantage)
        # logging.debug(f"after advantage reshaped: {advantage.shape}")

        q = self.add([value, advantage])
        q = tf.keras.activations.softmax(q, axis=-1)
        # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
        # q = self.clip_qs(q)
        # q = self.outputs(q)
        return q

    def reset_noise(self):
        if self.has_dense_layers and self.config["conv_layers_noisy"]:
            for layer in self.conv_layers:
                layer.reset_noise()
        if self.has_dense_layers and self.config["dense_layers_noisy"]:
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
