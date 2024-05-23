import tensorflow as tf
from utils import prepare_kernel_initializers


class ConvStack(tf.keras.Model):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation,
        kernel_initializer,
        kernel_regularizer=None,
        data_format: str = "channels_last",
        noisy_sigma: float = None,
    ):
        super(ConvStack, self).__init__()
        filters = filters
        kernel_sizes = kernel_sizes
        strides = strides
        num_layers = len(filters)
        self.conv_layers = []

        if noisy_sigma != 0:
            assert noisy_sigma is not None, "Noisy Conv requires sigma"
            raise NotImplementedError("Noisy convolutions not implemented yet")
        else:
            # First Layer is an Input Layer
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters[0],
                    kernel_sizes[0],
                    strides=strides[0],
                    kernel_initializer=prepare_kernel_initializers(kernel_initializer),
                    kernel_regularizer=kernel_regularizer,
                    activation=activation,
                    input_shape=input_shape,
                    padding="same",
                    data_format=data_format,
                )
            )

            for i in range(1, num_layers):
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        filters[i],
                        kernel_sizes[i],
                        strides=strides[i],
                        kernel_initializer=prepare_kernel_initializers(
                            kernel_initializer
                        ),
                        kernel_regularizer=kernel_regularizer,
                        activation=activation,
                        padding="same",
                        data_format=data_format,
                    )
                )
        # self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        # x = self.flatten(x)
        return x

    def reset_noise(self):
        for layer in self.conv_layers:
            if isinstance(layer, NoisyConv):
                layer.reset_noise()
        return
