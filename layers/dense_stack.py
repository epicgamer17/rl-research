import tensorflow as tf
from utils.utils import prepare_kernel_initializers

from layers.noisy_dense import NoisyDense


class DenseStack(tf.keras.Model):
    def __init__(
        self,
        widths: list[int],
        activation,
        kernel_initializer,
        noisy: bool = False,
        noisy_sigma=None,
    ):
        super(DenseStack, self).__init__()
        self.noisy = noisy
        assert len(widths) > 0
        assert self.noisy and noisy_sigma is not None, "Noisy Dense requires sigma"
        self.dense_layers = []
        if self.noisy:
            for i in range(len(widths)):
                self.dense_layers.append(
                    NoisyDense(
                        widths[i],
                        sigma=noisy_sigma,
                        kernel_initializer=prepare_kernel_initializers(
                            kernel_initializer
                        ),
                        activation=activation,
                    )
                )
        else:
            for i in range(len(widths)):
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        widths[i],
                        kernel_initializer=prepare_kernel_initializers(
                            kernel_initializer
                        ),
                        activation=activation,
                    )
                )

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x
