# From tensorflow_addons
import tensorflow as tf
from tensorflow.keras import (
    activations,
    initializers,
    regularizers,
    constraints,
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec


class Residual(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride=1, downsample=None, **kwargs):
        super(Residual, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            use_bias=False,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample

    def call(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(inputs)
        x += residual
        x = self.relu(x)
        return x
