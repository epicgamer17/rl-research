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
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        downsample=None,
        kernel_initializer="he_uniform",
        regularizer=None,
        **kwargs
    ):
        super(Residual, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizer,
            data_format="channels_first",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(
            beta_regularizer=regularizer,
            gamma_regularizer=regularizer,
            axis=1,  # AXIS SHOULD BE CHANNEL AXIS I THINK SO CHANGE THIS WHEN BOARD HISTORY IS USED
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=regularizer,
            data_format="channels_first",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(
            beta_regularizer=regularizer, gamma_regularizer=regularizer, axis=1
        )

        self.relu = tf.keras.layers.Activation("relu")
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
