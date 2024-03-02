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

def _scaled_noise(size, dtype):
    x = tf.random.normal(shape=size, dtype=dtype)
    return tf.sign(x) * tf.sqrt(tf.abs(x))

class NoisyDense(tf.keras.layers.Dense):
    def __init__(
        self,
        units: int,
        sigma: float = 0.5, # might want to make sigma 0.1 for CPU's
        use_factorised: bool = True,
        activation = None,
        use_bias: bool = True,
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        delattr(self, "kernel_initializer")
        delattr(self, "bias_initializer")
        self.sigma = sigma
        self.use_factorised = use_factorised

    def build(self, input_shape):
        # Make sure dtype is correct
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )

        input_shape = tf.TensorShape(input_shape)
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        sqrt_dim = self.last_dim ** (1 / 2)
        if self.last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.last_dim})

        # use factorising Gaussian variables
        if self.use_factorised:
            mu_init = 1.0 / sqrt_dim
            sigma_init = self.sigma / sqrt_dim
        # use independent Gaussian variables
        else:
            mu_init = (3.0 / self.last_dim) ** (1 / 2)
            sigma_init = 0.017

        sigma_init = initializers.Constant(value=sigma_init)
        mu_init = initializers.RandomUniform(minval=-mu_init, maxval=mu_init)

        # Learnable parameters
        self.sigma_kernel = self.add_weight(
            "sigma_kernel",
            shape=[self.last_dim, self.units],
            initializer=sigma_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.mu_kernel = self.add_weight(
            "mu_kernel",
            shape=[self.last_dim, self.units],
            initializer=mu_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.eps_kernel = self.add_weight(
            "eps_kernel",
            shape=[self.last_dim, self.units],
            initializer=initializers.Zeros(),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False,
        )

        if self.use_bias:
            self.sigma_bias = self.add_weight(
                "sigma_bias",
                shape=[
                    self.units,
                ],
                initializer=sigma_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.mu_bias = self.add_weight(
                "mu_bias",
                shape=[
                    self.units,
                ],
                initializer=mu_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.eps_bias = self.add_weight(
                "eps_bias",
                shape=[
                    self.units,
                ],
                initializer=initializers.Zeros(),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False,
            )
        else:
            self.sigma_bias = None
            self.mu_bias = None
            self.eps_bias = None
        self.reset_noise()
        self.built = True

    @property
    def kernel(self):
        return self.mu_kernel + (self.sigma_kernel * self.eps_kernel)

    @property
    def bias(self):
        if self.use_bias:
            return self.mu_bias + (self.sigma_bias * self.eps_bias)

    def reset_noise(self):
        """Create the factorised Gaussian noise."""

        if self.use_factorised:
            # Generate random noise
            in_eps = _scaled_noise([self.last_dim, 1], dtype=self.dtype)
            out_eps = _scaled_noise([1, self.units], dtype=self.dtype)

            # Scale the random noise
            self.eps_kernel.assign(tf.matmul(in_eps, out_eps))
            self.eps_bias.assign(out_eps[0])
        else:
            # generate independent variables
            self.eps_kernel.assign(
                tf.random.normal(shape=[self.last_dim, self.units], dtype=self.dtype)
            )
            self.eps_bias.assign(
                tf.random.normal(
                    shape=[
                        self.units,
                    ],
                    dtype=self.dtype,
                )
            )

    def remove_noise(self):
        """Remove the factorised Gaussian noise."""

        self.eps_kernel.assign(tf.zeros([self.last_dim, self.units], dtype=self.dtype))
        self.eps_bias.assign(tf.zeros([self.units], dtype=self.dtype))

    def call(self, inputs):
        # TODO(WindQAQ): Replace this with `dense()` once public.
        return super().call(inputs)

    def get_config(self):
        # TODO(WindQAQ): Get rid of this hacky way.
        config = super(tf.keras.layers.Dense, self).get_config()
        config.update(
            {
                "units": self.units,
                "sigma": self.sigma,
                "use_factorised": self.use_factorised,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config