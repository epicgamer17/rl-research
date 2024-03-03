import tensorflow as tf
from layers.noisy_dense import NoisyDense
import numpy as np

class Network(tf.keras.Model):
    def __init__(self, config, input_shape, output_shape):
        super(Network, self).__init__()
        self.actor = ActorNetwork(config, input_shape, output_shape)
        self.critic = CriticNetwork(config, input_shape)

    def call(self, inputs):
        return self.actor(inputs), self.critic(inputs)


class CriticNetwork(tf.keras.Model):
    def __init__(self, config, input_shape):
        super(CriticNetwork, self).__init__()
        self.config = config
        kernel_initializers = []
        for i in range(len(config['conv_layers']) + config['critic_dense_layers'] + 1):
            if config['kernel_initializer'] == 'glorot_uniform':
                kernel_initializers.append(tf.keras.initializers.glorot_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'glorot_normal':
                kernel_initializers.append(tf.keras.initializers.glorot_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_normal':
                kernel_initializers.append(tf.keras.initializers.he_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_uniform':
                kernel_initializers.append(tf.keras.initializers.he_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_baseline':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.1':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.1, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.3':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.8':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.8, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_3':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_5':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=5, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_10':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=10, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_uniform':
                kernel_initializers.append(tf.keras.initializers.lecun_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_normal':
                kernel_initializers.append(tf.keras.initializers.lecun_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'orthogonal':
                kernel_initializers.append(tf.keras.initializers.orthogonal(seed=np.random.seed()))

        activation = None
        if config['activation'] == 'linear':
            activation = None
        elif config['activation'] == 'relu':
            activation = tf.keras.activations.relu
        elif config['activation'] == 'relu6':
            activation = tf.keras.activations.relu(max_value=6)
        elif config['activation'] == 'sigmoid':
            activation = tf.keras.activations.sigmoid
        elif config['activation'] == 'softplus':
            activation = tf.keras.activations.softplus
        elif config['activation'] == 'soft_sign':
            activation = tf.keras.activations.softsign
        elif config['activation'] == 'silu':
            activation = tf.nn.silu
        elif config['activation'] == 'swish':
            activation = tf.nn.swish
        elif config['activation'] == 'log_sigmoid':
            activation = tf.math.log_sigmoid
        elif config['activation'] == 'hard_sigmoid':
            activation = tf.keras.activations.hard_sigmoid
        elif config['activation'] == 'hard_silu':
            activation = tf.keras.activations.hard_silu
        elif config['activation'] == 'hard_swish':
            activation = tf.keras.activations.hard_swish
        elif config['activation'] == 'hard_tanh':
            activation = tf.keras.activations.hard_tanh
        elif config['activation'] == 'elu':
            activation = tf.keras.activations.elu
        elif config['activation'] == 'celu':
            activation = tf.keras.activations.celu
        elif config['activation'] == 'selu':
            activation = tf.keras.activations.selu
        elif config['activation'] == 'gelu':
            activation = tf.nn.gelu
        elif config['activation'] == 'glu':
            activation = tf.keras.activations.glu

        self.inputs = tf.keras.layers.Input(shape=input_shape, name='my_input')
        self.has_conv_layers = len(config['conv_layers']) > 0
        self.has_dense_layers = config['critic_dense_layers'] > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config['conv_layers']):
                if config['conv_layers_noisy']:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape, padding='same'))
                    else:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, padding='same'))
            self.conv_layers.append(tf.keras.layers.Flatten())
        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config['critic_dense_layers']):
                if config['critic_dense_layers_noisy']:
                    self.dense_layers.append(NoisyDense(config['critic_width'], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation=activation))
                else:
                    self.dense_layers.append(tf.keras.layers.Dense(config['critic_width'], kernel_initializer=kernel_initializers.pop(), activation=activation))
        self.value = tf.keras.layers.Dense(1, kernel_initializer=kernel_initializers.pop(), activation=None, name='value')

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
        kernel_initializers = []
        for i in range(len(config['conv_layers']) + config['actor_dense_layers'] + 1):
            if config['kernel_initializer'] == 'glorot_uniform':
                kernel_initializers.append(tf.keras.initializers.glorot_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'glorot_normal':
                kernel_initializers.append(tf.keras.initializers.glorot_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_normal':
                kernel_initializers.append(tf.keras.initializers.he_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'he_uniform':
                kernel_initializers.append(tf.keras.initializers.he_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_baseline':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.1':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.1, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.3':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_0.8':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=0.8, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_3':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=3, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_5':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=5, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'variance_10':
                kernel_initializers.append(tf.keras.initializers.VarianceScaling(scale=10, seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_uniform':
                kernel_initializers.append(tf.keras.initializers.lecun_uniform(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'lecun_normal':
                kernel_initializers.append(tf.keras.initializers.lecun_normal(seed=np.random.seed()))
            elif config['kernel_initializer'] == 'orthogonal':
                kernel_initializers.append(tf.keras.initializers.orthogonal(seed=np.random.seed()))

        activation = None
        if config['activation'] == 'linear':
            activation = None
        elif config['activation'] == 'relu':
            activation = tf.keras.activations.relu
        elif config['activation'] == 'relu6':
            activation = tf.keras.activations.relu(max_value=6)
        elif config['activation'] == 'sigmoid':
            activation = tf.keras.activations.sigmoid
        elif config['activation'] == 'softplus':
            activation = tf.keras.activations.softplus
        elif config['activation'] == 'soft_sign':
            activation = tf.keras.activations.softsign
        elif config['activation'] == 'silu':
            activation = tf.nn.silu
        elif config['activation'] == 'swish':
            activation = tf.nn.swish
        elif config['activation'] == 'log_sigmoid':
            activation = tf.math.log_sigmoid
        elif config['activation'] == 'hard_sigmoid':
            activation = tf.keras.activations.hard_sigmoid
        elif config['activation'] == 'hard_silu':
            activation = tf.keras.activations.hard_silu
        elif config['activation'] == 'hard_swish':
            activation = tf.keras.activations.hard_swish
        elif config['activation'] == 'hard_tanh':
            activation = tf.keras.activations.hard_tanh
        elif config['activation'] == 'elu':
            activation = tf.keras.activations.elu
        elif config['activation'] == 'celu':
            activation = tf.keras.activations.celu
        elif config['activation'] == 'selu':
            activation = tf.keras.activations.selu
        elif config['activation'] == 'gelu':
            activation = tf.nn.gelu
        elif config['activation'] == 'glu':
            activation = tf.keras.activations.glu

        self.inputs = tf.keras.layers.Input(shape=input_shape, name='my_input')
        self.has_conv_layers = len(config['conv_layers']) > 0
        self.has_dense_layers = config['actor_dense_layers'] > 0
        if self.has_conv_layers:
            self.conv_layers = []
            for i, (filters, kernel_size, strides) in enumerate(config['conv_layers']):
                if config['conv_layers_noisy']:
                    # if i == 0:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape))
                    # else:
                    #     self.conv_layers.append(NoisyConv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation))
                    pass
                else:
                    if i == 0:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, input_shape=input_shape, padding='same'))
                    else:
                        self.conv_layers.append(tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializers.pop(), activation=activation, padding='same'))
            self.conv_layers.append(tf.keras.layers.Flatten())
        if self.has_dense_layers:
            self.dense_layers = []
            for i in range(config['actor_dense_layers']):
                if config['actor_dense_layers_noisy']:
                    self.dense_layers.append(NoisyDense(config['actor_width'], sigma=config['noisy_sigma'], kernel_initializer=kernel_initializers.pop(), activation=activation))
                else:
                    self.dense_layers.append(tf.keras.layers.Dense(config['actor_width'], kernel_initializer=kernel_initializers.pop(), activation=activation))
        if self.discrete:
            self.actions = tf.keras.layers.Dense(output_shape, kernel_initializer=kernel_initializers.pop(), activation='softmax', name='actions')
        else:
            self.mean = tf.keras.layers.Dense(output_shape, kernel_initializer=kernel_initializers.pop(), activation='tanh', name='mean')
            self.std = tf.keras.layers.Dense(output_shape, kernel_initializer=kernel_initializers.pop(), activation='softplus', name='std')

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
