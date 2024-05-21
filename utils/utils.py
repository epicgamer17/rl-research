import tensorflow as tf
from tensorflow import keras
from keras.initializers import (
    VarianceScaling,
    Orthogonal,
    GlorotUniform,
    GlorotNormal,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    Initializer,
)

from keras.activations import (
    relu,
    sigmoid,
    softplus,
    softsign,
    hard_sigmoid,
    elu,
    selu,
)

from tensorflow.nn import (
    silu,
    swish,
    gelu,
)

import numpy as np


def normalize_policy(policy):
    policy /= tf.reduce_sum(policy)
    return policy


def action_mask(actions, legal_moves, mask_value=0):
    mask = np.zeros(self.num_actions, dtype=np.int8)
    mask[legal_moves] = 1
    actions[mask == 0] = mask_value
    return actions


def get_legal_moves(self, info):
    # info["legal_moves"] if self.config.game.has_legal_moves else None
    return info["legal_moves"] if "legal_moves" in info else None


def normalize_image(image, single_image=True):
    image_copy = np.array(image)
    image_copy = image_copy / 255.0
    if single_image:
        make_stack(image_copy)
    else:
        normalized_image = image_copy
    return normalized_image


def make_stack(item):
    new_shape = (1,) + item.shape
    return item.reshape(new_shape)


def update_per_beta(per_beta, per_beta_final, per_beta_steps):
    # could also use an initial per_beta instead of current (multiply below equation by current step)
    per_beta = min(
        per_beta_final, per_beta + (per_beta_final - per_beta) / (per_beta_steps)
    )

    return per_beta


def prepare_kernel_initializers(kernel_initializer=None):
    if kernel_initializer == "glorot_uniform":
        return GlorotUniform(seed=np.random.seed())
    elif kernel_initializer == "glorot_normal":
        return GlorotNormal(seed=np.random.seed())
    elif kernel_initializer == "he_uniform":
        return HeUniform(seed=np.random.seed())
    elif kernel_initializer == "he_normal":
        return HeNormal(seed=np.random.seed())
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling(seed=np.random.seed())
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1, seed=np.random.seed())
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3, seed=np.random.seed())
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8, seed=np.random.seed())
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3, seed=np.random.seed())
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5, seed=np.random.seed())
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10, seed=np.random.seed())
    elif kernel_initializer == "lecun_uniform":
        return LecunUniform(seed=np.random.seed())
    elif kernel_initializer == "lecun_normal":
        return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return Orthogonal(seed=np.random.seed())

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def prepare_activations(activation=None):
    # print("Activation to prase: ", activation)
    if activation == "linear":
        return None
    elif activation == "relu":
        return relu
    elif activation == "relu6":
        return relu(max_value=6)
    elif activation == "sigmoid":
        return sigmoid
    elif activation == "softplus":
        return softplus
    elif activation == "soft_sign":
        return softsign
    elif activation == "silu":
        return silu
    elif activation == "swish":
        return swish
    # elif activation == "log_sigmoid":
    #     return log_sigmoid
    elif activation == "hard_sigmoid":
        return hard_sigmoid
    # elif activation == "hard_silu":
    #     return hard_silu
    # elif activation == "hard_swish":
    #     return hard_swish
    # elif activation == "hard_tanh":
    #     return hard_tanh
    elif activation == "elu":
        return elu
    # elif activation == "celu":
    #     return celu
    elif activation == "selu":
        return selu
    elif activation == "gelu":
        return gelu
    # elif activation == "glu":
    #     return glu

    raise ValueError(f"Activation {activation} not recognized")
