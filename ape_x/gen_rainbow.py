import tensorflow as tf

from agent_configs import RainbowConfig
from game_configs import CartPoleConfig


rainbow_dict = dict(
    activation="relu",
    kernel_initializer="glorot_uniform",
    loss_function=tf.keras.losses.CategoricalCrossentropy(),
)


c = RainbowConfig(rainbow_dict, CartPoleConfig())
c.dump("configs/rainbow_base_example.yaml")
