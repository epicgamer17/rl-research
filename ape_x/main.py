import gym
import numpy as np
import actor
import learner


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


def make_pacman_env():
    # as recommended by the original paper, should already include max pooling
    pacman_gym = gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array")
    preprocessed = gym.wrappers.AtariPreprocessing(
        pacman_gym, terminal_on_life_loss=True
    )
    env = ClipReward(preprocessed, -1, 1)
    env.reset()
    return env


def main():

    actor_config = actor.default_actor_config
    rainbow_config = actor.default_rainbow_config

    l = learner.SingleMachineLearner(
        env=make_pacman_env(),
        config=rainbow_config,
        learner_config=learner.default_learner_config,
    )

    a = actor.SingleMachineActor(
        0,
        env=make_pacman_env(),
        config=rainbow_config,
        actor_config=actor_config,
        single_machine_learner=l,
    )

    a.run()


if __name__ == "__main__":
    main()
