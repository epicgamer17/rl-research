"""
AlphaZero was great at games where a perfect simulator was available.
MuZero improves on this by learning a model of the dynamics of the game to be used in search. This goes with the overall philosophy of minimizing prior human knowledge in reinforcement learning algorithms.

It comes with a few downsides. Obviously, more compute is needed for the dynamics and representation, though they attempt to minimize the effect of this by making each network smaller. Additionally, the search (like AlphaZero) is still limited to deterministic games, though it has be expanded to single player gmaes, and non zero sum games. But lacks support for stochastic environments (see stochastic_muzero), among other things.

Unlike other model based methods, MuZero does not attempt to reconstruct the true environment state or a sequence of observations (like Dreamer), instead it only is required to predict rewards, policies, and values. The hidden states that arise only have the requirement that they must be good enough to predict those three things (and they are learned). This is called value equivalence. The idea that starting from the same state, the cummulative rewards will be the same in both environemnts (this allows planning in the learned MDP to transfer to the real MDP).

This is because methods that try to learn the complex visual or raw dynamics of a game often fail in complex domains or visually noisy ones (like Atari). Arguably, Dreamer is a counter example of this, but nonetheless the principle of the idea is useful. Additionally, in visually rich domains, like Atari, model-free methods tended to be the best, as they don't need to reconstruct the observations and instead just predict the value and/or policy.

In general, RL is either model-free or model based. Arguably, today, it has been shown that model based methods are superior to model free methods in most games, but potentially harder to apply in the real world, and they often require more training compute.

The learned model often takes in either an action or option, and then outputs a next state and next reward. Once that model is learned MCTS or value iteration can be used (or it can be done in paralell during training).

Interestingly, they "showed" (hypothesized) that some of the computation of the value estimation was being cached in the search, as MuZero performed better than AlphaZero with the same simulation count,and a smaller value and policy net, but with the addition of the dynamics and representation networks on Go only. Though seemingly this could apply to other domains (likely ones with simple dynamics).

Additionally, they showed that MuZeros policy iteration using search was better than the targets set by Q-Learning. Indicating one of the many benefits of planning and search over model free methods.

TODO: Citaitons 14, 20, 48 and 49 (I think this is like Dreamer)
"""
