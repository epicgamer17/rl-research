"""
Stochstic MuZero Notes:
Some environments can be so large or complex compared to the agent that they can appear stochastic to a finite agent.
There are also Stochastic environments and partially observable environments where deterministic models can be insufficient.

Stochastic MuZero uses after states. Essentially, it makes MuZero learn the distribution from these afterstates to the hidden states. After states represent chance events during the environment transition, they don't represent a real observation. They are more like an state between observations. The value predicted by the agent at these states is the value of the hidden state conditioned on the action selected (before randomness has occurred), then after sampling (or quasi sampling) from the possible chance events, we get the next hidden state. (TODO: Is the POMDP to Stochastic valid? Is there a jumping between beliefs that might occur? ie at one node assuming your opponent has one card, and the very next assuming they have a different card)

The major downside of this is that tree search depth for similar performance is essentially doubled. Additionally, the branching factor is significantly increased meaning that in practice its possible that required search compute is more than twice as much.

Additionally, in practice the learning of the chance distributions can be prone to chance code collapse in the VAE leading to essentially a deterministic model that requires twice as much search depth.

---

Alright lets make a plan for implementing Stochastic AlphaZero based on [stochastic_muzero.pdf](file;file:///Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/papers/stochastic_muzero.pdf)

I want to implement it myself. So the plan should be rough. I'll need a run down of existing components and related code.

I will give the feature list. Otherwise I want to make the plan on my own, maybe implement it (even if its partially flawed) and then review it with you. After I implement it and you review it you will probably reimplement it yourself. But the goal is for me to get back into coding and used to that process. If I get stuck I might ask for hints etc. If I get really stuck try and recommend a simple or easy individual part of it i can implement on my own.

Alright here are my implementation notes:

I think we will need to add chance nodes to our tree. I think in practice we might do this via a flag indicating the node type. And maybe overloading other existing stuff like the values and policies. Though it may be less readable and then maybe its better to actually have seperate tensors for the chance codes and distributions and stuff.


"""
