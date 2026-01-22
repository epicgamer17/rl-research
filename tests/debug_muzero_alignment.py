import torch
import numpy as np
from replay_buffers.game import Game
from replay_buffers.processors import MuZeroGameInputProcessor, MuZeroUnrollOutputProcessor

def debug_alignment():
    num_actions = 1
    num_players = 1
    
    # 1. Simulate play_game exactly
    game = Game(num_players)
    game.append(observation=np.array([0]), info={"player": 0, "legal_moves": [0], "done": False})
    game.append(observation=np.array([1]), info={"player": 0, "legal_moves": [0], "done": False}, action=0, reward=10.0, policy=torch.tensor([1.0]), value=0.0)
    game.append(observation=np.array([2]), info={"player": 0, "legal_moves": [0], "done": False}, action=0, reward=50.0, policy=torch.tensor([1.0]), value=0.0)
    game.append(observation=np.array([3]), info={"player": 0, "legal_moves": [], "done": True}, action=0, reward=0.0, policy=torch.tensor([1.0]), value=0.0)
    
    print(f"Game rewards in storage: {game.rewards}")
    
    # 2. Input Processor
    input_proc = MuZeroGameInputProcessor(num_actions, num_players)
    storage_data = input_proc.process_game(game)
    
    # Ensure buffer is large enough for the lookahead window
    # Storage rewards has length 4: [10, 50, 0, 0]
    # We will sample index 0
    full_buffers = {k: v for k, v in storage_data.items()}
    full_buffers["ids"] = torch.zeros(len(game.observation_history))
    full_buffers["training_steps"] = torch.zeros(len(game.observation_history))
    full_buffers["game_ids"] = torch.ones(len(game.observation_history)).long()
    
    print(f"Buffer Rewards (rews_t): {full_buffers['rewards'].tolist()}")
    
    # 3. Output Processor
    output_proc = MuZeroUnrollOutputProcessor(
        unroll_steps=2,
        n_step=1,
        gamma=1.0,
        num_actions=num_actions,
        num_players=num_players,
        max_size=10
    )
    
    # Indices for sampling at S0
    batch = output_proc.process_batch([0], full_buffers)
    
    print(f"\nFinal Targets for k=0, 1, 2 (sampled at S0):")
    print(f"Target Observations: {batch['unroll_observations'][0].flatten().tolist()}")
    print(f"Target Rewards:      {batch['rewards'][0].tolist()}")
    print(f"Target Values:       {batch['values'][0].tolist()}")
    
    # Analysis:
    # Buffer is [10, 50, 0, 0]
    # k=0 target reward: 0.0
    # k=1 target reward: raw_rewards[0] = 10.0
    # k=2 target reward: raw_rewards[1] = 50.0
    
    assert batch['rewards'][0, 1] == 10.0
    assert batch['rewards'][0, 2] == 50.0
    print("\nVerified: Target Reward for k=1 matches R1 and k=2 matches R2.")

if __name__ == "__main__":
    debug_alignment()
