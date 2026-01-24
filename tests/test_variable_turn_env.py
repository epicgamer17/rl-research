import numpy as np
import pytest
from custom_gym_envs.envs.variable_turn_tictactoe import env as VariableTurnTicTacToeEnv

def test_variable_turn_logic():
    # Initialize environment
    env = VariableTurnTicTacToeEnv(render_mode=None, min_moves=2, max_moves=3, win_length=4)
    env.reset()
    
    # Check initial player
    first_player = env.agent_selection
    assert first_player in ["player_0", "player_1"]
    
    # We want to track how many moves the first player gets
    moves = 0
    initial_moves_left = env.unwrapped.moves_left
    print(f"Initial moves allocated: {initial_moves_left}")
    
    assert 2 <= initial_moves_left <= 3
    
    current_player = first_player
    
    # Perform moves until turn changes
    for _ in range(initial_moves_left):
        assert env.agent_selection == current_player
        
        # Find a legal move
        observation, reward, termination, truncation, info = env.last()
        assert not termination 
        assert not truncation
        
        mask = observation["action_mask"]
        legal_moves = np.flatnonzero(mask)
        action = legal_moves[0]
        
        env.step(action)
        moves += 1
        
    # After 'initial_moves_left' steps, the turn should switch (unless game ended)
    # Note: verify game didn't end (it shouldn't in 2-3 moves on empty board)
    
    assert env.agent_selection != current_player
    print("Turn switched successfully after", moves, "moves.")

def test_pettingzoo_api():
    from pettingzoo.test import api_test
    env = VariableTurnTicTacToeEnv()
    api_test(env, num_cycles=1000, verbose_progress=False)

if __name__ == "__main__":
    test_variable_turn_logic()
    test_pettingzoo_api()
    print("All tests passed!")
