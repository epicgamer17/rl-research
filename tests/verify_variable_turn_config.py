from game_configs.variable_turn_tictactoe_config import VariableTurnTicTacToeConfig

def test_config():
    config = VariableTurnTicTacToeConfig()
    env = config.make_env()
    print("Environment created successfully")
    print(f"Observation space: {env.observation_space('player_1')}")
    print(f"Action space: {env.action_space('player_1')}")
    
    env.reset()
    obs, _, _, _, _ = env.last()
    print(f"Initial observation shape: {obs.shape}")

    # ActionMaskInInfoWrapper makes observation a Box, but keeps 'action_mask' in info?
    # Wait, ActionMaskInInfoWrapper: "Wrapper to convert dict observations to processed Box observations."
    # It returns Box.
    # So `obs` should be a Box (numpy array).
    
    # But wait, env.last() returns (observation, reward, terminated, truncated, info)
    
    print("Test passed!")

if __name__ == "__main__":
    test_config()
