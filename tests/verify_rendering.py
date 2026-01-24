from custom_gym_envs.envs.variable_turn_tictactoe import env as VariableTurnTicTacToeEnv
import numpy as np

def test_rendering():
    # Initialize with rgb_array
    # screen_height default is 1000
    env = VariableTurnTicTacToeEnv(render_mode="rgb_array", screen_height=600)
    env.reset()
    
    # Take some steps
    for _ in range(5):
        if env.terminations[env.agent_selection] or env.truncations[env.agent_selection]:
            env.reset()
            continue
            
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            continue
            
        mask = observation["action_mask"]
        legal_moves = np.flatnonzero(mask)
        if len(legal_moves) > 0:
            env.step(legal_moves[0])
        else:
            break
            
    # Render
    frame = env.render()
    
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    # Pygame surfarray is (W, H, 3) usually, but traverse/transpose logic transforms it.
    # Code does: np.transpose(observation, axes=(1, 0, 2))
    # Pygame's array3d is (W, H, 3).
    # Transposed: (H, W, 3)?
    # Let's check shape.
    print(f"Frame shape: {frame.shape}")
    
    expected_h = 600
    expected_w = 600
    
    assert frame.shape == (expected_h, expected_w, 3)
    
    print("Rendering test passed!")

if __name__ == "__main__":
    test_rendering()
