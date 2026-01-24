
import torch
import numpy as np
from search.search_factories import create_mcts
from tests.test_batched_search import MockConfig, MockInference

def debug_bs0():
    print("Debugging BS=0 Failure...")
    batch_size = 0
    virtual_loss = 1.0
    
    # Setup
    state = torch.zeros(1, 10)
    info = {"legal_moves": [0, 1, 2, 3]}
    mocker = MockInference(mode="biased")
    inference_fns = mocker.get_fns()
    
    config = MockConfig(batch_size=batch_size, virtual_loss=virtual_loss)
    config.num_simulations = 400
    config.root_exploration_fraction = 0.0
    
    # Run
    mcts = create_mcts(config, "cpu", config.num_actions)
    val, _, target_policy, best_action, _ = mcts.run(state, info, 0, inference_fns)
    
    print(f"Result Best Action: {best_action}")
    print(f"Target Policy: {target_policy}")
    
    # Inspect Root Children
    root = mcts.root # Accessing internal root if possible. 
    # Wait, mcts.run doesn't return root.
    # But I can modify the script or just debug referencing the run method.
    # Ah, I mocked `run` in my logic? No, I'm calling real `run`.
    # I can't access root easily unless I modify code or use a debugger.
    # BUT, I can perform a trick: mock `backpropagator` to print stats?
    # Or just use the `tests/test_batched_search.py` approach of subclassing or mocking.
    
    # Actually, let's just inspect the policy and valid action values.
    # The MockInference "biased" mode: action 2 -> Reward 1.0, Value 1.0. Others -> Reward 0.0, Value 0.5.
    
    # Let's check if the inference is actually CALLED correctly.
    # Maybe checking what `mocker` sees.
    
if __name__ == "__main__":
    debug_bs0()
