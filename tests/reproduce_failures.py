
import torch
import numpy as np
from search.search_factories import create_mcts
from tests.test_batched_search import MockConfig, MockInference

def test_bs0_flakiness():
    print("Testing BS=0 Flakiness...")
    batch_size = 0
    virtual_loss = 1.0
    
    config = MockConfig(batch_size=batch_size, virtual_loss=virtual_loss)
    config.num_simulations = 400
    config.root_exploration_fraction = 0.0
    
    mocker = MockInference(mode="biased")
    inference_fns = mocker.get_fns()
    
    # Run 10 times
    for i in range(10):
        mcts = create_mcts(config, "cpu", config.num_actions)
        state = torch.zeros(1, 10)
        info = {"legal_moves": [0, 1, 2, 3]}
        val, _, target_policy, best_action, _ = mcts.run(state, info, 0, inference_fns)
        if best_action != 2:
            print(f"FAILED iter {i}: Action {best_action}, Policy {target_policy}")
        else:
            if i == 0:
                print(f"Passed iter {i}")

def test_gumbel_batched_failure(use_virtual_mean=False):
    print(f"\nTesting Gumbel Batched (VM={use_virtual_mean})...")
    state = torch.zeros(1, 10)
    info = {"legal_moves": [0, 1, 2, 3]}
    mocker = MockInference(mode="biased")
    inference_fns = mocker.get_fns()

    torch.manual_seed(42)
    # Batched Gumbel
    config_batch = MockConfig(batch_size=4, gumbel=True, use_virtual_mean=use_virtual_mean)
    # If using VL (default), VL=1.0. If VM, VL irrelevant.
    config_batch.virtual_loss = 1.0
    
    mcts_batch = create_mcts(config_batch, "cpu", 4)
    _, _, target_batch, action_batch, _ = mcts_batch.run(state, info, 0, inference_fns)
    
    print(f"Best Action: {action_batch}")
    print(f"Target Policy: {target_batch}")

if __name__ == "__main__":
    test_bs0_flakiness()
    test_gumbel_batched_failure(use_virtual_mean=False) 
    test_gumbel_batched_failure(use_virtual_mean=True) 
