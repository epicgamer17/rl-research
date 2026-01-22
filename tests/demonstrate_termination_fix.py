import torch
import numpy as np

def compute_n_step_old(raw_rewards, raw_values, raw_dones, valid_mask, n_step, gamma):
    batch_size = 1
    unroll_steps = 0
    target_values = torch.zeros((batch_size, unroll_steps + 1))
    
    u = 0
    computed_value = torch.zeros(batch_size)
    has_ended = torch.zeros(batch_size, dtype=torch.bool)
    
    for k in range(n_step):
        r_idx = u + k
        # Old Logic: check has_ended which was set by PREVIOUS step's raw_dones[r_idx]
        r_is_valid = valid_mask[:, r_idx] & (~has_ended)
        
        reward_chunk = (gamma**k) * raw_rewards[:, r_idx]
        computed_value += torch.where(r_is_valid, reward_chunk, torch.tensor(0.0))
        
        # Old Logic: update has_ended based on current state terminality
        has_ended = has_ended | (raw_dones[:, r_idx] & valid_mask[:, r_idx])
        
    # Bootstrap logic omitted for simplicity or simplified
    target_values[:, u] = computed_value
    return target_values

def compute_n_step_new(raw_rewards, raw_values, raw_dones, valid_mask, n_step, gamma):
    batch_size = 1
    unroll_steps = 0
    target_values = torch.zeros((batch_size, unroll_steps + 1))
    
    u = 0
    computed_value = torch.zeros(batch_size)
    has_ended = torch.zeros(batch_size, dtype=torch.bool)
    
    for k in range(n_step):
        r_idx = u + k
        r_is_valid = valid_mask[:, r_idx] & (~has_ended)
        
        reward_chunk = (gamma**k) * raw_rewards[:, r_idx]
        computed_value += torch.where(r_is_valid, reward_chunk, torch.tensor(0.0))
        
        # New Logic: update has_ended based on NEXT state terminality
        if r_idx + 1 < raw_dones.shape[1]:
            has_ended = has_ended | (raw_dones[:, r_idx + 1])
            
    target_values[:, u] = computed_value
    return target_values

def run_demonstration():
    # Scenario: Episode ends at index 1 (transition 0->1 is the last)
    # Replay buffer circularly wraps or contains data from a new game at index 2
    raw_rewards = torch.tensor([[10.0, 50.0, 999.0, 999.0]]) # 999 is noise from next game
    raw_dones = torch.tensor([[False, True, False, False]]) # State 1 is terminal
    valid_mask = torch.tensor([[True, True, True, True]])   # Assume all valid for this demo
    n_step = 3
    gamma = 1.0
    
    # Expected: V0 = Reward(0->1) = 10.0. 
    # Transition 1->2 shouldn't happen because State 1 is terminal.
    
    v_old = compute_n_step_old(raw_rewards, None, raw_dones, valid_mask, n_step, gamma)
    v_new = compute_n_step_new(raw_rewards, None, raw_dones, valid_mask, n_step, gamma)
    
    print("--- Termination Logic Demonstration ---")
    print(f"Rewards: {raw_rewards[0].tolist()}")
    print(f"Dones (State is terminal): {raw_dones[0].tolist()}")
    print(f"Old Target Value (V0): {v_old[0,0].item()} (Includes noise from next transition!)")
    print(f"New Target Value (V0): {v_new[0,0].item()} (Correctly cuts off at terminal state)")
    
    if v_old[0,0].item() != v_new[0,0].item():
        print("\nConclusion: The old logic allowed the reward from the transition FOLLOWING a terminal state")
        print("to be included in the N-step return because it checked 'done' at the START of the transition")
        print("instead of the END.")

if __name__ == "__main__":
    run_demonstration()
