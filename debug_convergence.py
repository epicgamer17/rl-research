
import torch
import numpy as np
import random
from types import SimpleNamespace
from search.search_factories import create_mcts
from search.nodes import DecisionNode, ChanceNode

# Mock Infrastructure (Copied from test_batched_search.py)
class MockConfig:
    def __init__(self, batch_size=1, gumbel=False, stochastic=False, virtual_loss=3.0, use_virtual_mean=False):
        self.action_space_size = 4
        self.num_actions = 4
        self.discount_factor = 0.99
        self.value_prefix = True
        self.lstm_hidden_size = 16
        self.lstm_horizon_len = 5
        self.inverse_value_transform = lambda x: x
        self.support_range = None
        self.pb_c_init = 1.25
        self.pb_c_base = 19652
        self.long_term_discount = 1.0
        self.game = SimpleNamespace(num_players=1, is_discrete=True)
        self.num_simulations = 2000 
        self.gumbel = gumbel
        self.stochastic = stochastic
        
        # Gumbel specific
        self.gumbel_m = 4
        self.gumbel_cvisit = 50
        self.gumbel_cscale = 1.0
        
        # Search params
        self.search_batch_size = batch_size
        self.virtual_loss = virtual_loss
        self.known_bounds = None
        self.soft_update = False
        self.min_max_epsilon = 1e-6
        self.q_estimation_method = "mcts_value"
        self.root_dirichlet_alpha_adaptive = False
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.0 # No noise
        self.use_virtual_mean = use_virtual_mean

class MockInference:
    def __init__(self, mode="biased", branching_bias=None):
        self.mode = mode
        self.branching_bias = branching_bias
        self.sleep_time = 0.0
        self.call_count = 0
        self.action_2_visits = 0

    def initial(self, state, model=None):
        if self.mode == "biased":
            val = torch.tensor(0.5)
            policy = torch.ones(1, 4) / 4.0
            hidden = torch.zeros(1, 16)
            return val, policy, hidden

    def recurrent(self, state, action, rh, rc, model=None):
        B = state.shape[0]
        # Normalize action to (B, 1) to be safe
        # Normalize action to (B, 1) to be safe
        if action.dim() == 1:
            action = action.unsqueeze(1)
            
        # print(f"DEBUG Recurrent Action Input: {action}")
        if self.call_count < 20: # Limit spam
             print(f"DEBUG Recurrent Action Input: {action.tolist()}")
            
        rewards = torch.zeros(B, 1)
        hidden = torch.zeros(B, 16)
        values = torch.ones(B) * 0.5
        policies = torch.ones(B, 4) / 4.0
        
        self.call_count += B
        for i in range(B):
            a = int(action[i, 0].item())
            if a == 2:
                self.action_2_visits += 1
                rewards[i, 0] = 1.0
                values[i] = 1.0
                # print(f"DEBUG: Visited Action 2! R=1.0, V=1.0 at batch idx {i}")
            
            if self.call_count < 50:
                 print(f"DEBUG MOCK: Action={a} Reward={rewards[i, 0].item()}")
            # else:
                # print(f"DEBUG: Visited Action {a}. R=0.0, V=0.5")
        
        to_play = torch.zeros(B, 1)
        rh_new = torch.zeros(B, 1, 16)
        rc_new = torch.zeros(B, 1, 16)
        return rewards, hidden, values, policies, to_play, rh_new, rc_new

    def get_fns(self):
        return {
            "initial": self.initial,
            "recurrent": self.recurrent,
            "afterstate": lambda *args, **kwargs: (None, None, None)
        }

def run_debug(batch_size, virtual_loss):
    print(f"\n--- Running Debug for BatchSize={batch_size}, VL={virtual_loss} ---")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    state = torch.zeros(1, 10)
    info = {"legal_moves": [0, 1, 2, 3]}
    mocker = MockInference(mode="biased")
    inference_fns = mocker.get_fns()
    
    config = MockConfig(batch_size=batch_size, virtual_loss=virtual_loss)
    mcts = create_mcts(config, "cpu", config.num_actions)
    
    # Hook the root node creation or inspect after run?
    # We can inspect the root node stats if we capture it.
    # But currently run() creates root internally.
    # We can't access it easily unless we modify run() or use the return values if possible.
    # But MCTS run returns (value, policy, target, action).
    # We can rely on print debugging inside search/modular_search.py OR just look at the high level output first.
    
    # To get more detail, we can hack the search to print children stats
    # But let's first see the policy distribution.
    val, policy, target_policy, best_action = mcts.run(state, info, 0, inference_fns)
    
    print(f"Best Action: {best_action}")
    print(f"Target Policy: {target_policy}")
    print(f"Action 2 Visits in Mock: {state.sum() if isinstance(state, torch.Tensor) else 'N/A'}") # wait, state is tensor.
    # Better: Track in MockInference
    print(f"Total calls to recurrent: {mocker.call_count}")
    print(f"Action 2 visits: {mocker.action_2_visits}")

if __name__ == "__main__":
    run_debug(batch_size=0, virtual_loss=0.0) # Iterative
    run_debug(batch_size=1, virtual_loss=0.0) # Batched BS=1
