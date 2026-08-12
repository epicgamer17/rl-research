"""AlphaZero on PettingZoo TicTacToe (Single-Process Version)

================================================================================

Paper Reference:
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm"
    Silver et al., Science 2018 (arXiv 2017: https://arxiv.org/abs/1712.01815)

Single-Process Architecture:
    - Self-Play Collector: Steps vectorized environments simultaneously using batched MCTS
      (root embeddings [NUM_VECTOR_ENVS, 2, 3, 3]).
    - Learner: Continuously samples minibatches from ReplayBuffer, computes policy and value loss,
      and updates network weights.
    - Evaluator: Periodically evaluates the model against a Random baseline in PettingZoo tictactoe_v3,
      logging metrics to W&B.
"""

import copy
import random
from typing import Callable, Dict, List, Tuple

from atomic_rl.action_selection import (
    apply_action_mask,
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers.replay import (
    circular_write_strategy_,
    init_buffer,
    uniform_sample,
)
from atomic_rl.envs import get_agent_index, get_terminal_rewards_vec
from atomic_rl.envs.functions.tictactoe import tictactoe_dynamics_fn
from atomic_rl.envs.wrappers import PettingZooChannelFirstWrapper
from atomic_rl.evaluation import evaluate_pettingzoo_vs_random
from atomic_rl.losses import cross_entropy_loss, mse_loss
from atomic_rl.networks import ResNetBackbone
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import mcts_search
from pettingzoo.classic import tictactoe_v3

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_VECTOR_ENVS = 4
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL_STEPS = 250
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 25

# MCTS PUCT Search Constants
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Temperature Schedule Constants
TEMP_THRESHOLD_MOVES = 5
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0

# Optimization & Network Parameters
BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_FILTERS = 24
NUM_RES_BLOCKS = 6

NUM_EVAL_GAMES = 100
SEED = 42


# ============================================================================
# 1. Dual-Head AlphaZero ResNet Neural Network
# ============================================================================
class TicTacToeNet(nn.Module):
    def __init__(
        self, num_filters: int = NUM_FILTERS, num_res_blocks: int = NUM_RES_BLOCKS
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(2, num_filters, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = ResNetBackbone(
            in_channels=num_filters,
            num_filters=num_filters,
            num_blocks=num_res_blocks,
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 9),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = F.relu(self.bn_in(self.conv_in(x)))
        features = self.res_blocks(features)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value.squeeze(-1)


# ============================================================================
# 2. MCTS Search Wrappers & Helpers
# ============================================================================
def create_recurrent_fn(model: nn.Module) -> Callable:
    """Creates an N-player compliant recurrent_fn for MCTS simulation."""

    def recurrent_fn(
        actions_taken: torch.Tensor,
        embeddings: torch.Tensor,
        to_play: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            next_embed,
            reward_vec,
            next_to_play,
            is_terminal,
            next_legal_mask,
        ) = tictactoe_dynamics_fn(embeddings, actions_taken, to_play)

        with torch.no_grad():
            logits, scalar_value = model(next_embed)

        masked_logits = apply_action_mask(logits, next_legal_mask)
        v_p0 = torch.where(next_to_play == 0, scalar_value, -scalar_value)
        value_vec = torch.stack([v_p0, -v_p0], dim=-1)

        value_vec = torch.where(
            is_terminal.unsqueeze(-1), torch.zeros_like(value_vec), value_vec
        )
        discount_vec = torch.where(
            is_terminal.unsqueeze(-1),
            torch.zeros_like(reward_vec),
            torch.ones_like(reward_vec),
        )

        return (
            masked_logits,
            value_vec,
            reward_vec,
            discount_vec,
            next_embed,
            next_to_play,
        )

    return recurrent_fn


def run_batched_mcts(
    model: nn.Module,
    recurrent_fn: Callable,
    root_embed: torch.Tensor,
    curr_players: torch.Tensor,
    legal_masks: torch.Tensor,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    temperature: float = 1.0,
    explore_dirichlet: bool = True,
) -> torch.Tensor:
    """Runs batched MCTS search across parallel env states."""
    with torch.no_grad():
        root_logits, root_scalar_value = model(root_embed)

    v_p0_root = torch.where(curr_players == 0, root_scalar_value, -root_scalar_value)
    root_vector_value = torch.stack([v_p0_root, -v_p0_root], dim=-1)
    dirichlet_eps = DIRICHLET_EPSILON if explore_dirichlet else 0.0

    action_probs, _ = mcts_search(
        root_embeddings=root_embed,
        root_logits=root_logits,
        root_value=root_vector_value,
        root_to_play=curr_players,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        num_actions=9,
        num_players=2,
        legal_mask=legal_masks,
        dirichlet_epsilon=dirichlet_eps,
        dirichlet_alpha=DIRICHLET_ALPHA,
        pb_c_init=C_PUCT,
        temperature=temperature,
    )
    return action_probs


# ============================================================================
# 3. Vectorized Self-Play Environment Collector
# ============================================================================
class VectorizedSelfPlayCollector:
    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.envs = [
            PettingZooChannelFirstWrapper(tictactoe_v3.env()) for _ in range(num_envs)
        ]
        for env in self.envs:
            env.reset()

        self.move_counts = [0] * num_envs
        self.trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, int]]] = [
            [] for _ in range(num_envs)
        ]

    def step_all(
        self,
        actor_model: nn.Module,
        recurrent_fn: Callable,
        num_simulations: int = NUM_MCTS_SIMULATIONS,
    ) -> torch.Tensor:
        completed_tds = []
        obs_tensors, legal_mask_tensors, curr_player_list = [], [], []

        for b_idx, env in enumerate(self.envs):
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                rewards_vec = get_terminal_rewards_vec(env)
                states, policies, step_players = zip(*self.trajectories[b_idx])
                players_t = torch.tensor(step_players)
                targets_t = rewards_vec[players_t].unsqueeze(-1)

                completed_tds.append(
                    TensorDict(
                        {
                            "state": torch.stack(states),
                            "target_policy": torch.stack(policies),
                            "target_value": targets_t,
                        },
                        batch_size=[len(states)],
                    )
                )

                self.trajectories[b_idx] = []
                self.move_counts[b_idx] = 0
                env.reset()
                obs, reward, term, trunc, _ = env.last()

            curr_player_list.append(get_agent_index(env))
            obs_tensors.append(
                torch.tensor(
                    obs["observation"], device=self.device, dtype=torch.float32
                )
            )
            legal_mask_tensors.append(
                torch.tensor(obs["action_mask"], device=self.device, dtype=torch.bool)
            )

        root_embed = torch.stack(obs_tensors, dim=0)
        legal_masks = torch.stack(legal_mask_tensors, dim=0)
        curr_players = torch.tensor(
            curr_player_list, device=self.device, dtype=torch.long
        )

        action_probs = run_batched_mcts(
            actor_model,
            recurrent_fn,
            root_embed,
            curr_players,
            legal_masks,
            num_simulations=num_simulations,
            temperature=1.0,
            explore_dirichlet=True,
        )

        for b_idx, env in enumerate(self.envs):
            self.move_counts[b_idx] += 1
            curr_player = curr_players[b_idx].item()
            target_policy = action_probs[b_idx]

            temp = get_stepwise_schedule(
                step=self.move_counts[b_idx],
                thresholds=[TEMP_THRESHOLD_MOVES],
                values=[TEMPERATURE_EXPLORATION, TEMPERATURE_EXPLOITATION],
            )


            if temp == 0.0:
                action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
                action_idx = action_idx_tensor.squeeze().item()
            else:
                dist = torch.distributions.Categorical(probs=target_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action_idx = action_idx_t.item()

            self.trajectories[b_idx].append(
                (root_embed[b_idx].cpu(), target_policy.cpu(), curr_player)
            )
            env.step(action_idx)

        if len(completed_tds) > 0:
            return torch.cat(completed_tds, dim=0).to(self.device)
        return None


# ============================================================================
# 4. Main AlphaZero Training Loop
# ============================================================================
def train_alphazero_tictactoe():
    """Main AlphaZero self-play training script."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    random.seed(SEED)

    learner_model = TicTacToeNet(
        num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS
    ).to(device)
    actor_model = copy.deepcopy(learner_model).to(device)
    actor_model.eval()

    optimizer = torch.optim.Adam(
        learner_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    buffer_shapes = {
        "state": (2, 3, 3),
        "target_policy": (9,),
        "target_value": (1,),
    }
    buffer_state = init_buffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)

    wandb.init(
        project="alphazero-tictactoe",
        name=f"alphazero_continuous_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_envs{NUM_VECTOR_ENVS}",
        config={
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "num_vector_envs": NUM_VECTOR_ENVS,
            "min_buffer_size": MIN_BUFFER_SIZE,
            "eval_interval_steps": EVAL_INTERVAL_STEPS,
            "param_sync_interval": PARAM_SYNC_INTERVAL,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "c_puct": C_PUCT,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "dirichlet_epsilon": DIRICHLET_EPSILON,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
    )

    collector = VectorizedSelfPlayCollector(NUM_VECTOR_ENVS, device=device)
    step_count = 0

    def select_eval_action(env, curr_player_idx, obs_t, action_mask_t):
        recurrent_fn = create_recurrent_fn(actor_model)
        root_embed = obs_t.unsqueeze(0)
        legal_masks = action_mask_t.unsqueeze(0)
        players_t = torch.tensor([curr_player_idx], device=device, dtype=torch.long)

        action_probs = run_batched_mcts(
            actor_model,
            recurrent_fn,
            root_embed,
            players_t,
            legal_masks,
            num_simulations=NUM_MCTS_SIMULATIONS,
            temperature=0.0,
            explore_dirichlet=False,
        )
        action_idx_tensor, _ = argmax_selector(action_probs)
        return action_idx_tensor.squeeze().item()

    while step_count < TOTAL_TRAINING_STEPS:
        recurrent_fn = create_recurrent_fn(actor_model)
        new_tds = collector.step_all(
            actor_model, recurrent_fn, num_simulations=NUM_MCTS_SIMULATIONS
        )
        if new_tds is not None:
            buffer_state, _ = circular_write_strategy_(buffer_state, new_tds)

        if buffer_state.size < MIN_BUFFER_SIZE:
            continue

        minibatch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

        states = minibatch["state"].to(device)
        target_policies = minibatch["target_policy"].to(device)
        target_values = minibatch["target_value"].to(device)

        policy_logits, predicted_value = learner_model(states)

        raw_p_loss, _ = cross_entropy_loss(policy_logits, target_policies)
        policy_loss = raw_p_loss.mean()

        raw_v_loss, _ = mse_loss(predicted_value.view(-1), target_values.view(-1))
        value_loss = raw_v_loss.mean()

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)
        optimizer.step()

        step_count += 1

        if step_count % PARAM_SYNC_INTERVAL == 0:
            actor_model.load_state_dict(learner_model.state_dict())

        eval_metrics = {}
        if step_count % EVAL_INTERVAL_STEPS == 0:
            eval_metrics = evaluate_pettingzoo_vs_random(
                env_fn=lambda: PettingZooChannelFirstWrapper(tictactoe_v3.env()),
                select_action_fn=select_eval_action,
                num_games=NUM_EVAL_GAMES,
                device=device,
            )

        log_dict = {
            "global_step": step_count,
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "buffer/size": buffer_state.size,
        }
        log_dict.update(eval_metrics)

        if step_count % 100 == 0:
            print(
                f"[Step {step_count}/{TOTAL_TRAINING_STEPS}] Loss: {loss.item():.4f} | Buffer Size: {buffer_state.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


if __name__ == "__main__":
    train_alphazero_tictactoe()
