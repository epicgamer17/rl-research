"""AlphaZero on PettingZoo TicTacToe (Single-Process Version)

================================================================================

Paper Reference:
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm"
    Silver et al., Science 2018 (arXiv 2017: https://arxiv.org/abs/1712.01815)

Single-Process Architecture:
    - Self-Play Collector: Steps vectorized environments simultaneously using batched MCTS
      (root embeddings shape [NUM_VECTOR_ENVS, 3, 3, 2]).
    - Learner: Continuously samples minibatches from ReplayBuffer, computes policy and value loss,
      and updates weights.
    - Evaluator: Periodically evaluates the model against a Random baseline in PettingZoo tictactoe_v3,
      logging metrics to W&B.

    NOTE: Although our MCTS is based on mctx, we use vector values for each player for rewards and values.
    This does two things. First and most important, it allows the MCTS to work on n-player games,
    cooperative or competitive. Second, it is more in line with PettingZoo standards, which return a reward
    for each player, not a single scalar reward. For this example, we use the knowledge that Tic Tac Toe
    is zero sum; on another game you could avoid using that knowledge, and have the agent predict and learn
    multiple values and rewards for each state.
"""

import copy
import random
import time
from typing import Any, Callable, Dict, List, Tuple

from atomic_rl.action_selection import sample_distribution
from atomic_rl.buffers.replay import (
    circular_write_strategy_,
    init_buffer,
    uniform_sample,
)
from atomic_rl.envs.functions.tictactoe import (
    check_tictactoe_winner,
    embeddings_to_canonical,
    get_canonical_obs,
    tictactoe_dynamics_fn,
)
from atomic_rl.losses import cross_entropy_loss, mse_loss
from atomic_rl.networks import ResNetBackbone
from atomic_rl.search import get_mcts_visit_policy, mcts_search
from pettingzoo.classic import tictactoe_v3

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# TODO: use get_legal_action_mask
# TODO: use argmax selector for evaluation
# TODO: Update so that +1 always means canonically winning. So player 2 should not learn to predict -1 in winning positions like i think we do right now. We should correctly use the pettingzoo reward vector that is returned for the value of a state.
# TODO: why didn't we use any of our returns.py or returns/ folder or whatever (compute_mc_return fn). is it because we are not working on whole game sequences? should we update to use that?


# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_VECTOR_ENVS = 4  # Parallel vectorized self-play environments
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
# 1. Dual-Head AlphaZero ResNet Neural Network (Scalar Value Head)
# ============================================================================
class TicTacToeNet(nn.Module):
    def __init__(
        self, num_filters: int = NUM_FILTERS, num_res_blocks: int = NUM_RES_BLOCKS
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
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
# 2. Helpers & Recurrent Function Creator
# ============================================================================
# TODO: i think we have a function for this already. Use it.
def apply_action_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Masks illegal actions to the minimum finite value of the dtype."""
    min_logit = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_mask, min_logit)


def create_recurrent_fn(local_model: nn.Module) -> Callable:
    """Creates an N-player-compliant recurrent_fn for PyTorch MCTS."""

    def recurrent_fn(actions_taken: torch.Tensor, embeddings: torch.Tensor) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # 1. Step environment dynamics (reward_p0 is ALWAYS P0 absolute: +1 for P0 win, -1 for P1 win)
        (
            next_embed,
            reward_p0,
            next_to_play,
            is_terminal,
            next_legal_mask,
        ) = tictactoe_dynamics_fn(embeddings, actions_taken)

        # 2. Evaluate policy logits and scalar value prediction for next state
        with torch.no_grad():
            canonical_next = embeddings_to_canonical(next_embed)
            logits, scalar_value = local_model(canonical_next)

        # 3. Mask illegal actions
        masked_logits = apply_action_mask(logits, next_legal_mask)

        # 4. Value Conversion to Absolute [P0, P1]:
        # scalar_value is canonical for next_to_play.
        # If next_to_play == 0, v_p0 = scalar_value.
        # If next_to_play == 1, v_p0 = -scalar_value.
        v_p0 = torch.where(next_to_play == 0, scalar_value, -scalar_value)
        value_vec = torch.stack([v_p0, -v_p0], dim=-1)

        # Terminal nodes have 0 future value
        value_vec = torch.where(
            is_terminal.unsqueeze(-1), torch.zeros_like(value_vec), value_vec
        )

        # Reward Conversion to Absolute [P0, P1]:
        reward_vec = torch.stack([reward_p0, -reward_p0], dim=-1)

        # Discounts: 1.0 for non-terminal, 0.0 for terminal
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


# ============================================================================
# 3. Vectorized Self-Play Collector
# ============================================================================
class VectorizedSelfPlayCollector:
    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.boards = torch.zeros(num_envs, 3, 3, device=device)
        self.players = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.move_counts = [0] * num_envs
        self.trajectories = [[] for _ in range(num_envs)]

    def collect_step(
        self,
        actor_model: nn.Module,
        recurrent_fn: Callable,
        num_simulations: int = NUM_MCTS_SIMULATIONS,
    ) -> List[Dict[str, torch.Tensor]]:
        # Construct root embeddings for search batch size [B, 3, 3, 2]
        root_embed = self.boards.new_zeros(self.num_envs, 3, 3, 2)
        root_embed[..., 0] = self.boards
        root_embed[..., 1] = self.players.view(-1, 1, 1).expand(-1, 3, 3).float()
        legal_mask = self.boards.view(self.num_envs, -1) == 0.0

        # Root Model Prediction (Scalar Head)
        with torch.no_grad():
            canonical_x = embeddings_to_canonical(root_embed)
            root_logits, root_scalar_value = actor_model(canonical_x)

        # Map scalar value to vector [P0, P1] based on active player
        v_p0_root = torch.where(
            self.players == 0, root_scalar_value, -root_scalar_value
        )
        root_vector_value = torch.stack([v_p0_root, -v_p0_root], dim=-1)

        # Execute N-Player MCTS Search
        search_action, action_probs, tree = mcts_search(
            root_embeddings=root_embed,
            root_logits=root_logits,
            root_value=root_vector_value,
            root_to_play=self.players,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            num_actions=9,
            num_players=2,
            legal_mask=legal_mask,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            dirichlet_alpha=DIRICHLET_ALPHA,
            pb_c_init=C_PUCT,
            temperature=1.0,
        )

        completed_samples = []

        # Step each env forward
        for b_idx in range(self.num_envs):
            self.move_counts[b_idx] += 1
            curr_player = self.players[b_idx].item()

            # Target policy from search
            target_policy = action_probs[b_idx]

            # Select action according to move-count temperature schedule
            temp = (
                TEMPERATURE_EXPLORATION
                if self.move_counts[b_idx] <= TEMP_THRESHOLD_MOVES
                else TEMPERATURE_EXPLOITATION
            )

            if temp == 0.0:
                action_idx = target_policy.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs=target_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action_idx = action_idx_t.item()

            canonical_obs = get_canonical_obs(self.boards[b_idx], curr_player).squeeze(
                0
            )

            self.trajectories[b_idx].append(
                {
                    "state": canonical_obs.cpu(),
                    "target_policy": target_policy.cpu(),
                    "player": curr_player,
                }
            )

            # Fast internal dynamics step
            row, col = action_idx // 3, action_idx % 3
            piece = 1.0 if curr_player == 0 else -1.0
            self.boards[b_idx, row, col] = piece

            winner, is_term = check_tictactoe_winner(self.boards[b_idx].unsqueeze(0))

            if is_term.item():
                p0_reward = winner.item()
                p1_reward = -p0_reward

                for step in self.trajectories[b_idx]:
                    pl = step["player"]
                    z = p0_reward if pl == 0 else p1_reward
                    completed_samples.append(
                        {
                            "state": step["state"],
                            "target_policy": step["target_policy"],
                            "target_value": torch.tensor([z], dtype=torch.float32),
                        }
                    )

                # Reset environment for new game
                self.boards[b_idx].zero_()
                self.players[b_idx] = 0
                self.move_counts[b_idx] = 0
                self.trajectories[b_idx] = []
            else:
                self.players[b_idx] = 1 - curr_player

        return completed_samples


# ============================================================================
# 4. Baseline Evaluator (AlphaZero vs. Random Player)
# ============================================================================
def evaluate_vs_random(
    model: nn.Module,
    num_games: int = NUM_EVAL_GAMES,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluates trained AlphaZero model against a Random agent.

    Plays half games as Player 0 ('player_1'), half as Player 1 ('player_2').
    Tracks separate P1 and P2 statistics.
    """
    model.eval()
    recurrent_fn = create_recurrent_fn(model)

    p1_wins, p1_draws, p1_losses = 0, 0, 0
    p2_wins, p2_draws, p2_losses = 0, 0, 0

    for game_i in range(num_games):
        az_player = 0 if game_i % 2 == 0 else 1
        az_agent_name = "player_1" if az_player == 0 else "player_2"
        env = tictactoe_v3.env()
        env.reset()

        board_3x3 = torch.zeros(3, 3, device=device)
        az_reward = 0.0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if agent == az_agent_name and reward != 0:
                az_reward = reward

            if termination or truncation:
                env.step(None)
                continue

            player = 0 if agent == "player_1" else 1
            action_mask = torch.tensor(
                obs["action_mask"], device=device, dtype=torch.bool
            )
            legal_actions = action_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

            if player == az_player:
                root_embed = board_3x3.new_zeros(1, 3, 3, 2)
                root_embed[0, ..., 0] = board_3x3
                root_embed[0, ..., 1] = float(player)

                with torch.no_grad():
                    canonical_x = embeddings_to_canonical(root_embed)
                    root_logits, root_scalar_value = model(canonical_x)

                player_t = torch.tensor([player], device=device, dtype=torch.long)
                v_p0_root = torch.where(
                    player_t == 0, root_scalar_value, -root_scalar_value
                )
                root_vector_value = torch.stack([v_p0_root, -v_p0_root], dim=-1)

                search_action, action_probs, tree = mcts_search(
                    root_embeddings=root_embed,
                    root_logits=root_logits,
                    root_value=root_vector_value,
                    root_to_play=player_t,
                    recurrent_fn=recurrent_fn,
                    num_simulations=num_simulations,
                    num_actions=9,
                    num_players=2,
                    legal_mask=action_mask.unsqueeze(0),
                    dirichlet_epsilon=0.0,
                    pb_c_init=C_PUCT,
                    temperature=0.0,
                )

                action_idx = search_action[0].item()
            else:
                action_idx = random.choice(legal_actions)

            row, col = action_idx // 3, action_idx % 3
            piece = 1.0 if player == 0 else -1.0
            board_3x3[row, col] = piece

            env.step(action_idx)

        if az_player == 0:
            if az_reward > 0:
                p1_wins += 1
            elif az_reward < 0:
                p1_losses += 1
            else:
                p1_draws += 1
        else:
            if az_reward > 0:
                p2_wins += 1
            elif az_reward < 0:
                p2_losses += 1
            else:
                p2_draws += 1

    model.train()
    total_p1 = max(1, p1_wins + p1_draws + p1_losses)
    total_p2 = max(1, p2_wins + p2_draws + p2_losses)
    total_all = num_games

    return {
        "eval/win_rate": (p1_wins + p2_wins) / total_all,
        "eval/draw_rate": (p1_draws + p2_draws) / total_all,
        "eval/loss_rate": (p1_losses + p2_losses) / total_all,
        "eval/p1_win_rate": p1_wins / total_p1,
        "eval/p1_draw_rate": p1_draws / total_p1,
        "eval/p1_loss_rate": p1_losses / total_p1,
        "eval/p2_win_rate": p2_wins / total_p2,
        "eval/p2_draw_rate": p2_draws / total_p2,
        "eval/p2_loss_rate": p2_losses / total_p2,
    }


# ============================================================================
# 5. Main AlphaZero Training Loop
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
        "state": (3, 3, 3),
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
            "temp_threshold_moves": TEMP_THRESHOLD_MOVES,
            "batch_size": BATCH_SIZE,
            "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_filters": NUM_FILTERS,
            "num_res_blocks": NUM_RES_BLOCKS,
            "num_eval_games": NUM_EVAL_GAMES,
            "seed": SEED,
        },
    )
    wandb.define_metric("*", step_metric="global_step")

    collector = VectorizedSelfPlayCollector(NUM_VECTOR_ENVS, device=device)
    actor_recurrent_fn = create_recurrent_fn(actor_model)

    initial_eval = evaluate_vs_random(
        learner_model,
        num_games=NUM_EVAL_GAMES,
        num_simulations=NUM_MCTS_SIMULATIONS,
        device=device,
    )
    wandb.log(
        {
            "global_step": 0,
            "eval/win_rate": initial_eval["eval/win_rate"],
            "eval/draw_rate": initial_eval["eval/draw_rate"],
            "eval/loss_rate": initial_eval["eval/loss_rate"],
            "eval/p1_win_rate": initial_eval["eval/p1_win_rate"],
            "eval/p1_draw_rate": initial_eval["eval/p1_draw_rate"],
            "eval/p1_loss_rate": initial_eval["eval/p1_loss_rate"],
            "eval/p2_win_rate": initial_eval["eval/p2_win_rate"],
            "eval/p2_draw_rate": initial_eval["eval/p2_draw_rate"],
            "eval/p2_loss_rate": initial_eval["eval/p2_loss_rate"],
        }
    )

    for step in range(1, TOTAL_TRAINING_STEPS + 1):
        if step % PARAM_SYNC_INTERVAL == 0:
            actor_model.load_state_dict(learner_model.state_dict())

        completed_samples = collector.collect_step(
            actor_model=actor_model,
            recurrent_fn=actor_recurrent_fn,
            num_simulations=NUM_MCTS_SIMULATIONS,
        )

        if len(completed_samples) > 0:
            batch_td = TensorDict(
                {
                    "state": torch.stack([s["state"] for s in completed_samples]),
                    "target_policy": torch.stack(
                        [s["target_policy"] for s in completed_samples]
                    ),
                    "target_value": torch.stack(
                        [s["target_value"] for s in completed_samples]
                    ),
                },
                batch_size=[len(completed_samples)],
            ).to(device)
            buffer_state, _ = circular_write_strategy_(buffer_state, batch_td)

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

        log_dict = {
            "global_step": step,
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "buffer/size": buffer_state.size,
        }

        if step % EVAL_INTERVAL_STEPS == 0 or step == TOTAL_TRAINING_STEPS:
            eval_metrics = evaluate_vs_random(
                learner_model,
                num_games=NUM_EVAL_GAMES,
                num_simulations=NUM_MCTS_SIMULATIONS,
                device=device,
            )
            log_dict.update(eval_metrics)

        if step % 100 == 0:
            print(
                f"[Step {step}/{TOTAL_TRAINING_STEPS}] Loss: {loss.item():.4f} | Buffer: {buffer_state.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


if __name__ == "__main__":
    train_alphazero_tictactoe()
