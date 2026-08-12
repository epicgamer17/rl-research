"""AlphaZero on PettingZoo TicTacToe (torch.multiprocessing Multiprocess Version)

================================================================================

Paper Reference:
    "Mastering Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm"
    Silver et al., Science 2018 (arXiv 2017: https://arxiv.org/abs/1712.01815)

Multiprocessing Architecture:
    - Actor Processes: Each actor runs batched self-play across parallel
      PettingZoo environments using batched MCTS (root embeddings [ENVS, 2, 3, 3]).
    - Learner Process: Continuously samples minibatches from SharedReplayBuffer,
      computes policy and value loss, updates network weights, and syncs shared model.
    - Evaluator Process: Periodically evaluates the shared model against a Random
      baseline in PettingZoo tictactoe_v3, logging metrics to W&B asynchronously.

    NOTE: Although or MCTS is based on mctx, we use vector values for each player for rewards and values. This does two things. First and most important, it allows the MCTS to work on n-player games, cooperative or competitive. Second it is more inline with PettingZoo standards, which return a reward for each player, not a single scalar reward. For this example, we use the knowledge that Tic Tac Toe is zero sum, on another game you could avoid using that knowledge, and have the agent predict and learn multiple values and rewards for each state.
"""

import random
import time
from typing import Callable, Dict, List, Tuple

from atomic_rl.action_selection import (
    apply_action_mask,
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers import SharedReplayBuffer
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
import torch.multiprocessing as mp
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
NUM_ACTORS = 3
ENVS_PER_ACTOR = 3
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
# 3. Multiprocessing Workers
# ============================================================================
def actor_worker(
    actor_id: int,
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    envs_per_actor: int = ENVS_PER_ACTOR,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device_str: str = "cpu",
):
    """Worker process generating self-play trajectories via batched MCTS."""
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    torch.manual_seed(SEED + actor_id * 1000)
    random.seed(SEED + actor_id * 1000)

    envs = [
        PettingZooChannelFirstWrapper(tictactoe_v3.env()) for _ in range(envs_per_actor)
    ]
    for env in envs:
        env.reset()

    move_counts = [0] * envs_per_actor
    trajectories: List[List[Tuple[torch.Tensor, torch.Tensor, int]]] = [
        [] for _ in range(envs_per_actor)
    ]

    step_counter = 0
    recurrent_fn = create_recurrent_fn(local_model)

    while True:
        step_counter += 1
        if step_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        completed_tds = []
        obs_tensors, legal_mask_tensors, curr_player_list = [], [], []

        for b_idx, env in enumerate(envs):
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                rewards_vec = get_terminal_rewards_vec(env)
                states, policies, step_players = zip(*trajectories[b_idx])
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

                trajectories[b_idx] = []
                move_counts[b_idx] = 0
                env.reset()
                obs, reward, term, trunc, _ = env.last()

            curr_player_list.append(get_agent_index(env))
            obs_tensors.append(
                torch.tensor(obs["observation"], device=device, dtype=torch.float32)
            )
            legal_mask_tensors.append(
                torch.tensor(obs["action_mask"], device=device, dtype=torch.bool)
            )

        root_embed = torch.stack(obs_tensors, dim=0)
        legal_masks = torch.stack(legal_mask_tensors, dim=0)
        curr_players = torch.tensor(curr_player_list, device=device, dtype=torch.long)

        action_probs = run_batched_mcts(
            local_model,
            recurrent_fn,
            root_embed,
            curr_players,
            legal_masks,
            num_simulations=num_simulations,
            temperature=1.0,
            explore_dirichlet=True,
        )

        for b_idx, env in enumerate(envs):
            move_counts[b_idx] += 1
            curr_player = curr_players[b_idx].item()
            target_policy = action_probs[b_idx]

            temp = get_stepwise_schedule(
                step=move_counts[b_idx],
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

            trajectories[b_idx].append(
                (root_embed[b_idx].cpu(), target_policy.cpu(), curr_player)
            )
            env.step(action_idx)

        if len(completed_tds) > 0:
            batch_td = torch.cat(completed_tds, dim=0).to(device)
            buffer.add_samples(batch_td)


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    num_eval_games: int = NUM_EVAL_GAMES,
    device_str: str = "cpu",
):

    """Background process evaluating model performance asynchronously."""
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()

    def select_eval_action(env, curr_player_idx, obs_t, action_mask_t):
        recurrent_fn = create_recurrent_fn(local_model)
        root_embed = obs_t.unsqueeze(0)
        legal_masks = action_mask_t.unsqueeze(0)
        players_t = torch.tensor([curr_player_idx], device=device, dtype=torch.long)

        action_probs = run_batched_mcts(
            local_model,
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

    while True:
        local_model.load_state_dict(shared_model.state_dict())
        eval_metrics = evaluate_pettingzoo_vs_random(
            env_fn=lambda: PettingZooChannelFirstWrapper(tictactoe_v3.env()),
            select_action_fn=select_eval_action,
            num_games=num_eval_games,
            device=device,
        )
        eval_queue.put(eval_metrics)



def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    eval_queue: mp.Queue,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    """Learner process training model weights on sampled minibatches."""
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.train()
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = torch.optim.Adam(
        local_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    wandb.init(
        project="alphazero-tictactoe",
        name=f"alphazero_mp_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_actors{NUM_ACTORS}",
        config={
            "total_training_steps": max_steps,
            "num_actors": NUM_ACTORS,
            "envs_per_actor": ENVS_PER_ACTOR,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "c_puct": C_PUCT,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    )

    step_count = 0
    latest_eval_metrics = {}

    while step_count < max_steps:
        minibatch = buffer.sample_batch(rng_key, batch_size, min_size=MIN_BUFFER_SIZE)
        if minibatch is None:
            time.sleep(0.01)
            continue

        states = minibatch["state"].to(device)
        target_policies = minibatch["target_policy"].to(device)
        target_values = minibatch["target_value"].to(device)

        policy_logits, predicted_value = local_model(states)

        raw_p_loss, _ = cross_entropy_loss(policy_logits, target_policies)
        policy_loss = raw_p_loss.mean()

        raw_v_loss, _ = mse_loss(predicted_value.view(-1), target_values.view(-1))
        value_loss = raw_v_loss.mean()

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
        optimizer.step()

        shared_model.load_state_dict(local_model.state_dict())
        step_count += 1

        while not eval_queue.empty():
            try:
                latest_eval_metrics = eval_queue.get_nowait()
            except Exception:
                break

        log_dict = {
            "global_step": step_count,
            "loss/total": loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "buffer/size": buffer.size,
        }
        log_dict.update(latest_eval_metrics)

        if step_count % 100 == 0:
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {loss.item():.4f} | Buffer: {buffer.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 4. Main Multiprocessing Pipeline Setup
# ============================================================================
def model_creator_fn() -> TicTacToeNet:
    return TicTacToeNet(num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device_str = "cpu"
    device = torch.device(device_str)

    shared_model = model_creator_fn().to(device)
    shared_model.share_memory()

    buffer_shapes = {
        "state": (2, 3, 3),
        "target_policy": (9,),
        "target_value": (1,),
    }
    buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)
    eval_queue = mp.Queue()

    processes = []

    learner_p = mp.Process(
        target=learner_worker,
        args=(model_creator_fn, shared_model, buffer, eval_queue),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, model_creator_fn, shared_model, buffer),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    eval_p = mp.Process(
        target=evaluator_worker,
        args=(model_creator_fn, shared_model, eval_queue),
        kwargs={"device_str": device_str},
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    print(
        f"Launched AlphaZero Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Evaluator)."
    )
    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()

    print("AlphaZero Multiprocessing Training Completed Successfully.")


if __name__ == "__main__":
    main()
