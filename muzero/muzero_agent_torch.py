import datetime
import sys


sys.path.append("../")
from custom_gym_envs.envs.catan import ACTIONS_ARRAY

from time import time
import traceback

from numpy import save
from pygame import init

from modules.utils import scalar_to_support, support_to_scalar


from agents.random import RandomAgent

import os
from stats.stats import PlotType, StatTracker

from base_agent import agent
from base_agent.agent import MARLBaseAgent
from muzero.muzero_minmax_stats import MinMaxStats
from packages.agent_configs.agent_configs.muzero_config import MuZeroConfig
import torch
import copy
from replay_buffers.muzero_replay_buffer import MuZeroReplayBuffer, Game
from muzero.muzero_mcts import Node
from muzero.muzero_network import Network
import datetime

from utils import (
    clip_low_prob_actions,
    normalize_policies,
    action_mask,
    get_legal_moves,
    CategoricalCrossentropyLoss,
    MSELoss,
    # process_petting_zoo_obs,
    update_per_beta,
)

from modules.utils import scale_gradient

from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from tqdm import tqdm


class MuZeroAgent(MARLBaseAgent):
    def __init__(
        self,
        env,
        config: MuZeroConfig,
        name=datetime.datetime.now().timestamp(),
        test_agents=[],
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
        from_checkpoint=False,
    ):
        super(MuZeroAgent, self).__init__(
            env,
            config,
            name,
            test_agents=test_agents,
            device=device,
            from_checkpoint=from_checkpoint,
        )
        self.env.reset()  # for multiprocessing

        # Add learning rate scheduler
        self.model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            action_function=self.config.action_function,
        ).share_memory()

        print("Layer weights:")
        for name, param in self.model.named_parameters():
            if "weight" in name:
                print(f"{name}:")
                print(param.data)
                print(
                    f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
                )
            if "bias" in name:
                print(f"{name}:")
                print(param.data)
                print(
                    f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
                )

        if not self.config.multi_process:
            self.model.to(device)

        self.replay_buffer = MuZeroReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.observation_dtype,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            num_players=self.config.game.num_players,
            batch_size=self.config.minibatch_size,
            n_step=self.config.n_step,
            num_unroll_steps=self.config.unroll_steps,
            gamma=self.config.discount_factor,
            # has_intermediate_rewards=self.config.game.has_intermediate_rewards,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            epsilon=self.config.per_epsilon,
            use_batch_weights=self.config.per_use_batch_weights,
            initial_priority_max=self.config.per_initial_priority_max,
        )

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]
        self.stats = StatTracker(
            model_name=self.model_name,
            stat_keys=[
                "score",
                "policy_loss",
                "value_loss",
                "reward_loss",
                "to_play_loss",
                "loss",
                "test_score",
                "episode_length",
            ]
            + test_score_keys,
            target_values={
                "score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
                "test_score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
            },
            use_tensor_dicts={
                "test_score": ["score", "max_score", "min_score"],
                **{
                    key: ["score"]
                    + [
                        "player_{}_score".format(player)
                        for player in range(self.config.game.num_players)
                    ]
                    + [
                        "player_{}_win%".format(player)
                        for player in range(self.config.game.num_players)
                    ]
                    for key in test_score_keys
                },
            },
        )
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            # PlotType.EXPONENTIAL_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
            ema_beta=0.6,
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types(
            "policy_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "reward_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "to_play_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "episode_length", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stop_flag = mp.Value("i", 0)

    def worker_fn(
        self, worker_id, stop_flag, stats_client: StatTracker, error_queue: mp.Queue
    ):
        print(f"[Worker {worker_id}] Starting self-play...")
        # os.environ["OMP_NUM_THREADS"] = "1"
        # os.environ["MKL_NUM_THREADS"] = "1"
        # torch.set_num_threads(1)
        # torch.set_grad_enabled(False)
        worker_env = self.config.game.make_env()  # each process needs its own env
        self.model.to(self.device)
        self.model.eval()
        try:
            while not stop_flag.value:
                score, num_steps = self.play_game(env=worker_env)
                # print(f"[Worker {worker_id}] Finished a game with score {score}")
                stats_client.append("score", score)
                stats_client.append("episode_length", num_steps)
                stats_client.increment_steps(num_steps)
        except Exception as e:
            # Send both exception and traceback back
            error_queue.put((e, traceback.format_exc()))
            raise  # ensures worker process exits with error

    def train(self):
        if self.config.multi_process:
            stats_client = self.stats.get_client()
            error_queue = mp.Queue()

            workers = [
                mp.Process(
                    target=self.worker_fn,
                    args=(i, self.stop_flag, stats_client, error_queue),
                )
                for i in range(self.config.num_workers)
            ]
            for w in workers:
                w.start()

        start_time = time() - self.stats.get_time_elapsed()
        self.model.to(self.device)
        while self.training_step < self.config.training_steps:
            if self.config.multi_process:
                if not error_queue.empty():
                    err, tb = error_queue.get()

                    # Stop all workers
                    self.stop_flag.value = 1
                    for w in workers:
                        w.terminate()

                    # Re-raise the *exact same* exception type with traceback
                    print("".join(tb))  # optional: print worker traceback
                    raise err

                self.stats.drain_queue()

            if not self.config.multi_process:
                for training_game in tqdm(range(self.config.games_per_generation)):
                    if self.stop_flag.value:
                        print("Stopping game generation")
                        break

                    score, num_steps = self.play_game()
                    self.stats.append("score", score)
                    self.stats.increment_steps(num_steps)
                if self.stop_flag.value:
                    print("Stopping training")
                    break

            # STAT TRACKING
            if (self.training_step * self.config.minibatch_size + 1) / (
                max(0, self.stats.get_num_steps() - self.config.min_replay_buffer_size)
                + 1
            ) > self.config.lr_ratio:
                continue

            if self.replay_buffer.size >= self.config.min_replay_buffer_size:
                for minibatch in range(self.config.num_minibatches):
                    value_loss, policy_loss, reward_loss, to_play_loss, loss = (
                        self.learn()
                    )
                    self.stats.append("value_loss", value_loss)
                    self.stats.append("policy_loss", policy_loss)
                    # if self.config.game.has_intermediate_rewards:
                    self.stats.append("to_play_loss", to_play_loss)
                    self.stats.append("reward_loss", reward_loss)
                self.stats.append("loss", loss)
                self.training_step += 1
                print("Losses", value_loss, policy_loss, reward_loss, loss)
                print("Training Step:", self.training_step)

                self.replay_buffer.set_beta(
                    update_per_beta(
                        self.replay_buffer.beta,
                        self.config.per_beta_final,
                        self.config.training_steps,
                        self.config.per_beta,
                    )
                )

            if self.training_step % self.test_interval == 0 and self.training_step > 0:
                print("running tests")
                if self.config.multi_process:
                    try:
                        testing_worker = mp.Process(
                            target=self.run_tests, args=(stats_client,)
                        )
                        testing_worker.start()
                        self.stats.drain_queue()
                    except Exception as e:
                        print(f"Error starting testing worker: {e}")
                else:
                    self.run_tests(stats=self.stats)

            # CHECKPOINTING
            if (
                self.training_step % self.checkpoint_interval == 0
                and self.training_step > 0
            ):
                self.stats.set_time_elapsed(time() - start_time)
                print("Saving Checkpoint")
                self.save_checkpoint(
                    save_weights=self.config.save_intermediate_weights,
                )
        if self.config.multi_process:
            self.stop_flag.value = 1
            for w in workers:
                print("Stopping workers")
                w.terminate()
            print("All workers stopped")

        if self.config.multi_process:
            try:
                testing_worker.join()
            except:
                pass
            self.stats.drain_queue()

        self.stats.set_time_elapsed(time() - start_time)
        print("Finished Training")
        self.run_tests(self.stats)
        self.save_checkpoint(save_weights=True)

    def monte_carlo_tree_search(self, env, state, info):
        root = Node(0.0)
        _, policy, hidden_state = self.predict_single_initial_inference(
            state,
            info,
        )
        if self.config.game.num_players != 1:
            legal_moves = get_legal_moves(info)[0]
            to_play = env.agents.index(env.agent_selection)
        else:
            legal_moves = list(range(self.num_actions))
            to_play = 1
        root.expand(legal_moves, to_play, policy, hidden_state, 0.0)

        if env == self.env:
            root.add_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            to_play = root.to_play
            # old_to_play = to_play
            # GO UNTIL A LEAF NODE IS REACHED
            while node.expanded():
                action, node = node.select_child(
                    min_max_stats,
                    self.config.pb_c_base,
                    self.config.pb_c_init,
                    self.config.discount_factor,
                    self.config.game.num_players,
                )
                # old_to_play = (old_to_play + 1) % self.config.game.num_players
                # PREDICT THE TO PLAY HERE
                search_path.append(node)
            parent = search_path[-2]
            reward, hidden_state, value, policy, to_play = (
                self.predict_single_recurrent_inference(
                    parent.hidden_state,
                    action,  # model=model
                )
            )
            if self.config.support_range is not None:
                reward = support_to_scalar(reward, self.config.support_range)
                value = support_to_scalar(value, self.config.support_range)
            else:
                reward = reward.item()
                value = value.item()

            # onehot_to_play = to_play
            to_play = int(to_play.argmax().item())
            # if to_play != old_to_play and self.training_step > 1000:
            #     print("WRONG TO PLAY", onehot_to_play)

            node.expand(
                list(range(self.num_actions)),
                to_play,
                policy,
                hidden_state,
                (
                    reward  # if self.config.game.has_intermediate_rewards else 0.0
                ),  # for board games and games with no intermediate rewards
            )

            # for node in reversed(search_path):
            #     node.value_sum += value if node.to_play == to_play else -value
            #     node.visits += 1
            #     min_max_stats.update(
            #         node.reward
            #         + self.config.discount_factor
            #         * (
            #             node.value()
            #             if self.config.game.num_players == 1
            #             else -node.value()
            #         )
            #     )
            #     value = (
            #         -node.reward
            #         if node.to_play == to_play and self.config.game.num_players > 1
            #         else node.reward
            #     ) + self.config.discount_factor * value

            n = len(search_path)
            if n == 0:
                return []

            # --- 1) Build per-player accumulator array acc[p] = Acc_p(i) for current i (starting from i = n-1) ---
            # Acc_p(i) definition: discounted return from node i for a node whose player is p:
            # Acc_p(i) = sum_{j=i+1..n-1} discount^{j-i-1} * sign(p, j) * reward_j
            #            + discount^{n-1-i} * sign(p, leaf) * leaf_value
            # Where sign(p, j) = +1 if acting_player_at_j (which is search_path[j-1].to_play) == p else -1.
            #
            # We compute Acc_p(n-1) = sign(p, leaf) * leaf_value as base, then iterate backward:
            # Acc_p(i-1) = s(p, i) * reward_i + discount * Acc_p(i)

            # Initialize acc for i = n-1 (base: discounted exponent 0 for leaf value)
            # acc is a Python list of floats length num_players
            acc = [0.0] * self.config.game.num_players
            for p in range(self.config.game.num_players):
                acc[p] = value if to_play == p else -value

            # totals[i] will hold Acc_{node_player}(i)
            totals = [0.0] * n

            # Iterate from i = n-1 down to 0
            for i in range(n - 1, -1, -1):
                node = search_path[i]
                node_player = node.to_play
                # totals for this node = acc[node_player] (current Acc_p(i))
                totals[i] = acc[node_player]

                # Prepare acc for i-1 (if any)
                if i > 0:
                    # reward at index i belongs to acting_player = search_path[i-1].to_play
                    r_i = search_path[i].reward
                    acting_player = search_path[i - 1].to_play

                    # Update per-player accumulators in O(num_players)
                    # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # sign(p, i) = +1 if acting_player == p else -1
                    # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(self.config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + 1.0 * acc[p]

            # --- 2) Apply totals to nodes in reverse order and update MinMaxStats (parent-perspective scalar) ---
            # We must update nodes (value_sum, visits) from the leaf back to the root so that when
            # computing parent-perspective scalars we can use child.value() (which should reflect the
            # just-updated child totals).
            for i in range(n - 1, -1, -1):
                node = search_path[i]

                # apply computed discounted total for this node's player
                node.value_sum += totals[i]
                node.visits += 1

                # compute scalar that MinMaxStats expects for this child from its parent's perspective:
                # parent_value_contrib = child.reward + discount * (sign * child.value())
                # sign = +1 if single-player OR child.to_play == parent.to_play else -1
                if i > 0:
                    parent = search_path[i - 1]
                    if self.config.game.num_players == 1:
                        sign = 1.0
                    else:
                        sign = 1.0 if node.to_play == parent.to_play else -1.0
                else:
                    # root: choose sign = +1 convention (root has no parent)
                    sign = 1.0

                parent_value_contrib = node.reward + 1.0 * (sign * node.value())
                min_max_stats.update(parent_value_contrib)

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]
        return root.value(), visit_counts

    def learn(self):
        samples = self.replay_buffer.sample()
        # print("Samples:", samples)
        observations = samples["observations"]
        target_policies = samples["policy"].to(self.device)
        target_values = samples["values"].to(self.device)
        target_rewards = samples["rewards"].to(self.device)
        actions = samples["actions"].to(self.device)
        target_to_plays = samples["to_plays"].to(self.device)
        # infos = samples["infos"].to(self.device)
        inputs = self.preprocess(observations)

        for training_iteration in range(self.config.training_iterations):
            loss = 0
            val_loss = 0
            pol_loss = 0
            rew_loss = 0
            tp_loss = 0
            priorities = []
            for item in range(self.config.minibatch_size):
                value, policy, hidden_state = self.predict_single_initial_inference(
                    inputs[item], {}  # infos[item]
                )

                gradient_scales = [1.0]
                values = [value]
                rewards = [
                    (
                        torch.tensor([0.0]).to(self.device)
                        if self.config.support_range is None
                        else (
                            torch.zeros(self.config.support_range * 2 + 1)
                            .scatter(
                                0,
                                torch.tensor(
                                    [(self.config.support_range * 2 + 1) // 2]
                                ).long(),
                                1.0,
                            )
                            .to(self.device)
                        )
                    )
                ]
                policies = [policy]
                to_plays = [torch.zeros((1, self.config.game.num_players))]

                for action in actions[item]:
                    if action == -1:
                        # for self absorbing states
                        # self absorbing state, give a random action (legal moves not important as state is not technically valid)
                        # item_player = self.env.agents[infos[item]["player"]]
                        if self.config.game.num_players != 1:
                            action = self.env.action_space(self.env.agents[0]).sample()
                        else:
                            action = self.env.action_space.sample()
                    # why do we not scale the gradient of the hidden state here?? TODO
                    reward, hidden_state, value, policy, to_play = (
                        self.predict_single_recurrent_inference(hidden_state, action)
                    )
                    gradient_scales.append(1.0 / self.config.unroll_steps)
                    values.append(value)
                    rewards.append(reward)
                    policies.append(policy)
                    to_plays.append(to_play)

                    hidden_state = scale_gradient(hidden_state, 0.5)

                # print(to_plays)
                values_tensor = torch.stack([v for v in values])
                rewards_tensor = torch.stack([r for r in rewards])
                policies_tensor = torch.stack([p for p in policies])
                to_plays_tensor = torch.stack([tp for tp in to_plays])
                gradient_scales_tensor = torch.tensor(gradient_scales)

                # if self.config.game.has_intermediate_rewards:
                assert len(rewards) == len(target_rewards[item])
                assert len(values) == len(target_values[item])
                assert len(policies) == len(target_policies[item])

                for (
                    k,
                    value,
                    reward,
                    policy,
                    to_play,
                    target_value,
                    target_reward,
                    target_policy,
                    target_to_play,
                    scale,
                ) in zip(
                    range(len(values)),
                    values_tensor,
                    rewards_tensor,
                    policies_tensor,
                    to_plays_tensor,
                    target_values[item],
                    target_rewards[item],
                    target_policies[item],
                    target_to_plays[item],
                    gradient_scales_tensor,
                ):
                    if k == 0:
                        if self.config.support_range is not None:
                            priority = (
                                abs(
                                    target_value
                                    - support_to_scalar(
                                        value, self.config.support_range
                                    )
                                ).item()
                                + self.config.per_epsilon
                            )

                        else:
                            priority = (
                                abs(target_value - value.item())
                                + self.config.per_epsilon
                            )
                        priorities.append(priority)

                    if self.config.support_range is not None:
                        target_value = scalar_to_support(
                            target_value, self.config.support_range
                        )
                        # if self.config.game.has_intermediate_rewards:
                        target_reward = scalar_to_support(
                            target_reward, self.config.support_range
                        )

                    value_loss = (
                        self.config.value_loss_factor
                        * self.config.value_loss_function(value, target_value)
                    )

                    if k == 0:  # or not self.config.game.has_intermediate_rewards:
                        # NO REWARD ON INITIAL OBSERVATION
                        reward_loss = torch.tensor(0.0)
                        to_play_loss = torch.tensor(0.0)
                    else:
                        reward_loss = self.config.reward_loss_function(
                            reward, target_reward
                        )
                        to_play_loss = self.config.to_play_loss_function(
                            to_play, target_to_play
                        )

                    policy_loss = self.config.policy_loss_function(
                        policy, target_policy
                    )

                    scaled_loss = (
                        scale_gradient(
                            value_loss + reward_loss + policy_loss + to_play_loss, scale
                        )
                        * samples["weights"][
                            item
                        ]  # TODO: COULD DO A PRIORITY/WEIGHT FUNCTION THAT INCLUDES THE RECURRENT STEPS AS, SO IT DOESNT JUST MULIPTIY BY samples["weights"][item] but samples["weights"][item][k]
                    )

                    if item == 0 and self.training_step % self.checkpoint_interval == 0:
                        print("unroll step", k)
                        print("observation", observations)
                        print("predicted value", value)
                        print("target value", target_value)
                        print("predicted reward", reward)
                        print("target reward", target_reward)
                        print("predicted policy", policy)
                        print("target policy", target_policy)
                        print("to_play", to_play)
                        print("target to_player", target_to_play)
                        print(
                            "sample losses",
                            value_loss,
                            reward_loss,
                            policy_loss,
                            to_play_loss,
                        )

                    val_loss += value_loss.item()
                    rew_loss += reward_loss.item()
                    pol_loss += policy_loss.item()
                    tp_loss += to_play_loss.item()
                    loss += scaled_loss

            # compute losses
            loss = loss / self.config.minibatch_size
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

            self.optimizer.step()

            self.replay_buffer.update_priorities(samples["indices"], priorities)

        # Convert tensors to float for return values
        return (
            val_loss / self.config.minibatch_size,
            pol_loss / self.config.minibatch_size,
            rew_loss / self.config.minibatch_size,
            tp_loss / self.config.minibatch_size,
            loss.item(),
        )

    def predict_single_initial_inference(
        self,
        state,
        info,
    ):
        state_input = self.preprocess(state)
        value, policy, hidden_state = self.model.initial_inference(state_input)
        # should we action mask the priors?
        # legal_moves = get_legal_moves(info)
        # policy = action_mask(policy, legal_moves, device=self.device)
        # policy = policy / torch.sum(policy)  # Normalize policy
        return value[0], policy[0], hidden_state

    def predict_single_recurrent_inference(self, hidden_state, action):
        reward, hidden_state, value, policy, to_play = self.model.recurrent_inference(
            hidden_state, action
        )
        return reward[0], hidden_state, value[0], policy[0], to_play

    def predict(
        self,
        state,
        info: dict = None,
        env=None,
        temperature=1.0,
        *args,
        **kwargs,
    ):
        value, visit_counts = self.monte_carlo_tree_search(
            env,
            state,
            info,  # model=model
        )
        actions = [action for _, action in visit_counts]
        visit_counts = torch.tensor(
            [count for count, _ in visit_counts], dtype=torch.float32
        )

        temperature_visit_counts = torch.pow(visit_counts, 1 / temperature)
        temperature_visit_counts /= torch.sum(temperature_visit_counts)

        target_policy = torch.zeros(self.num_actions)
        target_policy[actions] = visit_counts / torch.sum(visit_counts)

        # SHOULD TARGET POLICY BE TEMPERATURE VISIT COUNTS???
        return temperature_visit_counts, target_policy, actions, value

    def select_actions(self, predictions, *args, **kwargs):
        probs = torch.zeros(self.num_actions)
        for i, action in enumerate(predictions[2]):
            probs[action] = predictions[0][i].item()
        action = torch.multinomial(probs, 1)
        return action

    def play_game(self, env=None):
        if env is None:
            env = self.env
        # if model is None:
        #     model = self.model
        with torch.no_grad():
            # initialization_time = time()
            if self.config.game.num_players != 1:
                env.reset()
                state, reward, terminated, truncated, info = env.last()
                agent_id = env.agent_selection
                current_player = env.agents.index(agent_id)
            else:
                state, info = env.reset()

            # state, info = process_petting_zoo_obs(state, info, current_player)
            game = Game(self.config.game.num_players)

            game.append(state, info)

            done = False
            while not done:
                # total_game_step_time = time()
                temperature = self.config.temperatures[0]
                for i, temperature_step in enumerate(self.config.temperature_updates):
                    if self.config.temperature_with_training_steps:
                        if self.training_step >= temperature_step:
                            temperature = self.config.temperatures[i + 1]
                        else:
                            break
                    else:
                        if len(game) >= temperature_step:
                            temperature = self.config.temperatures[i + 1]
                        else:
                            break

                # prediction_wait_time = time()
                prediction = self.predict(
                    state,
                    info,
                    env=env,
                    temperature=temperature,  # model=model
                )
                # print(f"Prediction took {time()-prediction_wait_time} seconds")
                # action_wait_time = time()
                action = self.select_actions(prediction).item()
                # print(f"Action selection took {time()-action_wait_time} seconds")
                if self.replay_buffer.size < 100000:
                    print(
                        f"Turn {env.game.state.num_turns} Action {len(game) + 1}: {ACTIONS_ARRAY[action]}"
                    )

                # env_step_time = time()
                if self.config.game.num_players != 1:
                    env.step(action)
                    next_state, _, terminated, truncated, next_info = env.last()
                    reward = env.rewards[env.agents[current_player]]
                    agent_id = env.agent_selection
                    current_player = env.agents.index(agent_id)
                else:
                    next_state, reward, terminated, truncated, next_info = env.step(
                        action
                    )
                # next_state, next_info = process_petting_zoo_obs(
                #     next_state, next_info, current_player
                # )
                # print(f"Environment step took {time()-env_step_time} seconds")
                done = terminated or truncated
                # essentially storing in memory, dont store terminal states for training as they are not predicted on
                game.append(
                    next_state,
                    next_info,
                    reward,
                    prediction[1],
                    value=prediction[3],
                    action=action,
                )
                # game.append(
                #     state,
                #     info,
                #     reward,
                #     prediction[1],
                #     value=prediction[3],
                #     action=action,
                # )

                state = next_state
                info = next_info
            self.replay_buffer.store(game)
        if self.config.game.num_players != 1:
            return env.rewards[env.agents[0]], len(game)
        else:
            return sum(game.rewards), len(game)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stop_flag"] = state["stop_flag"].value
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stop_flag = mp.Value("i", state["stop_flag"])
