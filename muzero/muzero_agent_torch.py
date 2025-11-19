import datetime
import math
import random
import sys

from packages.utils.utils.utils import KLDivergenceLoss


sys.path.append("../")
from custom_gym_envs.envs.catan import ACTIONS_ARRAY

from time import time
import traceback

from numpy import save
from pygame import init

from modules.utils import scalar_to_support, support_to_scalar


from agents.random import RandomAgent
import numpy as np
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

        self.target_model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            action_function=self.config.action_function,
        )
        # copy weights
        self.target_model.load_state_dict(self.model.state_dict())
        # self.model.share_memory()

        if self.config.multi_process:
            # make sure target is placed in shared memory so worker processes can read it
            self.target_model.share_memory()
        else:
            # non-multiprocess: keep target on device for faster inference
            self.target_model.to(device)

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

    def update_target_model(self):
        """Copy parameters from the learning model to the target model.
        This uses load_state_dict under torch.no_grad() for safety and speed.
        """
        with torch.no_grad():
            # Direct load_state_dict is safe and simple.

            # print("Layer weights before:")
            # for name, param in self.target_model.named_parameters():
            #     if "weight" in name:
            #         print(f"{name}:")
            #         print(param.data)
            #         print(
            #             f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
            #         )
            #     if "bias" in name:
            #         print(f"{name}:")
            #         print(param.data)
            #         print(
            #             f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
            #         )

            self.target_model.load_state_dict(self.model.state_dict())

            # print("Layer weights after:")
            # for name, param in self.target_model.named_parameters():
            #     if "weight" in name:
            #         print(f"{name}:")
            #         print(param.data)
            #         print(
            #             f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
            #         )
            #     if "bias" in name:
            #         print(f"{name}:")
            #         print(param.data)
            #         print(
            #             f"Shape: {param.shape}, std: {param.std():.4f}, mean: {param.mean():.4f}\n"
            #         )

            # If using multiprocessing, ensure target remains in shared memory.
            if self.config.multi_process:
                self.target_model.share_memory()

    def worker_fn(
        self, worker_id, stop_flag, stats_client: StatTracker, error_queue: mp.Queue
    ):
        print(f"[Worker {worker_id}] Starting self-play...")
        # os.environ["OMP_NUM_THREADS"] = "1"
        # os.environ["MKL_NUM_THREADS"] = "1"
        # torch.set_num_threads(1)
        # torch.set_grad_enabled(False)
        worker_env = self.config.game.make_env()  # each process needs its own env
        # from utils import record_video_wrapper

        # worker_env.render_mode = "rgb_array"
        # worker_env = record_video_wrapper(
        #     worker_env, f"./videos/{self.model_name}/{worker_id}", 1
        # )
        # Workers should use the target model for inference so training doesn't
        # destabilize ongoing self-play. Ensure the target model is on the worker's device
        # and set as the inference model.
        self.target_model.to(self.device)
        self.target_model.eval()

        try:
            while not stop_flag.value:
                if (
                    random.random() < self.config.reanalyze_ratio
                    and self.replay_buffer.size > 0
                ):
                    self.reanalyze_game(inference_model=self.target_model)
                else:
                    score, num_steps = self.play_game(
                        env=worker_env, inference_model=self.target_model
                    )
                    # print(f"[Worker {worker_id}] Finished a game with score {score}")
                    # worker_env.close()  # for saving video
                    stats_client.append("score", score)
                    stats_client.append("episode_length", num_steps)
                    stats_client.increment_steps(num_steps)
        except Exception as e:
            # Send both exception and traceback back
            error_queue.put((e, traceback.format_exc()))
            raise  # ensures worker process exits with error
        worker_env.close()

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

        # ensure inference uses the current target before any play in main thread
        self.inference_model = self.target_model

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
                # if not (
                #     len(self.stats.stats["score"]) < 5
                #     or not all(x == 0 for x in self.stats.stats["score"][-5:])
                # ):
                #     if self.config.multi_process:
                #         self.stop_flag.value = 1
                #         for w in workers:
                #             print("Stopping workers early")
                #             w.terminate()
                #         print("All workers stopped early")

                #     if self.config.multi_process:
                #         try:
                #             testing_worker.join()
                #         except:
                #             pass
                #         self.stats.drain_queue()
                #     self.env.close()

                # assert len(self.stats.stats["score"]) < 5 or not all(
                #     x == 0 for x in self.stats.stats["score"][-5:]
                # ), "last 5 games are truncated (for catan)"

            if not self.config.multi_process:
                for training_game in tqdm(range(self.config.games_per_generation)):
                    if self.stop_flag.value:
                        print("Stopping game generation")
                        break

                    score, num_steps = self.play_game(inference_model=self.target_model)
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
                # print("Losses", value_loss, policy_loss, reward_loss, loss)
                # print("Training Step:", self.training_step)

                self.replay_buffer.set_beta(
                    update_per_beta(
                        self.replay_buffer.beta,
                        self.config.per_beta_final,
                        self.config.training_steps,
                        self.config.per_beta,
                    )
                )

                if self.training_step % self.config.transfer_interval == 0:
                    # print(
                    #     f"Transferring weights to target model at step {self.training_step}"
                    # )
                    self.update_target_model()

            if self.training_step % self.test_interval == 0 and self.training_step > 0:
                # print("running tests")
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
                # print("Saving Checkpoint")
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
        self.env.close()

    def monte_carlo_tree_search(
        self, state, info, to_play, trajectory_action=None, inference_model=None
    ):
        root = Node(0.0)
        v_pi_raw, policy, hidden_state = self.predict_single_initial_inference(
            state,
            info,
            model=inference_model,
        )

        if self.config.support_range is not None:
            # support_to_scalar expects a support vector and returns a scalar tensor
            v_pi = support_to_scalar(v_pi_raw, self.config.support_range)
            # ensure it's a Python float where you use .item() later
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(v_pi_raw.item())

        if self.config.game.num_players != 1:
            legal_moves = get_legal_moves(info)[0]  # [0]
            # print(legal_moves)
        else:
            legal_moves = list(range(self.num_actions))
            to_play = 0

        # print("traj_action", trajectory_action)
        # print("legal moves", legal_moves)

        if self.config.gumbel:
            actions = legal_moves  # list of ints
            # policy is a tensor/prob vector for all actions (shape num_actions)
            logits = torch.log(policy + 1e-12).cpu()  # use numpy for gumbel math

            # legal_moves is the list of available actions

            # --- Gumbel sampling ---
            k = len(actions)
            m = min(
                self.config.gumbel_m, k
            )  # add config param gumbel_m, e.g., min(n,16)
            # sample Gumbel noise
            g = -torch.log(-torch.log(torch.rand(k)))  # shape (k,)
            # compute g + logits (only for legal actions)
            scores = g + logits[actions]
            # find top-m indices (indices into 'actions')
            top_idx = torch.argsort(scores, descending=True)[:m]
            sampled_actions = [actions[i] for i in top_idx]
            if trajectory_action is None:
                sampled_g_values = {
                    actions[i]: float(g[i] + logits[actions[i]]) for i in top_idx
                }
            else:
                if trajectory_action not in sampled_actions:
                    sampled_actions += [trajectory_action]
                    print(top_idx)
                    top_idx = torch.concat(
                        top_idx, torch.tensor([actions.index(trajectory_action)])
                    )
                    print(top_idx)

                if self.config.reanalyze_noise:
                    sampled_g_values = {
                        actions[i]: float(g[i] + logits[actions[i]]) for i in top_idx
                    }
                else:
                    sampled_g_values = {
                        actions[i]: float(logits[actions[i]]) for i in top_idx
                    }

            # expand root with only sampled_actions; pass in the per-child root_score
            root.expand(
                sampled_actions, to_play, policy, hidden_state, 0.0, value=v_pi_scalar
            )

            # attach the root_score to the created children
            for a in sampled_actions:
                root.children[a].root_score = sampled_g_values[a]  # store Gumbel+logit
        else:
            # TODO: add sample muzero for complex action spaces (right now just use gumbel)
            # print("policy", policy)
            root.expand(legal_moves, to_play, policy, hidden_state, 0.0)
            # print("root.children", root.children)
            if trajectory_action is None or self.config.reanalyze_noise:
                root.add_noise(
                    self.config.root_dirichlet_alpha,
                    self.config.root_exploration_fraction,
                )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        if trajectory_action is not None:
            # print("injecting trajectory action into distribution")
            # ensure action exists as child
            # print("root.children", root.children)
            assert (
                trajectory_action in root.children
            ), f"trajectory_action not in root.children, make sure if there is one it is garaunteed to be in the sampled actions, trajectory action: {trajectory_action}, root.children: {root.children}, legal_moves: {legal_moves}, info: {info}, buffer size {self.replay_buffer.size}"
            inject_frac = self.config.injection_frac  # 0.25 as paper used
            # renormalize priors: put (1-inject_frac) of current mass on existing priors, add inject_frac on the trajectory action
            # compute sum of current priors
            total_prior = sum(child.prior_policy for child in root.children.values())
            for a, child in root.children.items():
                child.prior_policy = (1.0 - inject_frac) * (
                    child.prior_policy / total_prior
                )
            # boost injected action
            root.children[trajectory_action].prior_policy += inject_frac
            # print("root.children after injecting", root.children)

        if self.config.gumbel:
            best_action = self.sequential_halving(
                root, min_max_stats, list(root.children.keys())
            )
        else:
            for _ in range(self.config.num_simulations):
                self._run_single_simulation(
                    root,
                    min_max_stats,
                    inference_model=inference_model,
                    allowed_actions=None,
                )

        # visit_counts = torch.tensor(
        #     [(child.visits, action) for action, child in root.children.items()]
        # )
        # print("root children", root.children.items())
        visit_counts = torch.tensor(
            [child.visits for action, child in root.children.items()]
        )
        actions = [action for action, child in root.children.items()]

        # print("mcts visit counts", visit_counts)

        policy = torch.zeros(self.num_actions)
        policy[actions] = visit_counts / torch.sum(visit_counts)

        if self.config.gumbel:
            # q(a) for visited actions (empirical Q from child.value())
            q_dict = {a: float(root.children[a].value()) for a in actions}

            # sum of visits across all children (sum_b N(b))
            sum_N = float(sum(root.children[a].visits for a in root.children.keys()))

            # p_vis_sum := sum_{b in visited} pi(b)  (network policy mass on visited actions)
            p_vis_sum = float(sum(policy[a] for a in actions))

            # expected_q_vis := sum_{a in visited} pi(a) * q(a)
            expected_q_vis = float(sum(policy[a] * q_dict[a] for a in actions))

            # term := sum_b N(b) * ( p_vis_sum * expected_q_vis )
            term = sum_N * (p_vis_sum * expected_q_vis)
            v_mix = (v_pi_scalar + term) / (1.0 + sum_N)

            # completedQ: visited actions keep q(a), unvisited set to v_mix
            completedQ = torch.full((self.num_actions,), v_mix.item())
            for a, qv in q_dict.items():
                completedQ[a] = qv

            # --- Build the improved policy π0 using completedQ ---
            # Compute sigma(completedQ) exactly as you do in selection (use min_max_stats or your normalization)
            # Here I reuse the same sigma computation you already use; replace with your exact implementation if different.

            # Normalization of completedQ: use MinMaxStats if you want consistency
            # Assuming min_max_stats exists and is the same used in search:
            normalized_completed = torch.tensor(
                [min_max_stats.normalize(float(x)) for x in completedQ]
            )

            # compute sigma per-action: sigma = (cvisit + max_visits) * cscale * normalized_completed
            max_visits = (
                max([ch.visits for ch in root.children.values()])
                if len(root.children) > 0
                else 0
            )
            sigma = (
                (self.config.gumbel_cvisit + max_visits)
                * self.config.gumbel_cscale
                * normalized_completed
            )  # numpy array length num_actions

            # combine network logits and sigma:
            logits = torch.log(policy + 1e-12)  # network log-probabilities
            pi0_logits = logits + sigma  # elementwise
            # pi0 = np.exp(pi0_logits - np.max(pi0_logits))
            # pi0 /= pi0.sum() + 1e-12
            pi0 = torch.softmax(pi0_logits, dim=0)

            # Final target policy (torch tensor on device for training)
            target_policy = torch.tensor(pi0, dtype=torch.float32).to(self.device)

            return root.value(), policy, target_policy, best_action
        else:
            return root.value(), policy, policy, torch.argmax(policy)

    def sequential_halving(
        self, root: Node, min_max_stats: MinMaxStats, candidates: list
    ):
        """
        Perform Sequential Halving among `candidates` (list of action ints).
        It splits self.config.num_simulations across rounds that eliminate ~half of candidates each round.
        Survivors remain in root.children and accumulate visits/values as usual.
        """
        n = len(candidates)
        # if n <= 1:
        #     return

        # number of rounds to reduce to 1 (ceil log2)
        rounds = max(1, math.ceil(math.log2(n)))
        total_sims = self.config.num_simulations
        # allocate sims per round roughly equal — you can refine this strategy
        sims_per_round = max(1, total_sims // rounds)

        survivors = candidates.copy()

        def _calculate_score(action, root, child, min_max_stats, config):
            # root_score: stored Gumbel+logit
            root_score = child.root_score

            # parent-perspective scalar (same as used in selection/backprop)
            if config.game.num_players == 1:
                sign = 1.0
            else:
                sign = 1.0 if child.to_play == root.to_play else -1.0

            parent_value_contrib = child.reward + config.discount_factor * (
                sign * child.value()
            )

            # normalize using the shared MinMaxStats
            normalized_q = min_max_stats.normalize(parent_value_contrib)

            # visit mass scaling
            max_visits = (
                max([ch.visits for ch in root.children.values()])
                if len(root.children) > 0
                else 0
            )
            cvisit = config.gumbel_cvisit
            cscale = config.gumbel_cscale
            sigma = (cvisit + max_visits) * cscale * normalized_q

            score = float(root_score + sigma)
            return (action, score)

        # Helper function for sorting scores by value (used for both priority and elimination)
        def sort_by_score(item):
            # item is a tuple (action, score)
            return item[1]

        sims_used = 0
        for r in range(rounds):
            if len(survivors) <= 1:
                break

            # # compute a score per survivor for elimination (use visits primarily)
            scores = []
            for a in survivors:
                child = root.children.get(a)
                if child is not None:
                    scores.append(
                        _calculate_score(a, root, child, min_max_stats, self.config)
                    )
            scores.sort(key=sort_by_score, reverse=True)
            survivors = [a for a, _ in scores]
            num_survivors = len(survivors)
            # run sims_per_round simulations restricted to current survivors
            sims_used += sims_per_round

            for i in range(sims_per_round):
                # The modulo operation cycles through the sorted survivors list
                action = survivors[i % num_survivors]

                # Run a single simulation, but ONLY allow the current `action` to be selected at the root
                self._run_single_simulation(
                    root, min_max_stats, allowed_actions=[action]
                )  # recompute a score per survivor for elimination (use visits primarily)

            scores = []
            for a in survivors:
                child = root.children.get(a)
                if child is not None:
                    scores.append(
                        _calculate_score(a, root, child, min_max_stats, self.config)
                    )  # sort ascending by computed score, eliminate bottom half
            scores.sort(key=sort_by_score)
            survivors = [a for a, _ in scores]
            num_survivors = len(survivors)
            num_to_eliminate = max(1, math.ceil(len(survivors) / 2.0))
            eliminated = [a for a, _ in scores[:num_to_eliminate]]
            survivors = [a for a in survivors if a not in eliminated]
            # print(survivors, scores)

        remaining = max(0, total_sims - sims_used)
        for _ in range(remaining):
            # free to explore among survivors or all children (paper uses final sims to refine survivors)
            # print("running remaining", _)
            self._run_single_simulation(
                root, min_max_stats, allowed_actions=set(survivors)
            )
        if len(survivors) > 1:
            return scores[-1][0]
        else:
            return survivors[0]

    def _run_single_simulation(
        self,
        root: Node,
        min_max_stats: MinMaxStats,
        inference_model=None,
        allowed_actions=None,
    ):
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
                self.config.gumbel_cvisit,
                self.config.gumbel_cscale,
                allowed_actions=allowed_actions,
                gumbel=self.config.gumbel,
            )
            # old_to_play = (old_to_play + 1) % self.config.game.num_players
            # PREDICT THE TO PLAY HERE
            search_path.append(node)
        parent = search_path[-2]
        reward, hidden_state, value, policy, to_play = (
            self.predict_single_recurrent_inference(
                parent.hidden_state, action, model=inference_model
            )
        )
        if self.config.support_range is not None:
            reward = support_to_scalar(reward, self.config.support_range).item()
            value = support_to_scalar(value, self.config.support_range).item()
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
            value=value,
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
        # print("n", n)
        # print("acc", acc)
        # Iterate from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            node = search_path[i]
            node_player = node.to_play
            # totals for this node = acc[node_player] (current Acc_p(i))
            # print(totals[i])
            # print(acc[node_player])
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
                    acc[p] = sign * r_i + self.config.discount_factor * acc[p]

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

            parent_value_contrib = node.reward + self.config.discount_factor * (
                sign * node.value()
            )
            min_max_stats.update(parent_value_contrib)

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
                    inputs[item], {}, model=self.model  # infos[item]
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
                    if torch.isnan(action).item():
                        # for self absorbing states
                        # self absorbing state, give a random action (legal moves not important as state is not technically valid)
                        # item_player = self.env.agents[infos[item]["player"]]
                        if self.config.game.num_players != 1:
                            action = self.env.action_space(self.env.agents[0]).sample()
                        else:
                            action = self.env.action_space.sample()
                    # why do we not scale the gradient of the hidden state here?? TODO
                    else:
                        action = int(action.item())
                    reward, hidden_state, value, policy, to_play = (
                        self.predict_single_recurrent_inference(
                            hidden_state, action, model=self.model
                        )
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
                        ).to(self.device)
                        target_reward = scalar_to_support(
                            target_reward, self.config.support_range
                        ).to(self.device)

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
                        if self.config.game.num_players != 1:
                            to_play_loss = (
                                self.config.to_play_loss_factor
                                * self.config.to_play_loss_function(
                                    to_play, target_to_play
                                )
                            )
                        else:
                            to_play_loss = torch.tensor(0.0)

                    # if self.config.gumbel and not isinstance(
                    #     self.config.policy_loss_function, KLDivergenceLoss
                    # ):
                    #     print("Warning gumbel should us KL Divergence Loss")
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

                    # if item == 0 and self.training_step % self.checkpoint_interval == 0:
                    # print("unroll step", k)
                    # print("observation", observations[item])
                    # print("predicted value", value)
                    # print("target value", target_value)
                    # print("predicted reward", reward)
                    # print("target reward", target_reward)
                    # print("predicted policy", policy)
                    # print("target policy", target_policy)
                    # print("to_play", to_play)
                    # print("target to_player", target_to_play)
                    # print(
                    #     "sample losses",
                    #     value_loss,
                    #     reward_loss,
                    #     policy_loss,
                    #     to_play_loss,
                    # )

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

            self.replay_buffer.update_priorities(
                samples["indices"], priorities, ids=samples["ids"]
            )

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
        model,
    ):
        if model == None:
            model = self.model
        state_input = self.preprocess(state)
        # print("state input shape", state_input.shape)
        value, policy, hidden_state = model.initial_inference(state_input)
        # should we action mask the priors?
        # legal_moves = get_legal_moves(info)
        # policy = action_mask(policy, legal_moves, device=self.device)
        # policy = policy / torch.sum(policy)  # Normalize policy
        # print("policy shape", policy.shape)
        return value[0], policy[0], hidden_state

    def predict_single_recurrent_inference(self, hidden_state, action, model):
        if model == None:
            model = self.model
        reward, hidden_state, value, policy, to_play = model.recurrent_inference(
            hidden_state, action
        )
        return reward[0], hidden_state, value[0], policy[0], to_play

    def predict(
        self,
        state,
        info: dict = None,
        env=None,
        inference_model=None,
        *args,
        **kwargs,
    ):
        if self.config.game.num_players != 1:
            to_play = env.agents.index(env.agent_selection)
        else:
            to_play = 0
        value, policy, target_policy, best_action = self.monte_carlo_tree_search(
            state, info, to_play, inference_model=inference_model  # model=model
        )
        return policy, target_policy, value, best_action

    def select_actions(
        self,
        prediction,
        temperature=0.0,
        *args,
        **kwargs,
    ):
        # print("probs", predictions[0])
        if temperature != 0:
            probs = prediction[0] ** temperature
            probs /= probs.sum()
            # print("temp probs", probs)
            action = torch.multinomial(probs, 1)
            return action
        else:
            return prediction[3]

    def play_game(self, env=None, inference_model=None):
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
                    inference_model=inference_model,
                )
                # print(f"Prediction took {time()-prediction_wait_time} seconds")
                # action_wait_time = time()
                action = self.select_actions(
                    prediction,
                    temperature=temperature,  # model=model
                ).item()
                # print(f"Action selection took {time()-action_wait_time} seconds")
                # if self.replay_buffer.size < 100000:
                #     print(
                #         f"Turn {env.game.state.num_turns} Action {len(game) + 1}: {ACTIONS_ARRAY[action]}"
                #     )

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
                    value=prediction[2],
                    action=action,
                )
                # game.append(
                #     state,
                #     info,
                #     reward,
                #     prediction[1],
                #     value=prediction[2],
                #     action=action,
                # )

                state = next_state
                info = next_info
            self.replay_buffer.store(game)
        if self.config.game.num_players != 1:
            return env.rewards[env.agents[0]], len(game)
        else:
            return sum(game.rewards), len(game)

    def reanalyze_game(self, inference_model=None):
        # or reanalyze buffer
        sample = self.replay_buffer.sample_game()
        observations = sample["observations"]
        root_values = sample["values"].to(self.device)[:, 0]
        rewards = sample["rewards"].to(self.device)[:, 1]
        traj_actions = sample["actions"].to(self.device)[:, 0]
        traj_to_plays = sample["to_plays"].to(self.device)[:, 0]
        legal_moves_masks = sample["legal_moves_masks"].to(self.device)
        indices = sample["indices"]
        ids = sample["ids"]
        # print("root_values", root_values)
        # print("traj to_plays", traj_to_plays)
        # print("traj actions", traj_actions)
        # print("legal move masks", legal_moves_masks)

        new_policies = []
        new_root_values = []
        new_priorities = []
        infos = []
        for (
            idx,
            obs,
            root_value,
            traj_action,
            traj_to_play,
            mask,
        ) in zip(
            indices,
            observations,
            root_values,
            traj_actions,
            traj_to_plays,
            legal_moves_masks,
        ):
            to_play = int(torch.argmax(traj_to_play).item())
            info = {
                "legal_moves": torch.nonzero(mask).view(-1).tolist(),
                "player": to_play,
            }
            assert not (
                "legal_moves" in info and len(info["legal_moves"]) == 0
            ), f"no legal moves, invalid sample {info}"
            infos.append(info)
            # print("info with legal moves from nonzero mask", info)
            # ADD INJECTING SEEN ACTION THING FROM MUZERO UNPLUGGED
            if self.config.reanalyze_method == "mcts":
                root_value, _, new_policy, best_ac_tion = self.monte_carlo_tree_search(
                    obs,
                    info,  # FOR LEGAL MOVES
                    to_play,
                    int(traj_action.item()),
                    inference_model=inference_model,
                )

                new_root_value = float(root_value)
            else:
                value, new_policy, _ = self.predict_single_initial_inference(
                    obs, info, model=inference_model
                )
                new_root_value = value

            # decide value target per your config (paper default: keep stored n-step TD for Atari)
            stored_n_step_value = float(
                self.replay_buffer.n_step_values_buffer[idx][0].item()
            )

            new_policies.append(new_policy)
            new_root_values.append(new_root_value)
            new_priorities.append(
                abs(float(root_value) - stored_n_step_value)
                + self.replay_buffer.epsilon
            )

        # now write back under write_lock and update priorities with ids
        # print("rewards", rewards)
        # print("traj_actions", traj_actions)
        # print("infos", infos)
        self.replay_buffer.reanalyze_game(
            indices, new_policies, new_root_values, rewards, traj_actions, infos, ids
        )
        if self.config.reanalyze_update_priorities:
            self.replay_buffer.update_priorities(
                indices, new_priorities, ids=np.array(ids)
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stop_flag"] = state["stop_flag"].value
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stop_flag = mp.Value("i", state["stop_flag"])
