import datetime
import math
import random
import sys

from modules.utils import KLDivergenceLoss, MSELoss


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
import torch.nn.functional as F
import copy
from replay_buffers.muzero_replay_buffer import MuZeroReplayBuffer, Game
from muzero.muzero_mcts import ChanceNode, DecisionNode
from muzero.muzero_network import Network
import datetime

from utils import (
    get_legal_moves,
)

from replay_buffers.utils import update_per_beta

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
            num_actions=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            # TODO: sort out when to do channel first and channel last
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        ).share_memory()

        self.target_model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
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
            lstm_horizon_len=self.config.lstm_horizon_len,
            value_prefix=self.config.value_prefix,
            tau=self.config.reanalyze_tau,
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
                "cons_loss",
                "q_loss",
                "sigma_loss",
                "vqvae_commitment_cost",
                "loss",
                "test_score",
                "episode_length",
                "num_codes",
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
        self.stats.add_plot_types("cons_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("q_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types(
            "sigma_loss", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "vqvae_commitment_cost", PlotType.ROLLING_AVG, rolling_window=100
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
        # from wrappers import record_video_wrapper

        # worker_env.render_mode = "rgb_array"
        # worker_env = record_video_wrapper(
        #     worker_env,
        #     f"./videos/{self.model_name}/{worker_id}",
        #     self.checkpoint_interval,
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
                    (
                        value_loss,
                        policy_loss,
                        reward_loss,
                        to_play_loss,
                        cons_loss,
                        q_loss,
                        sigma_loss,
                        vqvae_commitment_cost,
                        loss,
                    ) = self.learn()
                    self.stats.append("value_loss", value_loss)
                    self.stats.append("policy_loss", policy_loss)
                    self.stats.append("reward_loss", reward_loss)
                    self.stats.append("to_play_loss", to_play_loss)
                    self.stats.append("cons_loss", cons_loss)
                    self.stats.append("q_loss", q_loss)
                    self.stats.append("sigma_loss", sigma_loss)
                    self.stats.append("vqvae_commitment_cost", vqvae_commitment_cost)
                    self.stats.append("loss", loss)
                    print("learned")
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
        root = DecisionNode(0.0)
        ChanceNode.estimation_method = self.config.q_estimation_method
        ChanceNode.discount = self.config.discount_factor
        ChanceNode.value_prefix = self.config.value_prefix
        DecisionNode.estimation_method = self.config.q_estimation_method
        DecisionNode.discount = self.config.discount_factor
        DecisionNode.value_prefix = self.config.value_prefix
        DecisionNode.pb_c_init = self.config.pb_c_init
        DecisionNode.pb_c_base = self.config.pb_c_base
        DecisionNode.gumbel = self.config.gumbel
        DecisionNode.cvisit = self.config.gumbel_cvisit
        DecisionNode.cscale = self.config.gumbel_cscale
        DecisionNode.stochastic = self.config.stochastic

        v_pi_raw, policy, hidden_state = self.predict_initial_inference(
            state,
            model=inference_model,
        )
        policy = policy[0]
        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        if self.config.support_range is not None:
            # support_to_scalar expects a support vector and returns a scalar tensor
            v_pi = support_to_scalar(v_pi_raw, self.config.support_range)
            # ensure it's a Python float where you use .item() later
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(v_pi_raw.item())

        # if self.config.game.num_players != 1:
        #     legal_moves = get_legal_moves(info)[0]  # [0]
        #     # print(legal_moves)
        # else:
        #     legal_moves = list(range(self.num_actions))
        #     to_play = 0
        legal_moves = get_legal_moves(info)[0]  # [0]
        if legal_moves is None:
            legal_moves = list(range(self.num_actions))
            to_play = 0

        # print("traj_action", trajectory_action)
        # print("legal moves", legal_moves)
        root.visits += 1

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
                assert trajectory_action in actions
                if trajectory_action not in sampled_actions:
                    sampled_actions += [trajectory_action]
                    top_idx = torch.concat(
                        (top_idx, torch.tensor([actions.index(trajectory_action)]))
                    )

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
                sampled_actions,
                to_play,
                policy,
                hidden_state,
                0.0,
                value=v_pi_scalar,
                reward_h_state=reward_h_state,
                reward_c_state=reward_c_state,
                is_reset=True,
            )

            # attach the root_score to the created children
            for a in sampled_actions:
                root.children[a].root_score = sampled_g_values[a]  # store Gumbel+logit
        else:
            # TODO: add sample muzero for complex action spaces (right now just use gumbel)
            # print("policy", policy)
            root.expand(
                legal_moves,
                to_play,
                policy,
                hidden_state,
                0.0,
                reward_h_state=reward_h_state,
                reward_c_state=reward_c_state,
                is_reset=True,
            )
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
                root,
                min_max_stats,
                list(root.children.keys()),
                inference_model=inference_model,
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
            target_policy = root.get_gumbel_improved_policy(min_max_stats).to(
                self.device
            )

            return (
                root.value(),
                policy,
                target_policy,
                torch.tensor(best_action),
            )
        else:
            return (
                root.value(),
                policy,
                policy,
                torch.argmax(policy),
            )

    def sequential_halving(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        candidates: list,
        inference_model=None,
    ):
        """
        Perform Sequential Halving among `candidates` (list of action ints).
        It splits self.config.num_simulations across rounds that eliminate ~half of candidates each round.
        Survivors remain in root.children and accumulate visits/values as usual.
        """

        # Helper function for sorting scores by value (used for both priority and elimination)
        def sort_by_score(item):
            # item is a tuple (action, score)
            return item[1]

        m = len(candidates)

        # number of rounds to reduce to 1 (ceil log2)
        survivors = candidates.copy()
        scores = []
        for a in survivors:
            child = root.children[a]
            scores.append((a, root.get_gumbel_root_child_score(child, min_max_stats)))
        scores.sort(key=sort_by_score, reverse=True)
        survivors = [a for a, _ in scores]

        sims_used = 0
        while sims_used < self.config.num_simulations:
            if len(survivors) > 2:
                # TODO: should this be a min of 1 visit per thing per round?
                sims_this_round = max(
                    1,
                    math.floor(
                        self.config.num_simulations / (math.log2(m) * (len(survivors)))
                    ),
                ) * len(survivors)

            else:
                sims_this_round = self.config.num_simulations - sims_used

            if sims_used + sims_this_round > self.config.num_simulations:
                sims_this_round = self.config.num_simulations - sims_used
            # run sims_per_round simulations restricted to current survivors
            sims_used += sims_this_round
            for i in range(sims_this_round):
                # The modulo operation cycles through the sorted survivors list
                action = survivors[i % len(survivors)]

                # Run a single simulation, but ONLY allow the current `action` to be selected at the root
                self._run_single_simulation(
                    root,
                    min_max_stats,
                    inference_model=inference_model,
                    allowed_actions=[action],
                )  # recompute a score per survivor for elimination (use visits primarily)

            scores = []
            for a in survivors:
                child = root.children[a]
                scores.append(
                    (a, root.get_gumbel_root_child_score(child, min_max_stats))
                )
            scores.sort(key=sort_by_score, reverse=True)
            survivors = [a for a, _ in scores]
            # print("scores", scores)
            # print("survivors", survivors)

            num_to_eliminate = math.ceil(len(survivors) / 2.0)
            # leave 2 survivors
            if len(survivors) - num_to_eliminate < 2:
                # num_to_eliminate = len(survivors) - 2
                # print(num_to_eliminate)
                survivors = survivors[:2]
            else:
                survivors = survivors[:-num_to_eliminate]
            # eliminated = [a for a, _ in scores[:num_to_eliminate]]
            # survivors = [a for a in survivors if a not in eliminated]

            # print("survivors after elimination", survivors)
            # print(survivors, scores)

        # return survivors[0]
        final_scores = []
        for a in survivors:
            child = root.children[a]
            final_scores.append(
                (a, root.get_gumbel_root_child_score(child, min_max_stats))
            )

        final_scores.sort(key=sort_by_score, reverse=True)

        # Return the BEST action (Index 0), not the worst!
        return final_scores[0][0]

    def _run_single_simulation(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_model=None,
        allowed_actions=None,
    ):
        node = root
        search_path = [node]
        to_play = root.to_play
        horizon_index = 0
        # old_to_play = to_play
        # GO UNTIL A LEAF NODE IS REACHED
        # while node.expanded():
        #     action, node = node.select_child(
        #         min_max_stats=min_max_stats,
        #         allowed_actions=allowed_actions,
        #     )
        #     # old_to_play = (old_to_play + 1) % self.config.game.num_players
        #     search_path.append(node)
        #     horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
        # ---------------------------------------------------------------------
        # 1. SELECTION PHASE
        # ---------------------------------------------------------------------
        # We descend until we hit a leaf DecisionNode OR a ChanceNode that needs a new code.
        while True:
            if not node.expanded():
                break  # Reached a leaf state (DecisionNode)
            if isinstance(node, DecisionNode):
                # Decision -> Select Action -> ChanceNode
                action, node = node.select_child(
                    min_max_stats=min_max_stats,
                    allowed_actions=allowed_actions,
                )
                horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len

            elif isinstance(node, ChanceNode):
                # Chance -> Select Code -> DecisionNode

                code, node = node.select_child(
                    # TODO: Gumbel Top-K on chance nodes?
                )
            search_path.append(node)

        parent = search_path[-2]
        # if to_play != old_to_play and self.training_step > 1000:
        #     print("WRONG TO PLAY", onehot_to_play)
        if isinstance(node, DecisionNode):
            if isinstance(parent, DecisionNode):
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = self.predict_recurrent_inference(
                    parent.hidden_state,
                    torch.tensor(action).to(parent.hidden_state.device).unsqueeze(0),
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
            elif isinstance(parent, ChanceNode):
                # assert (
                #     node.value_prefix == False
                # ), "value prefix not implemented with chance nodes"
                # TODO: make value prefix work with chance nodes
                # print("code before recurrent inference", code.shape)
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = self.predict_recurrent_inference(
                    parent.afterstate,
                    code.to(parent.afterstate.device)
                    .unsqueeze(0)
                    .float(),  # a sampled code instead of an action
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
        elif isinstance(node, ChanceNode):
            # CASE B: Stochastic Expansion (The Core Change)
            # We are at (State, Action). We need to:
            # 1. Get Afterstate Value & Code Priors (Expand ChanceNode)
            # 2. Sample a Code
            # 3. Get Next State & Reward (Create DecisionNode)
            afterstate, value, code_priors = (
                self.predict_afterstate_recurrent_inference(  # <--- YOU NEED THIS METHOD
                    parent.hidden_state,
                    torch.tensor(action)
                    .to(parent.hidden_state.device)
                    .unsqueeze(0)
                    .float(),
                )
            )

            if self.config.support_range:
                value = support_to_scalar(value, self.config.support_range).item()
            else:
                value = value.item()

            # Expand the Chance Node with these priors
            node.expand(
                parent.to_play,
                afterstate,
                value,
                code_priors[0],
                reward_h_state=parent.reward_h_state,
                reward_c_state=parent.reward_c_state,
            )

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
            # print(totals[i])
            # print(acc[node_player])
            totals[i] = acc[node_player]

            node.value_sum += totals[i]
            node.visits += 1

            # Prepare acc for i-1 (if any)
            if i > 0:
                # reward at index i belongs to acting_player = search_path[i-1].to_play
                acting_player = search_path[i - 1].to_play
                if isinstance(search_path[i], DecisionNode):
                    r_i = search_path[i - 1].child_reward(search_path[i])

                    # # Update per-player accumulators in O(num_players)
                    # # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # # sign(p, i) = +1 if acting_player == p else -1
                    # # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(self.config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                elif isinstance(search_path[i], ChanceNode):
                    for p in range(self.config.game.num_players):
                        # sign = 1.0 if acting_player == p else -1.0
                        # acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                        # chance nodes can be thought to have 0 reward, and no discounting (as its like the roll after the action, or another way of thinking of it is that only on decision nodes do we discount expected reward, a chance node is not a decision point)
                        acc[p] = acc[p]
                child_q = search_path[i - 1].get_child_q_from_parent(search_path[i])
                min_max_stats.update(child_q)
            else:
                min_max_stats.update(search_path[i].value())

    def learn(self):
        samples = self.replay_buffer.sample()
        # --- 1. Unpack Data from New Buffer Structure ---
        observations = samples["observations"]

        # (B, unroll + 1)
        target_observations = samples["unroll_observations"].to(self.device)

        # (B, unroll + 1, num_actions)
        target_policies = samples["policy"].to(self.device)

        # (B, unroll + 1)
        target_values = samples["values"].to(self.device)

        # (B, unroll + 1)
        target_rewards = samples["rewards"].to(self.device)

        # (B, unroll) - Actions are only needed for the transitions
        actions = samples["actions"].to(self.device)

        # (B, unroll + 1, num_players)
        target_to_plays = samples["to_plays"].to(self.device)

        # (B, unroll + 1, 1) - Ground truth chance codes from environment (optional usage)
        target_chance_codes = samples["chance_codes"].to(self.device)

        # --- MASKS ---
        # valid_masks: (B, unroll + 1). True if state is in episode (inc. Terminal).
        # Use for: Value, Reward, Consistency.
        consistency_masks = samples["valid_masks"].to(self.device)

        # policy_masks: (B, unroll + 1). True if state is actionable (exc. Terminal).
        # Use for: Policy.
        policy_masks = samples["policy_masks"].to(self.device)

        weights = samples["weights"].to(self.device)
        inputs = self.preprocess(observations)

        (
            initial_values,
            initial_policies,
            hidden_states,
        ) = self.predict_initial_inference(inputs, model=self.model)
        # This list will capture predicted latent states ŝ_{t}, ŝ_{t+1}, ..., ŝ_{t+K}
        # `hidden_state` at this point is s_t (from initial inference)

        reward_h_states = torch.zeros(
            1, self.config.minibatch_size, self.config.lstm_hidden_size
        ).to(self.device)
        reward_c_states = torch.zeros(
            1, self.config.minibatch_size, self.config.lstm_hidden_size
        ).to(self.device)

        latent_states = [hidden_states]  # length will end up being unroll_steps + 1
        latent_afterstates = []
        latent_code_probabilities = []
        encoder_softmaxes = []
        encoder_onehots = []
        chance_values = []
        if self.config.support_range is not None:
            reward_shape = (
                self.config.minibatch_size,
                self.config.support_range * 2 + 1,
            )
        else:
            reward_shape = (self.config.minibatch_size, 1)

        values = [initial_values]
        rewards = [
            torch.zeros(reward_shape, device=self.device)
        ]  # R_t = 0 (Placeholder)
        policies = [initial_policies]
        to_plays = [
            torch.zeros(
                (self.config.minibatch_size, self.config.game.num_players),
                device=self.device,
            )
        ]
        gradient_scales = [1.0] + [
            1.0 / self.config.unroll_steps
        ] * self.config.unroll_steps

        for k in range(self.config.unroll_steps):
            actions_k = actions[:, k]
            target_observations_k = target_observations[:, k]
            target_observations_k_plus_1 = target_observations[:, k + 1]
            real_obs_k = self.preprocess(target_observations_k)
            real_obs_k_plus_1 = self.preprocess(target_observations_k_plus_1)
            encoder_input = torch.concat([real_obs_k, real_obs_k_plus_1], dim=1)
            if self.config.stochastic:
                afterstates, q_k, code_priors_k = (
                    self.predict_afterstate_recurrent_inference(  # <--- YOU NEED THIS METHOD
                        hidden_states, actions_k, model=self.model
                    )
                )

                encoder_softmax_k, encoder_onehot_k = self.model.encoder(encoder_input)
                if self.config.use_true_chance_codes:
                    codes_k = F.one_hot(
                        target_chance_codes[:, k + 1].squeeze(-1).long(),
                        self.config.num_chance,
                    )
                    assert (
                        codes_k.shape == encoder_onehot_k.shape
                    ), f"{codes_k.shape} == {encoder_onehot_k.shape}"
                    encoder_onehot_k = codes_k.float()

                latent_afterstates.append(afterstates)
                latent_code_probabilities.append(code_priors_k)
                chance_values.append(q_k)

                # σ^k is trained towards the one hot chance code c_t+k+1 = one hot (argmax_i(e((o^i)≤t+k+1))) produced by the encoder.
                encoder_onehots.append(encoder_onehot_k)
                encoder_softmaxes.append(encoder_softmax_k)
                # print("encoder onehot shape", encoder_onehot_k.shape)
                # print("afterstates shape", afterstates.shape)

                (
                    rewards_k,
                    hidden_states,
                    values_k,
                    policies_k,
                    to_plays_k,
                    reward_h_states,
                    reward_c_states,
                ) = self.predict_recurrent_inference(
                    afterstates,
                    encoder_onehot_k,
                    reward_h_states,
                    reward_c_states,
                    model=self.model,
                )

            else:
                # WARNING ACTIONS_K COULD BE NAN
                (
                    rewards_k,
                    hidden_states,
                    values_k,
                    policies_k,
                    to_plays_k,
                    reward_h_states,
                    reward_c_states,
                ) = self.predict_recurrent_inference(
                    hidden_states,
                    actions_k,
                    reward_h_states,
                    reward_c_states,
                    model=self.model,
                )

            latent_states.append(hidden_states)
            # Store the predicted states and outputs
            values.append(values_k)
            rewards.append(rewards_k)
            policies.append(policies_k)
            to_plays.append(to_plays_k)

            # Scale the gradient of the hidden state (applies to the whole batch)
            # Append the predicted latent (ŝ_{t+k+1}) BEFORE scaling for the next step
            hidden_states = scale_gradient(hidden_states, 0.5)

            # reset hidden states
            if self.config.value_prefix and (k + 1) % self.config.lstm_horizon_len == 0:
                # CHECK THAT THESE ARE CORRECT!!!
                reward_h_states = torch.zeros_like(reward_h_states).to(self.device)
                reward_c_states = torch.zeros_like(reward_c_states).to(self.device)

        # Stack the results into (K+1, B, ...) tensors, then permute to (B, K+1, ...)
        values_tensor = torch.stack(values).permute(
            1, 0, *range(2, len(values[0].shape) + 1)
        )
        rewards_tensor = torch.stack(rewards).permute(
            1, 0, *range(2, len(rewards[0].shape) + 1)
        )
        policies_tensor = torch.stack(policies).permute(
            1, 0, *range(2, len(policies[0].shape) + 1)
        )
        to_plays_tensor = torch.stack(to_plays).permute(
            1, 0, *range(2, len(to_plays[0].shape) + 1)
        )
        latent_states_tensor = torch.stack(latent_states).permute(
            1, 0, *range(2, len(latent_states[0].shape) + 1)
        )

        if self.config.stochastic:
            latent_afterstates_tensor = torch.stack(latent_afterstates).permute(
                1, 0, *range(2, len(latent_afterstates[0].shape) + 1)
            )
            latent_code_probabilities_tensor = torch.stack(
                latent_code_probabilities
            ).permute(1, 0, *range(2, len(latent_code_probabilities[0].shape) + 1))
            encoder_onehots_tensor = torch.stack(encoder_onehots).permute(
                1, 0, *range(2, len(encoder_onehots[0].shape) + 1)
            )
            encoder_softmaxes_tensor = torch.stack(encoder_softmaxes).permute(
                1, 0, *range(2, len(encoder_softmaxes[0].shape) + 1)
            )
            chance_values_tensor = torch.stack(chance_values).permute(
                1, 0, *range(2, len(chance_values[0].shape) + 1)
            )

        gradient_scales_tensor = torch.tensor(
            gradient_scales, device=self.device
        ).reshape(
            1, -1
        )  # (1, K+1)

        for training_iteration in range(self.config.training_iterations):
            total_loss = torch.tensor(0.0, device=self.device)
            val_loss_acc = torch.tensor(0.0, device=self.device)
            pol_loss_acc = torch.tensor(0.0, device=self.device)
            rew_loss_acc = torch.tensor(0.0, device=self.device)
            tp_loss_acc = torch.tensor(0.0, device=self.device)
            cons_loss_acc = torch.tensor(0.0, device=self.device)

            q_loss_acc = torch.tensor(0.0, device=self.device)
            sigma_loss_acc = torch.tensor(0.0, device=self.device)
            vqvae_commitment_cost_acc = torch.tensor(0.0, device=self.device)

            # Initialize priorities (only needed for k=0, shape (B,))
            priorities = torch.zeros(self.config.minibatch_size, device=self.device)
            for k in range(self.config.unroll_steps + 1):
                target_values_k = target_values[:, k]  # (B, ...)
                target_rewards_k = target_rewards[:, k]  # (B, ...)
                target_policies_k = target_policies[:, k]  # (B, policy_dim)
                target_to_plays_k = target_to_plays[:, k]  # (B, num_players)
                consistency_mask_k = consistency_masks[:, k]  # For Val, Rew, Cons
                policy_mask_k = policy_masks[:, k]  # For Policy ONLY

                values_k = values_tensor[:, k]  # (B, ...)
                rewards_k = rewards_tensor[:, k]  # (B, ...)
                policies_k = policies_tensor[:, k]  # (B, policy_dim)
                to_plays_k = to_plays_tensor[:, k]  # (B, num_players)
                latent_states_k = latent_states_tensor[:, k]  # (B, hidden_dim)

                scales_k = gradient_scales_tensor[:, k]  # (1,)

                # --- 1. Priority Update (Only for k=0) ---
                if k == 0:
                    if self.config.support_range is not None:
                        # Convert predicted value support to scalar for priority calculation
                        pred_scalar = support_to_scalar(
                            values_k, self.config.support_range
                        )
                        assert pred_scalar.shape == target_values_k.shape
                        priority = (
                            torch.abs(target_values_k - pred_scalar)
                            + self.config.per_epsilon
                        )
                    else:
                        priority = (
                            torch.abs(target_values_k - values_k.squeeze(-1))
                            + self.config.per_epsilon
                        )
                    priorities = priority.detach()  # Keep the B-length tensor

                # --- 2. Convert to Support if needed ---
                if self.config.support_range is not None:
                    target_values_k = scalar_to_support(
                        target_values_k, self.config.support_range
                    ).to(self.device)
                    target_rewards_k = scalar_to_support(
                        target_rewards_k, self.config.support_range
                    ).to(self.device)

                # --- 3. Loss Calculations ---

                # FIX: Squeeze the predicted values_k to ensure shape consistency with target_values_k
                # If using support, values_k is (B, support_dim) and target_values_k is (B, support_dim) - NO SQUEEZE.
                # If NOT using support, values_k is (B, 1) and target_values_k is (B,) - SQUEEZE IS NEEDED.
                if self.config.support_range is None:
                    predicted_values_k = values_k.squeeze(-1)  # Convert (B, 1) -> (B,)
                    predicted_rewards_k = rewards_k.squeeze(
                        -1
                    )  # Convert (B, 1) -> (B,)

                    # Check for unexpected dimensionality now that we've squeezed
                else:
                    # When using support, both are (B, support_dim), so no squeeze needed for loss
                    predicted_values_k = values_k
                    predicted_rewards_k = rewards_k
                assert (
                    predicted_values_k.shape == target_values_k.shape
                ), f"{predicted_values_k.shape} = {target_values_k.shape}"
                assert (
                    predicted_rewards_k.shape == target_rewards_k.shape
                ), f"{predicted_rewards_k.shape} = {target_rewards_k.shape}"

                # Value Loss: (B,)
                value_loss_k = (
                    self.config.value_loss_factor
                    * self.config.value_loss_function(
                        predicted_values_k, target_values_k
                    )
                )
                # Policy Loss: (B,)
                policy_loss_k = self.config.policy_loss_function(
                    policies_k, target_policies_k
                )

                q_loss_k = torch.zeros_like(value_loss_k)
                sigma_loss_k = torch.zeros_like(value_loss_k)
                vqvae_commitment_cost_k = torch.zeros_like(value_loss_k)

                # Reward, To-Play, and Consistency Losses (Only for k > 0)
                reward_loss_k = torch.zeros_like(value_loss_k)
                to_play_loss_k = torch.zeros_like(value_loss_k)
                consistency_loss_k = torch.zeros_like(value_loss_k)

                if k > 0:
                    # Reward Loss: (B,)
                    reward_loss_k = self.config.reward_loss_function(
                        predicted_rewards_k, target_rewards_k
                    )

                    # To-Play Loss: (B,)
                    if self.config.game.num_players != 1:
                        to_play_loss_k = (
                            self.config.to_play_loss_factor
                            * self.config.to_play_loss_function(
                                to_plays_k, target_to_plays_k
                            )
                        )

                    # Consistency Loss (EfficientZero): (B,)
                    target_observations_k = target_observations[:, k]  # (B, C, H, W)
                    # TODO: CONSISTENCY LOSS FOR AFTERSTATES
                    # 1. Encode the real future observation (Target)
                    # We need to preprocess the raw target observation
                    real_obs = self.preprocess(target_observations_k)
                    # Use predict_initial_inference to get the latent state (s'_t+k)
                    # It uses the representation network and the projection network
                    # We pass model=self.model to ensure it uses the correct network instance
                    # We use a detached real_latent for the consistency target
                    # Note: We can reuse the `predict_initial_inference` function if it only calls the representation/project models.

                    # with torch.no_grad():
                    # real_latent = self.model.representation(real_obs)
                    _, _, real_latents = self.predict_initial_inference(
                        real_obs, model=self.model
                    )
                    # may be unecessary but better safe than sorry
                    real_latents = real_latents.detach()
                    # Project the target to the comparison space
                    proj_targets = self.model.project(real_latents, grad=False)
                    f1 = F.normalize(proj_targets, p=2.0, dim=-1, eps=1e-5)

                    # 2. Process the predicted latent (Prediction)
                    # We project, then predict (SimSiam style predictor head)
                    # latent_states_tensor[k] is the output of the dynamics function
                    proj_preds = self.model.project(latent_states_k, grad=True)
                    f2 = F.normalize(proj_preds, p=2.0, dim=-1, eps=1e-5)

                    # 3. Calculate Negative Cosine Similarity: (B,)
                    current_consistency = -(f1 * f2).sum(dim=1)
                    consistency_loss_k = (
                        self.config.consistency_loss_factor * current_consistency
                    )

                    if self.config.stochastic:
                        afterstates_k = latent_afterstates_tensor[:, k - 1]
                        latent_code_probabilities_k = latent_code_probabilities_tensor[
                            :, k - 1
                        ]
                        encoder_onehot_k_plus_1 = encoder_onehots_tensor[
                            :, k - 1
                        ]  # the encoder code for the next observation (so k_plus_1)
                        encoder_softmax_k_plus_1 = encoder_softmaxes_tensor[
                            :, k - 1
                        ]  # the encoder embedding for the next observation
                        chance_values_k = chance_values_tensor[:, k - 1]

                        target_chance_values_k = target_values[:, k - 1]

                        target_chance_values_k = scalar_to_support(
                            target_chance_values_k, self.config.support_range
                        ).to(self.device)

                        # TODO: HAVE WE ALREADY RECOMPUTED TARGET Q IN THE CASE OF REANALYZE? I THINK WE HAVE

                        if self.config.support_range is None:
                            predicted_chance_values_k = chance_values_k.squeeze(
                                -1
                            )  # Convert (B, 1) -> (B,)
                        else:
                            # When using support, both are (B, support_dim), so no squeeze needed for loss
                            predicted_chance_values_k = chance_values_k
                        assert (
                            predicted_chance_values_k.shape
                            == target_chance_values_k.shape
                        ), f"{predicted_values_k.shape} = {target_chance_values_k.shape}"

                        q_loss_k = (
                            self.config.value_loss_factor
                            * self.config.value_loss_function(
                                predicted_chance_values_k,
                                target_chance_values_k,
                            )
                        )

                        # σ^k is trained towards the one hot chance code c_t+k+1 = one hot (argmax_i(e((o^i)≤t+k+1))) produced by the encoder.
                        sigma_loss_k = self.config.sigma_loss(
                            latent_code_probabilities_k,
                            encoder_onehot_k_plus_1.detach(),
                            # encoder_softmax_k_plus_1.detach(),  # or encoder_onehot_k_plus_1.detach()
                        )

                        # VQ-VAE commitment cost between c_t+k+1 and (c^e)_t+k+1 ||c_t+k+1 - (c^e)_t+k+1||^2
                        if not self.config.use_true_chance_codes:
                            diff = (
                                encoder_softmax_k_plus_1
                                - encoder_onehot_k_plus_1.detach()
                            )
                            vqvae_commitment_cost_k = (
                                self.config.vqvae_commitment_cost_factor
                                * torch.sum(diff.pow(2), dim=-1)
                            )
                        # vqvae_commitment_cost_k = MSELoss()(
                        #     encoder_softmax_k_plus_1, encoder_onehot_k_plus_1.detach()
                        # )

                # --- 4. Apply Mask, Weights, and Gradient Scale ---
                if self.training_step % self.checkpoint_interval == 0:
                    # torch.set_printoptions(profile="full")

                    print("actions shape", actions.shape)
                    print("target value shape", target_values.shape)
                    print("predicted values shape", values_tensor.shape)
                    print("target rewards shape", target_rewards.shape)
                    print("predicted rewards shape", rewards_tensor.shape)
                    if self.config.stochastic:
                        print("target qs shape", target_values.shape)
                        print("predicted qs shape", chance_values_tensor.shape)
                    print("target to plays shape", target_to_plays.shape)
                    print("predicted to_plays shape", to_plays_tensor.shape)
                    print("masks shape", policy_masks.shape, consistency_masks.shape)

                    print("actions", actions)
                    print("target value", target_values)
                    print("predicted values", values)
                    print("target rewards", target_rewards)
                    print("predicted rewards", rewards)
                    if self.config.stochastic:
                        print("target qs", target_values)
                        print("predicted qs", chance_values)
                    print("target to plays", target_to_plays)
                    print("predicted to_plays", to_plays)

                    if self.config.stochastic:
                        print("encoder embedding", encoder_softmaxes)
                        print("encoder onehot", encoder_onehots)
                        print("predicted sigmas", latent_code_probabilities)
                    print("masks", policy_masks, consistency_masks)
                    # torch.set_printoptions(profile="default")

                assert (
                    value_loss_k.shape
                    == reward_loss_k.shape
                    == policy_loss_k.shape
                    == to_play_loss_k.shape
                    == consistency_loss_k.shape
                    == q_loss_k.shape
                    == sigma_loss_k.shape
                    == vqvae_commitment_cost_k.shape
                ), f"{value_loss_k.shape} == {reward_loss_k.shape} == {policy_loss_k.shape} == {to_play_loss_k.shape} == {consistency_loss_k.shape} == {q_loss_k.shape} == {sigma_loss_k.shape} == {vqvae_commitment_cost_k.shape}"

                # Apply mask: (B,)
                # --- D. Apply Masks ---
                if self.config.mask_absorbing:
                    # Apply Consistency Mask to: Value, Reward, Q, Sigma, VQ, Consistency
                    value_loss_k *= consistency_mask_k
                    reward_loss_k *= consistency_mask_k
                    q_loss_k *= consistency_mask_k
                    # Consistency Loss Masking
                    # We can't calculate consistency if the current node is absorbing
                    # OR if it's the very last unrolled step (because we might not have a target obs for k+1 if we hit limit)
                    # But policy_mask_k handles the absorbing part.
                    # TODO: what should this use? consistency_mask_k if training initial on terminal, else policy?
                    consistency_loss_k *= consistency_mask_k  # can do consistency on terminal observation -> absorbing to have value 0

                # To Play Loss masking
                to_play_loss_k *= consistency_mask_k  # important to correctly predict whos turn it is on a terminal state, but unimportant afterwards
                vqvae_commitment_cost_k *= (
                    policy_mask_k  # no chance nodes from terminal -> absorbing
                )
                sigma_loss_k *= (
                    policy_mask_k  ## no chance nodes from terminal -> absorbing
                )

                # IMPORTANT: Policy Loss uses Policy Mask (excludes terminal)
                policy_loss_k *= policy_mask_k

                # Sum the losses for this step: (B,)
                step_loss = (
                    value_loss_k
                    + reward_loss_k
                    + policy_loss_k
                    + to_play_loss_k
                    + consistency_loss_k
                    + q_loss_k
                    + sigma_loss_k
                    + vqvae_commitment_cost_k
                )

                # Scale the gradient with the unroll scale (scalar for the whole batch)
                scaled_loss_k = scale_gradient(step_loss, scales_k.item())

                # Apply PER weights (weights are (B,), scale_k is (1,))
                weighted_scaled_loss_k = scaled_loss_k * weights
                # Accumulate the total loss (scalar)
                total_loss += weighted_scaled_loss_k.sum()

                # Accumulate unweighted losses for logging/return (scalar)
                val_loss_acc += value_loss_k.sum().item()
                rew_loss_acc += reward_loss_k.sum().item()
                pol_loss_acc += policy_loss_k.sum().item()
                tp_loss_acc += to_play_loss_k.sum().item()
                cons_loss_acc += consistency_loss_k.sum().item()
                q_loss_acc += q_loss_k.sum().item()
                sigma_loss_acc += sigma_loss_k.sum().item()
                vqvae_commitment_cost_acc += vqvae_commitment_cost_k.sum().item()

            # --- Backpropagation and Optimization ---

            # The total loss is already summed across all steps and items
            # Divide by batch_size * (K+1) steps for the final average loss (optional, but good for scaling)
            # The MuZero paper often divides by batch_size * unroll_steps, here we'll just divide by batch size
            # as the losses are already weighted by the gradient scale 1/K for recurrent steps.
            loss_mean = total_loss / self.config.minibatch_size

            self.optimizer.zero_grad()
            loss_mean.backward()
            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

            self.optimizer.step()

            # --- Update Priorities ---
            # priorities tensor is already of shape (B,) from k=0
            self.replay_buffer.update_priorities(
                samples["indices"], priorities.cpu().numpy(), ids=samples["ids"]
            )

            # STAT TRACKING:
            if self.config.stochastic:
                codes = encoder_onehots_tensor.argmax(
                    dim=-1
                )  # shape (B, K), dtype long
                # --- (A) Total unique codes across entire batch+time ---
                unique_codes_all = torch.unique(
                    codes
                )  # 1D tensor with sorted unique indices
                num_unique_all = unique_codes_all.numel()
                # Optionally: convert to Python int
                num_unique_all_int = int(num_unique_all)
                self.stats.append("num_codes", num_unique_all_int)

        # Convert accumulated sums to average loss per item/step for return values
        return (
            val_loss_acc / self.config.minibatch_size,
            pol_loss_acc / self.config.minibatch_size,
            rew_loss_acc / self.config.minibatch_size,
            tp_loss_acc / self.config.minibatch_size,
            cons_loss_acc / self.config.minibatch_size,
            q_loss_acc / self.config.minibatch_size,
            sigma_loss_acc / self.config.minibatch_size,
            vqvae_commitment_cost_acc / self.config.minibatch_size,
            loss_mean.item(),
        )

    def predict_initial_inference(
        self,
        states,
        model,
    ):
        if model == None:
            model = self.model
        state_inputs = self.preprocess(states)
        # print("state input shape", state_input.shape)
        values, policies, hidden_states = model.initial_inference(state_inputs)
        # should we action mask the priors?
        # legal_moves = get_legal_moves(info)
        # policy = action_mask(policy, legal_moves, device=self.device)
        # policy = policy / torch.sum(policy)  # Normalize policy
        # print("policy shape", policy.shape)
        return values, policies, hidden_states

    def predict_recurrent_inference(
        self,
        states,
        actions_or_codes,
        reward_h_states=None,
        reward_c_states=None,
        model=None,
    ):
        if model == None:
            model = self.model
        rewards, states, values, policies, to_play, reward_hidden = (
            model.recurrent_inference(
                states,
                actions_or_codes,
                reward_h_states,
                reward_c_states,
            )
        )

        # print(reward_hidden)
        reward_h_states = reward_hidden[0]
        reward_c_states = reward_hidden[1]
        # print(reward_h_states)
        # print(reward_c_states)

        return (
            rewards,
            states,
            values,
            policies,
            to_play,
            reward_h_states,
            reward_c_states,
        )

    def predict_afterstate_recurrent_inference(
        self, hidden_states, actions, model=None
    ):
        if model == None:
            model = self.model
        afterstates, value, chance_probs = model.afterstate_recurrent_inference(
            hidden_states,
            actions,
        )

        return afterstates, value, chance_probs

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
            self.replay_buffer.store(game, self.training_step)
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
            if self.config.game.has_legal_moves:
                info = {
                    "legal_moves": torch.nonzero(mask).view(-1).tolist(),
                    "player": to_play,
                }
            else:
                info = {
                    "player": to_play,
                }

            assert not (
                self.config.game.has_legal_moves and len(info["legal_moves"]) == 0
            ), f"no legal moves, invalid sample {info}"
            infos.append(info)
            # print("info with legal moves from nonzero mask", info)
            # ADD INJECTING SEEN ACTION THING FROM MUZERO UNPLUGGED
            if self.config.reanalyze_method == "mcts":
                root_value, _, new_policy, best_action = self.monte_carlo_tree_search(
                    obs,
                    info,  # FOR LEGAL MOVES
                    to_play,
                    int(traj_action.item()),
                    inference_model=inference_model,
                )

                new_root_value = float(root_value)
            else:
                value, new_policy, _ = self.predict_initial_inference(
                    obs, model=inference_model
                )
                new_root_value = value.item()

            # decide value target per your config (paper default: keep stored n-step TD for Atari)
            stored_n_step_value = float(
                self.replay_buffer.n_step_values_buffer[idx][0].item()
            )

            new_policies.append(new_policy[0])
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
            indices,
            new_policies,
            new_root_values,
            rewards,
            traj_actions,
            infos,
            ids,
            self.training_step,
            self.config.training_steps,
        )
        if self.config.reanalyze_update_priorities:
            self.replay_buffer.update_priorities(
                indices, new_priorities, ids=np.array(ids)
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stop_flag"] = state["stop_flag"].value
        del state["env"]
        del state["test_env"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stop_flag = mp.Value("i", state["stop_flag"])
        self.env = self.config.game.make_env()
        self.test_env = self.config.game.make_env(render_mode="rgb_array")
