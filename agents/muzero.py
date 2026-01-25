import copy
import datetime
import random
import sys

from replay_buffers.buffer_factories import create_muzero_buffer
from replay_buffers.game import Game, TimeStep
from search.search_factories import create_mcts
from search.nodes import DecisionNode, ChanceNode


sys.path.append("../")
from time import time
import traceback
from modules.utils import scalar_to_support, support_to_scalar, get_lr_scheduler
import numpy as np
from stats.stats import PlotType, StatTracker
from losses.losses import create_muzero_loss_pipeline
from utils.vector_aec_env import VectorAECEnv

from agents.agent import MARLBaseAgent
from agent_configs.muzero_config import MuZeroConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.agent_nets.muzero import Network
import datetime

from replay_buffers.utils import update_per_beta

# Set quantization engine globally for stable multiprocessing with quantized models
try:
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
except:
    pass

from modules.utils import scale_gradient

from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.ao.quantization
from torch.nn import quantized



def make_worker_env(game_config, worker_id, env_id, model_name, checkpoint_interval):
    env = game_config.make_env()
    if env_id == 0:
        try:
            from wrappers import record_video_wrapper
            env.render_mode = "rgb_array"
            env = record_video_wrapper(
                env,
                f"./videos/{model_name}/{worker_id}_{env_id}",
                checkpoint_interval,
            )
        except Exception as e:
            print(f"[Worker {worker_id}] Could not record video: {e}")
    return env

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
        loss_pipeline=None,
    ):
        super(MuZeroAgent, self).__init__(
            env,
            config,
            name,
            test_agents=test_agents,
            device=device,
            from_checkpoint=from_checkpoint,
        )
        self.env.reset()

        self.model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )

        self.target_model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        if self.config.quantize and not self.config.multi_process:
            self.target_model.to('cpu')
            self.target_model = torch.ao.quantization.quantize_dynamic(
                self.target_model,
                {nn.Linear},
                dtype=torch.qint8
            )
            self.target_model.eval()
            for p in self.target_model.parameters():
                p.requires_grad = False

        if self.config.multi_process:
            self.model.share_memory()
            self.target_model.share_memory()
        else:
            self.target_model.to(device)
            self.model.to(device)

        if self.config.compile and not self.config.multi_process:
            print("Compiling models...")
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
            self.target_model = torch.compile(self.target_model, mode=self.config.compile_mode)

        if loss_pipeline is None:
            self.loss_pipeline = create_muzero_loss_pipeline(
                config=self.config,
                device=self.device,
                predict_initial_inference_fn=self.predict_initial_inference,
                preprocess_fn=self.preprocess,
                model=self.model,
            )
        else:
            self.loss_pipeline = loss_pipeline

        self.search = create_mcts(config, self.device, self.num_actions)

        self.replay_buffer = create_muzero_buffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            num_players=self.config.game.num_players,
            unroll_steps=self.config.unroll_steps,
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
            batch_size=self.config.minibatch_size,
            observation_dtype=self.observation_dtype,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            epsilon=self.config.per_epsilon,
            use_batch_weights=self.config.per_use_batch_weights,
            use_initial_max_priority=self.config.per_use_initial_max_priority,
            lstm_horizon_len=self.config.lstm_horizon_len,
            value_prefix=self.config.value_prefix,
            tau=self.config.reanalyze_tau,
        )

        if self.config.optimizer == Adam:
            self.optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            self.optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
             self.optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)

        if self.config.use_mixed_precision:
            self.scaler = torch.amp.GradScaler(device=self.device.type)

        self._setup_stats()
        self.stop_flag = mp.Value("i", 0)

    def _setup_stats(self):
        """Initializes or updates the stat tracker with all required keys and plot types."""
        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]
        
        # EnsureStatTracker exists (might have been deleted in __getstate__)
        if not hasattr(self, "stats") or self.stats is None:
             self.stats = StatTracker(model_name=self.model_name)

        # 1. Initialize Keys (if not already present)
        stat_keys = [
            "score", "policy_loss", "value_loss", "reward_loss", "to_play_loss",
            "cons_loss", "loss", "test_score", "episode_length", "policy_entropy", 
            "value_diff", "policy_improvement", "root_children_values"
        ] + test_score_keys
        
        if self.config.stochastic:
            stat_keys += ["num_codes", "chance_probs", "chance_entropy", "q_loss", "sigma_loss", "vqvae_commitment_cost"]

        target_values = {
            "score": (self.env.spec.reward_threshold if hasattr(self.env, "spec") and self.env.spec.reward_threshold else None),
            "test_score": (self.env.spec.reward_threshold if hasattr(self.env, "spec") and self.env.spec.reward_threshold else None),
            "num_codes": 1 if self.config.game.is_deterministic else None,
        }
        
        use_tensor_dicts = {
            "test_score": ["score", "max_score", "min_score"],
            "policy_improvement": ["network", "search"],
            **{key: ["score"] + ["player_{}_score".format(p) for p in range(self.config.game.num_players)] + ["player_{}_win%".format(p) for p in range(self.config.game.num_players)] for key in test_score_keys},
        }

        # For host StatTracker, initialize keys that don't exist
        if not self.stats._is_client:
            for key in stat_keys:
                if key not in self.stats.stats:
                    self.stats._init_key(key, target_value=target_values.get(key), subkeys=use_tensor_dicts.get(key))

        # 2. Add/Refresh Plot Types
        self.stats.add_plot_types("score", PlotType.ROLLING_AVG, PlotType.BEST_FIT_LINE, rolling_window=100, ema_beta=0.6)
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("policy_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("value_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("reward_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("to_play_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("cons_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("num_codes", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("q_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("sigma_loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("vqvae_commitment_cost", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("episode_length", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("policy_entropy", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("value_diff", PlotType.ROLLING_AVG, rolling_window=100)
        self.stats.add_plot_types("policy_improvement", PlotType.BAR, bar_threshold=0.01, max_bars=20)
        self.stats.add_plot_types("root_children_values", PlotType.BAR, bar_threshold=1e-8, max_bars=30)
        
        if self.config.stochastic:
            self.stats.add_plot_types("chance_probs", PlotType.BAR, show_all_bars=True)
            self.stats.add_plot_types("chance_entropy", PlotType.ROLLING_AVG, rolling_window=100)

    def worker_fn(
        self, worker_id, stop_flag, stats_client: StatTracker, error_queue: mp.Queue
    ):
        self.stats = stats_client
        print(f"[Worker {worker_id}] Starting self-play with {self.config.num_envs_per_worker} vector envs...")
        
        import functools
        env_fns = [
            functools.partial(
                make_worker_env,
                self.config.game,
                worker_id,
                i,
                self.model_name,
                self.checkpoint_interval
            )
            for i in range(self.config.num_envs_per_worker)
        ]
        
        worker_env = VectorAECEnv(env_fns, auto_reset=True)
        
        # Ensure quantization engine is set in the child process
        try:
            if 'fbgemm' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'fbgemm'
            elif 'qnnpack' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'qnnpack'
        except:
            pass

        inference_model = self.target_model
        if self.config.quantize:
            print(f"[Worker {worker_id}] Quantizing model locally...")
            inference_model = torch.ao.quantization.quantize_dynamic(
                self.target_model,
                {nn.Linear},
                dtype=torch.qint8
            )
        
        if self.config.compile:
            print(f"[Worker {worker_id}] Compiling inference model...")
            inference_model = torch.compile(inference_model, mode=self.config.compile_mode)

        inference_model.to(self.device)
        inference_model.eval()

        try:
            self.play_game_vec(worker_env, inference_model=inference_model, stop_flag=stop_flag)
        except Exception as e:
            error_queue.put((e, traceback.format_exc()))
            raise
        finally:
            worker_env.close()

    def train(self):
        self._setup_stats()
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
                    print("learning")
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
                self.training_step += 1

                self.replay_buffer.set_beta(
                    update_per_beta(
                        self.replay_buffer.beta,
                        self.config.per_beta_final,
                        self.config.training_steps,
                        self.config.per_beta,
                    )
                )

                if self.training_step % self.config.transfer_interval == 0:
                    self.update_target_model()

            if self.training_step % self.test_interval == 0 and self.training_step > 0:
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

    def learn(self):
        samples = self.replay_buffer.sample()

        # --- 1. Unpack Data from New Buffer Structure ---
        observations = samples["observations"]

        # (B, unroll + 1)
        target_observations = samples["unroll_observations"].to(self.device)

        # (B, unroll + 1, num_actions)
        target_policies = samples["policies"].to(self.device)

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
        dones = samples["dones"].to(self.device)

        # --- MASKS ---
        # 1. Action Mask: Valid if not done (terminal states have no valid policy)
        has_valid_action_mask = ~dones

        # 2. Obs Mask: Valid if previous step was not done (if done, next obs is invalid/next game)
        # Shift dones right: If t-1 was done, t is invalid.
        shifted_dones = torch.roll(dones, 1, dims=1)
        shifted_dones[:, 0] = False
        has_valid_obs_mask = ~shifted_dones



        weights = samples["weights"].to(self.device)
        inputs = self.preprocess(observations)

        # --- 2. Initial Inference ---
        with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
            (
                initial_values,
                initial_policies,
                hidden_states,
            ) = self.predict_initial_inference(inputs, model=self.model)

            # This list will capture predicted latent states ŝ_{t}, ŝ_{t+1}, ..., ŝ_{t+K}
            # `hidden_state` at this point is s_t (from initial inference)

            reward_h_states = torch.zeros(
                1, self.config.minibatch_size, self.config.lstm_hidden_size
            ).to(self.device)
            reward_c_states = torch.zeros(
                1, self.config.minibatch_size, self.config.lstm_hidden_size
            ).to(self.device)

            gradient_scales = [1.0] + [
                1.0 / self.config.unroll_steps
            ] * self.config.unroll_steps

            network_output_sequences = self.model.world_model.unroll_sequence(
                agent=self,
                initial_hidden_state=hidden_states,
                initial_values=initial_values,
                initial_policies=initial_policies,
                actions=actions,
                target_observations=target_observations,
                target_chance_codes=target_chance_codes,
                reward_h_states=reward_h_states,
                reward_c_states=reward_c_states,
                preprocess_fn=self.preprocess,
            )

            # --- 5. Stack Results into (B, K+1, ...) Tensors ---
            predictions_tensor = self._stack_predictions(network_output_sequences)

            targets_tensor = {
                "values": target_values,
                "rewards": target_rewards,
                "policies": target_policies,
                "to_plays": target_to_plays,
            }

            # Add stochastic targets (indexed at k-1)
            if self.config.stochastic:
                # ensure chance_values have k + 1 steps for indexing consistency, first index is invalid so should be 0
                targets_tensor["chance_values"] = torch.zeros_like(target_values)
                targets_tensor["chance_values"][:, 1:] = target_values[
                    :, :-1
                ]  # TODO: LightZero this is the value of the next state (ie offset by one step)
                targets_tensor["encoder_onehots"] = predictions_tensor["encoder_onehots"]

            gradient_scales_tensor = torch.tensor(
                gradient_scales, device=self.device
            ).reshape(
                1, -1
            )  # (1, K+1)

            # --- 6. Create Context for Loss Computation ---
            context = {
                "has_valid_obs_mask": has_valid_obs_mask,
                "has_valid_action_mask": has_valid_action_mask,
                "target_observations": target_observations,
            }

            # --- 7. Train for Multiple Iterations ---
            for training_iteration in range(self.config.training_iterations):
                # Run the modular loss pipeline
                loss_mean, loss_dict, priorities = self.loss_pipeline.run(
                    predictions_tensor=predictions_tensor,
                    targets_tensor=targets_tensor,
                    context=context,
                    weights=weights,
                    gradient_scales=gradient_scales_tensor,
                    config=self.config,
                    device=self.device,
                )

            # --- 8. Logging at Checkpoint ---
            if self.training_step % self.checkpoint_interval == 0:
                self._log_training_step(
                    actions,
                    target_values,
                    predictions_tensor["values"],
                    target_rewards,
                    predictions_tensor["rewards"],
                    target_to_plays,
                    predictions_tensor["to_plays"],
                    has_valid_action_mask,
                    has_valid_obs_mask,
                    targets_tensor,
                    predictions_tensor if self.config.stochastic else None,
                )

            # --- 9. Backpropagation and Optimization ---
            self.optimizer.zero_grad()
            
            if self.config.use_mixed_precision:
                self.scaler.scale(loss_mean).backward()
                if self.config.clipnorm > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.clipnorm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_mean.backward()
                if self.config.clipnorm > 0:
                    clip_grad_norm_(self.model.parameters(), self.config.clipnorm)
                self.optimizer.step()
            
            self.lr_scheduler.step()
            
            if self.device == "mps":
                torch.mps.empty_cache()

            # --- 10. Update Priorities ---
            # priorities tensor is already of shape (B,) from k=0
            self.update_replay_priorities(
                samples["indices"], priorities.cpu().numpy(), ids=samples["ids"]
            )

            # --- 11. STAT TRACKING ---
            if self.config.stochastic:
                self._track_stochastic_stats(
                    predictions_tensor["encoder_onehots"],
                    predictions_tensor["latent_code_probabilities"],
                )
            
            # Categorize latent space by action
            if self.training_step % self.config.latent_viz_interval == 0:
                self._track_latent_visualization(
                    predictions_tensor["latent_states"],
                    actions,
                )

        # --- 12. Return Losses for Logging ---
        return self._prepare_return_losses(loss_dict, loss_mean.item())

    def _track_latent_visualization(self, latent_states, actions):
        """Track latent space representations categorized by action."""
        # Use root states (s0) and the first action (a0)
        # latent_states: (B, K+1, ...)
        # actions: (B, K)
        s0 = latent_states[:, 0]
        a0 = actions[:, 0]
        
        self.stats.add_latent_visualization(
            "latent_root", 
            s0, 
            labels=a0, 
            method=self.config.latent_viz_method
        )


    def _stack_predictions(self, network_output_sequences):
        """Stack prediction lists into (B, K+1, ...) tensors."""
        predictions = {}

        for key, tensor_list in network_output_sequences.items():
            if len(tensor_list) == 0:
                continue
            # Stack the list of tensors: Result is (Time, Batch, ...)
            stacked = torch.stack(tensor_list)

            # Permute to (Batch, Time, ...): Swap dimension 0 and 1
            # This is equivalent to the previous .permute(1, 0, *range(2, ...))
            dims = list(range(stacked.ndim))
            dims[0], dims[1] = dims[1], dims[0]

            predictions[key] = stacked.permute(*dims)

        return predictions

    def _log_training_step(
        self,
        actions,
        target_values,
        predicted_values,
        target_rewards,
        predicted_rewards,
        target_to_plays,
        predicted_to_plays,
        has_valid_action_mask,
        has_valid_obs_mask,
        targets_tensor,
        stochastic_preds,
    ):
        """Log training step information at checkpoint intervals."""
        # torch.set_printoptions(profile="full")
        print(self.training_step)
        print("actions shape", actions.shape)
        print("target value shape", target_values.shape)
        print("predicted values shape", predicted_values.shape)
        print("target rewards shape", target_rewards.shape)
        print("predicted rewards shape", predicted_rewards.shape)
        if self.config.stochastic:
            print("target qs shape", target_values.shape)
            print("predicted qs shape", stochastic_preds["chance_values"].shape)
        print("target to plays shape", target_to_plays.shape)
        print("predicted to_plays shape", predicted_to_plays.shape)
        print("masks shape", has_valid_action_mask.shape, has_valid_obs_mask.shape)

        print("actions", actions)
        print("target value", target_values)
        print("predicted values", predicted_values)
        print("target rewards", target_rewards)
        print("predicted rewards", predicted_rewards)
        if self.config.stochastic:
            print("target qs", targets_tensor["chance_values"])
            print("predicted qs", stochastic_preds["chance_values"])
        print("target to plays", target_to_plays)
        print("predicted to_plays", predicted_to_plays)

        if self.config.stochastic:
            print("encoder embedding", stochastic_preds["encoder_softmaxes"])
            print("encoder onehot", stochastic_preds["encoder_onehots"])
            print("predicted sigmas", stochastic_preds["latent_code_probabilities"])
        print("masks", has_valid_action_mask, has_valid_obs_mask)
        # torch.set_printoptions(profile="default")

    def _track_stochastic_stats(self, encoder_onehots_tensor, latent_code_probs_tensor):
        """Track statistics for stochastic MuZero."""
        # Calculate validity mask from probs (sum > 0.001)
        # latent_code_probs_tensor: (B, K, NumCodes) or (B, K+1, NumCodes)
        if latent_code_probs_tensor.ndim == 3:
             prob_sums = latent_code_probs_tensor.sum(dim=-1) # (B, K)
             mask = prob_sums > 0.001
        else:
             mask = torch.ones_like(codes, dtype=torch.bool)

        codes = encoder_onehots_tensor.argmax(dim=-1)  # shape (B, K), dtype long
        
        # Filter codes using the mask
        if mask.shape == codes.shape:
             valid_codes = codes[mask]
        else:
             # Fallback if shapes mismatch (e.g. if K vs K+1 issue arises)
             valid_codes = codes.flatten()

        # --- (A) Total unique codes across entire batch+time ---
        unique_codes_all = torch.unique(valid_codes)  # 1D tensor with sorted unique indices
        num_unique_all = unique_codes_all.numel()
        # Optionally: convert to Python int
        num_unique_all_int = int(num_unique_all)
        self.stats.append("num_codes", num_unique_all_int)

        # Track chance probability statistics (mean over batch and time)
        latent_node_probs = latent_code_probs_tensor  # (B, K+1, NumCodes) or (B, K, NumCodes)
        
        # Calculate mean probabilities for each code across batch and unroll steps
        # Result shape: (NumCodes,)
        if latent_node_probs.ndim == 3:
             valid_probs = latent_node_probs[mask] # (N_valid, NumCodes)
             if valid_probs.shape[0] > 0:
                 mean_probs = valid_probs.mean(dim=0)
             else:
                 mean_probs = torch.zeros(latent_node_probs.shape[-1], device=latent_node_probs.device)
        else:
             # Fallback if dimensions are unexpected
             mean_probs = latent_node_probs.mean(dim=0)

        # Append to stats as a 2D tensor (1, NumCodes) so we can stack them
        self.stats.append("chance_probs", mean_probs.unsqueeze(0))

        # --- (C) Track Entropy of Chance Probabilities ---
        # latent_code_probs_tensor: (B, K, NumCodes)
        probs = latent_code_probs_tensor
        # Avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (B, K)
        
        if mask is not None:
             if entropy.shape == mask.shape:
                  entropy = entropy[mask]
             else:
                  # shape mismatch fallback
                  entropy = entropy.flatten()

        if entropy.numel() > 0:
             mean_entropy = entropy.mean().item()
        else:
             mean_entropy = 0.0
             
        self.stats.append("chance_entropy", mean_entropy)


    def _prepare_return_losses(self, loss_dict, total_loss):
        """Prepare loss values for return."""
        # Helper to extract and detach/item()
        def get_val(key):
            val = loss_dict.get(key, 0.0)
            if isinstance(val, torch.Tensor):
                return val.item()
            return val

        # Extract by name, defaulting to 0 if not present
        val_loss = get_val("ValueLoss")
        pol_loss = get_val("PolicyLoss")
        rew_loss = get_val("RewardLoss")
        tp_loss = get_val("ToPlayLoss")
        cons_loss = get_val("ConsistencyLoss")
        q_loss = get_val("ChanceQLoss")
        sigma_loss = get_val("SigmaLoss")
        vqvae_loss = get_val("VQVAECommitmentLoss")

        return (
            val_loss,
            pol_loss,
            rew_loss,
            tp_loss,
            cons_loss,
            q_loss,
            sigma_loss,
            vqvae_loss,
            total_loss,
        )

    def predict_initial_inference(
        self,
        states,
        model,
    ):
        if model == None:
            model = self.model
        state_inputs = self.preprocess(states)
        values, policies, hidden_states = model.initial_inference(state_inputs)
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
        to_play=None,
        *args,
        **kwargs,
    ):
        if to_play is None:
            if self.config.game.num_players != 1:
                if info is not None and "player" in info:
                    to_play = info["player"]
                else:
                    to_play = env.agents.index(env.agent_selection)
            else:
                to_play = 0

        inference_fns = {
            "initial": self.predict_initial_inference,
            "recurrent": self.predict_recurrent_inference,
            "afterstate": self.predict_afterstate_recurrent_inference,
        }

        root_value, exploratory_policy, target_policy, best_action, search_metadata = self.search.run(
            state, info, to_play, inference_fns, inference_model=inference_model
        )

        return exploratory_policy, target_policy, root_value, best_action, search_metadata

    def select_actions(
        self,
        prediction,
        temperature=0.0,
        *args,
        **kwargs,
    ):
        if temperature != 0:
            probs = prediction[0] ** temperature
            probs /= probs.sum()
            action = torch.multinomial(probs, 1)
            # print("action", action)
            return action
        else:
            # print("prediction[2]", prediction[2])
            return prediction[3]

    def play_game(self, env=None, inference_model=None):
        if env is None:
            env = self.env
        with torch.no_grad():
            if self.config.game.num_players != 1:
                env.reset()
                state, reward, terminated, truncated, info = env.last()
                agent_id = env.agent_selection
                current_player = env.agents.index(agent_id)
            else:
                state, info = env.reset()
            game: Game = Game(self.config.game.num_players)

            game.append(state, info)
    
            done = False
            while not done:
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

                prediction = self.predict(
                    state,
                    info,
                    env=env,
                    inference_model=inference_model,
                )
                action = self.select_actions(
                    prediction,
                    temperature=temperature,  # model=model
                ).item()
                # print(f"step: {len(game)}, action: {action}")
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
                done = terminated or truncated
                # essentially storing in memory, dont store terminal states for training as they are not predicted on

                # game.append(
                #     TimeStep(
                #         observation=next_state,
                #         info=next_info,
                #         action=action,
                #         reward=reward,
                #         policy=prediction[1],
                #         value=prediction[2],
                #     )
                # )
                game.append(
                    observation=next_state,
                    info=next_info,
                    action=action,
                    reward=reward,
                    policy=prediction[1],
                    value=prediction[2],
                )

                self._track_search_stats(prediction[4])
                state = next_state
                info = next_info
            self.replay_buffer.store_aggregate(game_object=game)
        if self.config.game.num_players != 1:
            return env.rewards[env.agents[0]], len(game)
        else:
            return sum(game.rewards), len(game)

    def play_game_vec(self, vec_env, inference_model=None, stop_flag=None):
        if inference_model is None:
            inference_model = self.model
            
        # Ensure dummy env is reset so we can access metadata like .agents
        self.env.reset()
            
        num_envs = vec_env.num_envs
        games = [Game(self.config.game.num_players) for _ in range(num_envs)]
        searchers = [create_mcts(self.config, self.device, self.num_actions) for _ in range(num_envs)]
        
        vec_env.reset()
        obs_stack, rewards, terms, truncs, infos = vec_env.last()
        
        while True:
            # Check for overall done
            if stop_flag and stop_flag.value:
                break
                
            # Reanalyze check (skipping for now)
            
            # --- 1. Batch MCTS Inference with Synchronized Execution ---
            
            # A. Prepare Roots
            roots = []
            min_max_stats_list = []
            pruning_contexts = []
            
            for i in range(num_envs):
                roots.append(searchers[i].prepare_root())
                
            # B. Batch Initial Inference
            # Stack observations
            # obs_stack is list of ndarrays, convert to batched tensor
            obs_tensor = self.preprocess(obs_stack) # [B, ...]
            
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type):
                    initial_outputs = self.predict_initial_inference(obs_tensor, model=inference_model)
            
            # C. Expand Roots
            for i in range(num_envs):
                # Unpack batched output for individual expansion
                # outputs is (val, policy, hidden) each [B, ...]
                # Slice the i-th batch element
                sliced_output = (
                    initial_outputs[0][i : i + 1],
                    initial_outputs[1][i : i + 1],
                    initial_outputs[2][i : i + 1],
                )
                
                _, mm_stats, p_context = searchers[i].expand_root(
                    roots[i],
                    sliced_output,
                    infos[i],
                    infos[i].get("player", 0),
                    trajectory_action=None
                )
                min_max_stats_list.append(mm_stats)
                pruning_contexts.append(p_context)

            # D. Batched Simulations
            num_sims = self.config.num_simulations
            
            for sim_idx in range(num_sims):
                # 1. Select Leaf for ALL envs
                selection_results = []
                for i in range(num_envs):
                    res = searchers[i].run_step_select(
                        roots[i],
                        min_max_stats_list[i],
                        pruning_contexts[i],
                        current_sim_idx=sim_idx
                    )
                    selection_results.append(res) # node, path, action, horizon, virtual_vals
                
                # 2. Batched Inference
                # Group by inference type needed (Recurrent vs Afterstate)
                # But typically we batch them separately or assume homogeneous model usage?
                # Actually modular_search usually differentiates based on parent node classification
                # But here we need to inspect what `run_step_select` returned as `parent` (implicitly)
                # Wait, `run_step_select` returns `node` which is the LEAF.
                # If `node` is DecisionNode, it means we selected an action from a DecisionNode (or Chance?) 
                # actually check `run_step_expand_backprop` logic.
                # It looks at `search_path[-2]` to get parent.
                
                rec_inputs = {"states": [], "actions": [], "rhs": [], "rcs": [], "indices": []}
                aft_inputs = {"states": [], "actions": [], "indices": []}
                
                for i in range(num_envs):
                    node, path, action, _, _ = selection_results[i]
                    if node is None: continue # Failed selection (prob pruned)
                    
                    parent = path[-2]
                    
                    # Store action tensor
                    if isinstance(action, torch.Tensor):
                        act_val = action.clone().detach().float()
                    else:
                        act_val = torch.tensor(action).float()
                    if act_val.dim() == 0: act_val = act_val.unsqueeze(0)
                        
                    if isinstance(node, DecisionNode):
                        # Needs Recurrent Inference
                        # Parent could be Decision or Chance
                        state = parent.hidden_state if isinstance(parent, DecisionNode) else parent.afterstate
                        
                        rec_inputs["states"].append(state.squeeze(0)) # Assume [1, D] -> [D]
                        rec_inputs["actions"].append(act_val)
                        rec_inputs["rhs"].append(parent.reward_h_state.squeeze(0))
                        rec_inputs["rcs"].append(parent.reward_c_state.squeeze(0))
                        rec_inputs["indices"].append(i)
                        
                    elif isinstance(node, ChanceNode):
                        # Needs Afterstate Inference
                        state = parent.hidden_state
                        
                        aft_inputs["states"].append(state.squeeze(0))
                        aft_inputs["actions"].append(act_val)
                        aft_inputs["indices"].append(i)

                batch_inference_results = [None] * num_envs
                
                # Execute Recurrent Batch
                if rec_inputs["indices"]:
                    b_states = torch.stack(rec_inputs["states"]).to(self.device)
                    b_actions = torch.stack(rec_inputs["actions"]).to(self.device)
                    b_rhs = torch.stack(rec_inputs["rhs"]).to(self.device)
                    b_rcs = torch.stack(rec_inputs["rcs"]).to(self.device)
                    
                    if b_actions.dim() == 1: b_actions = b_actions.unsqueeze(1)
                    
                    with torch.no_grad():
                        with torch.autocast(device_type=self.device.type):
                            (rewards, hidden_states, values, policies, to_plays, rh_news, rc_news) = \
                                self.predict_recurrent_inference(b_states, b_actions, b_rhs, b_rcs, model=inference_model)
                                
                    for j, env_idx in enumerate(rec_inputs["indices"]):
                        batch_inference_results[env_idx] = {
                            "reward": rewards[j],
                            "hidden_state": hidden_states[j : j + 1],
                            "value": values[j],
                            "policy": policies[j : j + 1],
                            "to_play": to_plays[j : j + 1],
                            "rh": rh_news[j : j + 1],
                            "rc": rc_news[j : j + 1],
                        }

                # Execute Afterstate Batch
                if aft_inputs["indices"]:
                    b_states = torch.stack(aft_inputs["states"]).to(self.device)
                    b_actions = torch.stack(aft_inputs["actions"]).to(self.device)
                    if b_actions.dim() == 1: b_actions = b_actions.unsqueeze(1)

                    with torch.no_grad():
                        with torch.autocast(device_type=self.device.type):
                             afterstates, values, code_probs = \
                                self.predict_afterstate_recurrent_inference(b_states, b_actions, model=inference_model)

                    for j, env_idx in enumerate(aft_inputs["indices"]):
                        batch_inference_results[env_idx] = {
                             "afterstate": afterstates[j : j + 1],
                             "value": values[j],
                             "code_probs": code_probs[j : j + 1],
                        }
                
                # 3. Expand & Backprop for ALL envs
                for i in range(num_envs):
                     node, path, action, horizon, v_vals = selection_results[i]
                     inf_res = batch_inference_results[i]
                     
                     if node is None or inf_res is None: continue
                     
                     searchers[i].run_step_expand_backprop(
                         node, path, action, horizon, inf_res, min_max_stats_list[i], v_vals
                     )

            # E. Extract Results
            results = []
            for i in range(num_envs):
                # Construct result tuple manually to match SearchAlgorithm.run output
                # Just call a helper or reconstruct.
                # Since we have root and stats, we can generate the policies.
                
                root = roots[i]
                mm_stats = min_max_stats_list[i]
                
                target_policy = searchers[i].root_target_policy.get_policy(root, mm_stats)
                exploratory_policy = searchers[i].root_exploratory_policy.get_policy(root, mm_stats)
                
                if searchers[i].pruning_method.mask_target_policy:
                    # Need legal moves again? Or just use what we had (but we lost it)
                    # We can re-get or assume handled. 
                    # For simplification, assume safe if pruning isn't heavily used or handled inside.
                     target_policy = action_mask(
                        target_policy.unsqueeze(0), [infos[i]["legal_moves"]]
                    ).squeeze(0)
                
                # Reconstruct the dict
                # Note: network_policy/value are from initial root expansion which we didn't save explicitly
                # but root has stored them? root.value() is updated.
                # Actually, standard run returns initial network policy/value.
                # We can't easily get them back without storing them in step C.
                # This affects stats logging primarily.
                
                # Workaround for stats: use current root stats
                root_children_values = torch.zeros(self.num_actions)
                for action, child in root.children.items():
                    if isinstance(child, (DecisionNode, ChanceNode)): 
                         root_children_values[action] = child.value()
                
                result = (
                    root.value(),
                    exploratory_policy,
                    target_policy,
                    torch.argmax(target_policy),
                    {
                        "network_policy": target_policy, # Approx
                        "network_value": 0.0, # Lost this
                        "search_policy": target_policy,
                        "search_value": root.value(),
                        "root_children_values": root_children_values,
                    }
                )
                results.append(result)

            root_values = []
            policies = []
            predictions = [] 
            actions = []
            
            game_len = len(games[0]) # Approximation for temperature schedule
            temperature = self.config.temperatures[0]
            for j, temperature_step in enumerate(self.config.temperature_updates):
                if self.config.temperature_with_training_steps:
                    if self.training_step >= temperature_step:
                         temperature = self.config.temperatures[j + 1]
                    else:
                         break
                else:
                    if game_len >= temperature_step:
                        temperature = self.config.temperatures[j + 1]
                    else:
                        break

            for i, result in enumerate(results):
                 root_value, exploratory_policy, target_policy, best_action, metadata = result
                 
                 root_values.append(root_value)
                 policies.append(target_policy)
                 predictions.append(result)
                 
                 action = self.select_actions(
                    (exploratory_policy, target_policy, root_value, best_action, metadata),
                    temperature=temperature
                 ).item()
                 actions.append(action)

            
            # --- 2. Track Search Stats (Just track first env) ---
            if self.training_step > 0 and self.stats:
                prediction = predictions[0] 
                # metadata is the 5th element [4]
                self.stats.append("root_children_values", prediction[4]["root_children_values"])
                self.stats.append("policy_improvement", {"network": prediction[4]["network_policy"], "search": prediction[2]})
                self._track_search_stats(prediction[4])
                
            # Perform Vector Step
            vec_env.step(actions)
            next_obs_stack, rewards, terms, truncs, next_infos = vec_env.last()
            self.stats.increment_steps(num_envs)
            
            for i in range(num_envs):
                done = terms[i] or truncs[i]
                
                actor_id = infos[i].get("player", 0)
                current_agent_name = self.env.agents[actor_id]
                
                step_rewards = next_infos[i].get("rewards", {})
                if done and "final_info" in next_infos[i]:
                     step_rewards = next_infos[i]["final_info"].get("rewards", step_rewards)
                
                reward = step_rewards.get(current_agent_name, 0.0)
                
                games[i].append(
                    observation=obs_stack[i],
                    info=infos[i],
                    action=actions[i],
                    reward=reward,
                    policy=predictions[i][1],
                    value=predictions[i][2],
                )
                
                if done:
                    self.replay_buffer.store_aggregate(game_object=games[i])
                    
                    final_rewards = next_infos[i]["final_info"].get("rewards", {})
                    p0_score = final_rewards.get(self.env.agents[0], 0.0)
                    
                    self.stats.append("score", p0_score)
                    self.stats.append("episode_length", len(games[i]))
                    
                    # Reset game tracker (env is auto-reset)
                    games[i] = Game(self.config.game.num_players)

                    
            obs_stack = next_obs_stack
            infos = next_infos

    def reanalyze_game(self, inference_model=None):
        # or reanalyze buffer
        with torch.no_grad():
            sample = self.replay_buffer.sample_game()
            observations = sample["observations"]
            root_values = sample["values"].to(self.device)[:, 0]
            policies = sample["policies"].to(self.device)[:, 0]
            traj_actions = sample["actions"].to(self.device)[:, 0]
            traj_to_plays = sample["to_plays"].to(self.device)[:, 0]
            legal_moves_masks = sample["legal_moves_masks"].to(self.device)
            indices = sample["indices"]
            ids = sample["ids"]

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

                if not (
                    self.config.game.has_legal_moves and len(info["legal_moves"]) == 0
                ):
                    infos.append(info)
                    # print("info with legal moves from nonzero mask", info)
                    # ADD INJECTING SEEN ACTION THING FROM MUZERO UNPLUGGED
                    if self.config.reanalyze_method == "mcts":
                        inference_fns = {
                            "initial": self.predict_initial_inference,
                            "recurrent": self.predict_recurrent_inference,
                            "afterstate": self.predict_afterstate_recurrent_inference,
                        }

                        root_value, _, new_policy, best_action, _ = self.search.run(
                            obs,
                            info,
                            to_play,
                            inference_fns,
                            trajectory_action=int(traj_action.item()),
                            inference_model=inference_model,
                        )

                        new_root_value = float(root_value)
                    else:
                        value, new_policy, _ = self.predict_initial_inference(
                            obs, model=inference_model
                        )
                        new_root_value = value.item()
                else:
                    infos.append(info)
                    new_policy = torch.ones_like(policies[0]) / self.num_actions
                    new_root_value = 0.0

                new_policies.append(new_policy)
                new_root_values.append(new_root_value)

                # decide value target per your config (paper default: keep stored n-step TD for Atari)
            # now write back under write_lock and update priorities with ids
            self.replay_buffer.reanalyze_game(
                indices,
                new_policies,
                new_root_values,
                ids,
                self.training_step,
                self.config.training_steps,
            )
            if self.config.reanalyze_update_priorities:
                stored_n_step_value = float(
                    self.replay_buffer.n_step_values_buffer[idx][0].item()
                )

                new_policies.append(new_policy[0])
                new_root_values.append(new_root_value)
                new_priorities.append(abs(float(root_value) - stored_n_step_value))

                self.update_replay_priorities(
                    indices, new_priorities, ids=np.array(ids)
                )

    def _track_search_stats(self, search_metadata):
        """Track statistics from the search process."""
        if search_metadata is None:
            return

        network_policy = search_metadata["network_policy"]
        search_policy = search_metadata["search_policy"]
        network_value = search_metadata["network_value"]
        search_value = search_metadata["search_value"]

        # 1. Policy Entropy
        # search_policy: (num_actions,)
        probs = search_policy + 1e-10
        entropy = -torch.sum(probs * torch.log(probs)).item()
        self.stats.append("policy_entropy", entropy)

        # 2. Value Difference
        self.stats.append("value_diff", abs(search_value - network_value))

        # 3. Policy Improvement (BAR plot comparison)
        self.stats.append("policy_improvement", network_policy.unsqueeze(0), subkey="network")
        self.stats.append("policy_improvement", search_policy.unsqueeze(0), subkey="search")

        # 4. Root Children Values
        if "root_children_values" in search_metadata:
             self.stats.append("root_children_values", search_metadata["root_children_values"].unsqueeze(0))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stop_flag"] = state["stop_flag"].value
        if "env" in state:
            del state["env"]
        if "test_env" in state:
            del state["test_env"]
        if "optimizer" in state:
            del state["optimizer"]
        if "lr_scheduler" in state:
            del state["lr_scheduler"]
        
        # Only handle these if training has started (step > 0)
        # At step 0 (worker spawn), model is on CPU and replay_buffer is empty/picklable.
        if self.training_step > 0:
            # Manually serialize model to CPU state dict to avoid device sharing issues (MPS etc)
            if "model" in state:
                state["model_state_dict"] = {k: v.cpu() for k, v in self.model.state_dict().items()}
                del state["model"]
            if "target_model" in state:
                 del state["target_model"]
            
            if "loss_pipeline" in state:
                del state["loss_pipeline"]

            if "replay_buffer" in state:
                del state["replay_buffer"]

        return state

    def __setstate__(self, state):
        model_state_dict = state.pop("model_state_dict", None)
        
        # Strip _orig_mod prefix added by torch.compile
        if model_state_dict is not None:
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict

        self.__dict__.update(state)
        self.stop_flag = mp.Value("i", state["stop_flag"])
        self.env = self.config.game.make_env()
        self.test_env = self.config.game.make_env(render_mode="rgb_array")
        
        # Reconstruct model if we have weights
        if model_state_dict is not None:
             # self.config, self.observation_dimensions, self.num_actions are in state
             device = torch.device("cpu") # Initialize on CPU
             self.model = Network(
                config=self.config,
                num_actions=self.num_actions,
                input_shape=torch.Size((self.config.minibatch_size,) + self.observation_dimensions),
                channel_first=True,
                world_model_cls=self.config.world_model_cls,
            )
             self.model.load_state_dict(model_state_dict)
             self.model.to(self.device)
             self.target_model = copy.deepcopy(self.model)
             
             # Apply quantization to reconstructed target model
             if self.config.quantize:
                 try:
                     if 'qnnpack' in torch.backends.quantized.supported_engines:
                          torch.backends.quantized.engine = 'qnnpack'
                 except:
                     pass
                 self.target_model.to('cpu')
                 self.target_model = torch.ao.quantization.quantize_dynamic(
                     self.target_model,
                     {nn.Linear},
                     dtype=torch.qint8
                 )
                 self.target_model.eval()
                 for p in self.target_model.parameters():
                     p.requires_grad = False

             # Move target to shared memory if multi_process (though for testing we might just use local copy)
             if self.config.multi_process:
                 # Note: sharing CUDA/MPS tensors is tricky. If device is CPU, this is fine.
                 # If device is MPS, this might fail or be no-op. 
                 # Given we just deserialized, we are likely fine keeping it local for the test worker.
                 # But if this is a training worker, it might expect shared memory?
                 # Training workers use target_model for inference.
                 # A reconstructed target_model here is NOT shared with the main process 'target_model'.
                 # This implies __setstate__ is creating a LOCAL copy.
                 # If training workers need the SHARED target model updated by learner, 
                 # then training workers CANNOT rely on this reconstruction!
                 # Training workers rely on the pickled 'target_model' which points to shared memory.
                 # WE DELETED 'target_model' from state!
                 pass

    def update_target_model(self):
        """
        Override to support quantized target model updates.
        """
        if not self.config.quantize:
            super().update_target_model()
            return

        with torch.no_grad():
            def update_recursively(target_module, source_module):
                source_children = dict(source_module.named_children())
                for name, target_child in target_module.named_children():
                    if name not in source_children:
                        continue
                    source_child = source_children[name]
                    
                    if isinstance(target_child, (nn.quantized.dynamic.Linear, torch.ao.nn.quantized.dynamic.Linear)) and isinstance(source_child, nn.Linear):
                        # Workaround for set_weight_bias backend issues:
                        # Create new quantized module via from_float (identical to how quantize_dynamic does it)
                        # and steal its packed params.
                        # We use source_child directly (it's float). from_float usually copies weights.
                        # We must attach qconfig to a copy of source to avoid modifying training model.
                        temp_source = copy.copy(source_child)
                        temp_source.qconfig = torch.ao.quantization.default_dynamic_qconfig
                        new_q_child = type(target_child).from_float(temp_source)
                        target_child._packed_params = new_q_child._packed_params
                    else:
                        update_recursively(target_child, source_child)
            
            if self.config.soft_update:
                pass # Soft update not supported for quantized yet
            else:
                 update_recursively(self.target_model, self.model)

