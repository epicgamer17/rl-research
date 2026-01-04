import datetime
import random
import sys

from replay_buffers.buffer_factories import create_muzero_buffer
from replay_buffers.game import Game, TimeStep
from search.search_factories import create_mcts


sys.path.append("../")
from time import time
import traceback
from modules.utils import scalar_to_support, support_to_scalar
import numpy as np
from stats.stats import PlotType, StatTracker
from losses.losses import create_muzero_loss_pipeline

from agents.agent import MARLBaseAgent
from agent_configs.muzero_config import MuZeroConfig
import torch
import torch.nn.functional as F
from modules.agent_nets.muzero import Network
import datetime

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

        self.loss_pipeline = create_muzero_loss_pipeline(
            config=self.config,
            device=self.device,
            predict_initial_inference_fn=self.predict_initial_inference,
            preprocess_fn=self.preprocess,
            model=self.model,
        )

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

    def worker_fn(
        self, worker_id, stop_flag, stats_client: StatTracker, error_queue: mp.Queue
    ):
        print(f"[Worker {worker_id}] Starting self-play...")
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

        # --- MASKS ---
        # valid_masks: (B, unroll + 1). True if state is in episode (inc. Terminal).
        # Use for: Value, Reward, Consistency.
        consistency_masks = samples["valid_masks"].to(self.device)

        # policy_masks: (B, unroll + 1). True if state is actionable (exc. Terminal).
        # Use for: Policy.
        policy_masks = samples["policy_masks"].to(self.device)

        weights = samples["weights"].to(self.device)
        inputs = self.preprocess(observations)

        # --- 2. Initial Inference ---
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
            targets_tensor["chance_values"] = target_values[:, :-1]  # k-1 indexing
            targets_tensor["encoder_onehots"] = predictions_tensor["encoder_onehots"]

        gradient_scales_tensor = torch.tensor(
            gradient_scales, device=self.device
        ).reshape(
            1, -1
        )  # (1, K+1)

        # --- 6. Create Context for Loss Computation ---
        context = {
            "consistency_masks": consistency_masks,
            "policy_masks": policy_masks,
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
                    policy_masks,
                    consistency_masks,
                    predictions_tensor if self.config.stochastic else None,
                )

            # --- 9. Backpropagation and Optimization ---
            self.optimizer.zero_grad()
            loss_mean.backward()
            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)
            self.optimizer.step()

            # --- 10. Update Priorities ---
            # priorities tensor is already of shape (B,) from k=0
            self.update_replay_priorities(
                samples["indices"], priorities.cpu().numpy(), ids=samples["ids"]
            )

            # --- 11. STAT TRACKING ---
            if self.config.stochastic:
                self._track_stochastic_stats(predictions_tensor["encoder_onehots"])

        # --- 12. Return Losses for Logging ---
        return self._prepare_return_losses(loss_dict, loss_mean.item())

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
        policy_masks,
        consistency_masks,
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
        print("masks shape", policy_masks.shape, consistency_masks.shape)

        print("actions", actions)
        print("target value", target_values)
        print("predicted values", predicted_values)
        print("target rewards", target_rewards)
        print("predicted rewards", predicted_rewards)
        if self.config.stochastic:
            print("target qs", target_values)
            print("predicted qs", stochastic_preds["chance_values"])
        print("target to plays", target_to_plays)
        print("predicted to_plays", predicted_to_plays)

        if self.config.stochastic:
            print("encoder embedding", stochastic_preds["encoder_softmaxes"])
            print("encoder onehot", stochastic_preds["encoder_onehots"])
            print("predicted sigmas", stochastic_preds["latent_code_probabilities"])
        print("masks", policy_masks, consistency_masks)
        # torch.set_printoptions(profile="default")

    def _track_stochastic_stats(self, encoder_onehots_tensor):
        """Track statistics for stochastic MuZero."""
        codes = encoder_onehots_tensor.argmax(dim=-1)  # shape (B, K), dtype long
        # --- (A) Total unique codes across entire batch+time ---
        unique_codes_all = torch.unique(codes)  # 1D tensor with sorted unique indices
        num_unique_all = unique_codes_all.numel()
        # Optionally: convert to Python int
        num_unique_all_int = int(num_unique_all)
        self.stats.append("num_codes", num_unique_all_int)

    def _prepare_return_losses(self, loss_dict, total_loss):
        """Prepare loss values for return."""
        # Extract by name, defaulting to 0 if not present
        val_loss = loss_dict.get("ValueLoss", 0.0)
        pol_loss = loss_dict.get("PolicyLoss", 0.0)
        rew_loss = loss_dict.get("RewardLoss", 0.0)
        tp_loss = loss_dict.get("ToPlayLoss", 0.0)
        cons_loss = loss_dict.get("ConsistencyLoss", 0.0)
        q_loss = loss_dict.get("ChanceQLoss", 0.0)
        sigma_loss = loss_dict.get("SigmaLoss", 0.0)
        vqvae_loss = loss_dict.get("VQVAECommitmentLoss", 0.0)

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
        *args,
        **kwargs,
    ):
        if self.config.game.num_players != 1:
            to_play = env.agents.index(env.agent_selection)
        else:
            to_play = 0

        inference_fns = {
            "initial": self.predict_initial_inference,
            "recurrent": self.predict_recurrent_inference,
            "afterstate": self.predict_afterstate_recurrent_inference,
        }

        root_value, exploratory_policy, target_policy, best_action = self.search.run(
            state, info, to_play, inference_fns, inference_model=inference_model
        )

        return exploratory_policy, target_policy, root_value, best_action

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

                state = next_state
                info = next_info
            self.replay_buffer.store_aggregate(game_object=game)
        if self.config.game.num_players != 1:
            return env.rewards[env.agents[0]], len(game)
        else:
            return sum(game.rewards), len(game)

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

                        root_value, _, new_policy, best_action = self.search.run(
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
