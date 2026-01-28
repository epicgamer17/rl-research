import copy
import datetime
import random
import sys

from replay_buffers.buffer_factories import create_muzero_buffer
from replay_buffers.game import Game, TimeStep
from search.search_factories import create_mcts


sys.path.append("../")
from time import time
import traceback
from modules.utils import scalar_to_support, support_to_scalar, get_lr_scheduler
import numpy as np
from stats.stats import PlotType, StatTracker
from losses.losses import create_muzero_loss_pipeline

from agents.agent import MARLBaseAgent
from agent_configs.muzero_config import MuZeroConfig
import torch
import torch.nn as nn
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
import torch.ao.quantization
from torch.nn import quantized


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
        self.env.reset()  # for multiprocessing

        # Add learning rate scheduler
        self.model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=torch.Size(
                (self.config.minibatch_size,) + self.observation_dimensions
            ),
            # TODO: sort out when to do channel first and channel last
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )

        self.target_model = Network(
            config=config,
            num_actions=self.num_actions,
            input_shape=torch.Size(
                (self.config.minibatch_size,) + self.observation_dimensions
            ),
            channel_first=True,
            world_model_cls=self.config.world_model_cls,
        )
        # copy weights
        self.target_model.load_state_dict(self.model.state_dict())
        # self.model.share_memory()

        if self.config.multi_process:
            # make sure target is placed in shared memory so worker processes can read it
            self.target_model.share_memory()
            self.update_queues = [mp.Queue() for _ in range(self.config.num_workers)]
        else:
            # non-multiprocess: keep target on device for faster inference
            self.target_model.to(device)
            self.update_queues = []

        if not self.config.multi_process:
            self.model.to(device)
            self.target_model.to(device)

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

        self.lr_scheduler = None
        self.optimizer = None

        if self.config.use_mixed_precision:
            self.scaler = torch.amp.GradScaler(device=self.device.type)

        test_score_keys = [
            "test_score_vs_{}".format(agent.model_name) for agent in self.test_agents
        ]
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
            "score",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "to_play_loss",
            "cons_loss",
            "loss",
            "test_score",
            "episode_length",
            "policy_entropy",
            "value_diff",
            "policy_improvement",
            "root_children_values",
        ] + test_score_keys

        if self.config.stochastic:
            stat_keys += [
                "num_codes",
                "chance_probs",
                "chance_entropy",
                "q_loss",
                "sigma_loss",
                "vqvae_commitment_cost",
            ]

        target_values = {
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
            "num_codes": 1 if self.config.game.is_deterministic else None,
        }

        use_tensor_dicts = {
            "test_score": ["score", "max_score", "min_score"],
            "policy_improvement": ["network", "search"],
            **{
                key: ["score"]
                + ["{}_score".format(p) for p in self.env.possible_agents]
                + ["{}_win%".format(p) for p in self.env.possible_agents]
                for key in test_score_keys
            },
        }

        # For host StatTracker, initialize keys that don't exist
        if not self.stats._is_client:
            for key in stat_keys:
                if key not in self.stats.stats:
                    self.stats._init_key(
                        key,
                        target_value=target_values.get(key),
                        subkeys=use_tensor_dicts.get(key),
                    )

        # 2. Add/Refresh Plot Types
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
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
        self.stats.add_plot_types("num_codes", PlotType.ROLLING_AVG, rolling_window=100)
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
        self.stats.add_plot_types(
            "policy_entropy", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "value_diff", PlotType.ROLLING_AVG, rolling_window=100
        )
        self.stats.add_plot_types(
            "policy_improvement", PlotType.BAR, bar_threshold=0.01, max_bars=20
        )
        self.stats.add_plot_types(
            "root_children_values", PlotType.BAR, bar_threshold=1e-8, max_bars=30
        )

        if self.config.stochastic:
            self.stats.add_plot_types("chance_probs", PlotType.BAR, show_all_bars=True)
            self.stats.add_plot_types(
                "chance_entropy", PlotType.ROLLING_AVG, rolling_window=100
            )

    @staticmethod
    def run_worker(
        worker_id,
        config,
        model_queue: mp.Queue,
        replay_buffer,
        device,
        stop_flag,
        stats_client: StatTracker,
        error_queue: mp.Queue,
        model_name,
        checkpoint_interval,
        num_actions: int,
        training_step_shared,
    ):
        # Create a lightweight agent shell to reuse methods
        agent = MuZeroAgent.__new__(MuZeroAgent)
        agent.config = config
        agent.device = device
        agent.num_actions = num_actions
        # Ensure model is None so it's not accidentally used
        agent.model = None
        # target_model will be fetched from the queue
        agent.target_model = None
        agent.replay_buffer = replay_buffer
        agent.stats = stats_client
        agent.model_name = model_name
        agent.checkpoint_interval = checkpoint_interval
        # Set the shared training step value so agent.training_step property works
        agent._training_step = training_step_shared
        # Set stop flag just in case
        agent.stop_flag = stop_flag

        # 1. Set engine immediately!
        # This ensures unpickling quantized weights (which triggers __setstate__)
        # works correctly in the worker process.
        if config.qat or config.use_quantization:
            torch.backends.quantized.engine = "qnnpack"

        print(f"[Worker {worker_id}] Starting self-play...")
        worker_env = config.game.make_env()  # each process needs its own env
        agent.env = worker_env

        # Detect player_id (needed for observation dimension determination in some envs)
        if hasattr(worker_env, "possible_agents") and worker_env.possible_agents:
            agent.player_id = worker_env.possible_agents[0]
        elif hasattr(worker_env, "agents") and worker_env.agents:
            agent.player_id = worker_env.agents[0]
        else:
            agent.player_id = "player_0"

        # Determine observation dimensions/dtype (needed for preprocess)
        agent.observation_dimensions, agent.observation_dtype = (
            agent.determine_observation_dimensions(worker_env)
        )

        # Determine action space properties (needed for creating MCTS if not passed in config or internal logic)
        agent._setup_action_space(worker_env)

        try:
            from wrappers import record_video_wrapper

            worker_env.render_mode = "rgb_array"
            worker_env = record_video_wrapper(
                worker_env,
                f"./videos/{model_name}/{worker_id}",
                checkpoint_interval,
            )
        except Exception:
            print(f"[Worker {worker_id}] Could not record video")

        # 0. Re-initialize search for the worker process (MCTS tree can't be shared safely)
        agent.search = create_mcts(config, device, num_actions)

        # 1. Local Model Initialization
        def prepare_local_model():
            # agent.target_model is provided by the learner via model_queue
            if agent.target_model is None:
                return None

            # The target_model is already fully quantized if config.use_quantization/qat is True
            # via update_target_model().
            local_model = copy.deepcopy(agent.target_model)
            local_model.to("cpu")
            local_model.eval()

            # Ensure quantization engine is set for inference (redundant but safe)
            if config.qat or config.use_quantization:
                torch.backends.quantized.engine = "qnnpack"

            if config.compile:
                local_model = torch.compile(local_model, mode=config.compile_mode)

            for p in local_model.parameters():
                p.requires_grad = False
            return local_model

        # Wait for the first model from the learner
        print(f"[Worker {worker_id}] Waiting for initial model from queue...")
        agent.target_model = model_queue.get()
        local_inference_model = prepare_local_model()
        print(f"[Worker {worker_id}] Model received and prepared.")

        try:
            while not stop_flag.value:
                # 2. Synchronization Logic via Queue
                # Check if there's a new model in the queue
                if not model_queue.empty():
                    # Drain the queue to get the latest model
                    latest_model = None
                    while not model_queue.empty():
                        latest_model = model_queue.get()

                    if latest_model is not None:
                        agent.target_model = latest_model
                        local_inference_model = prepare_local_model()
                        # print(f"[Worker {worker_id}] Refreshed local model from queue")

                # 3. Inference Execution
                use_amp = config.use_mixed_precision and not config.use_quantization

                # Determine device type for autocast safely
                device_type = "cpu"
                if hasattr(local_inference_model, "parameters"):
                    try:
                        device_type = next(
                            local_inference_model.parameters()
                        ).device.type
                    except StopIteration:
                        pass

                with torch.amp.autocast(
                    device_type=device_type,
                    enabled=use_amp,
                ):
                    if (
                        random.random() < config.reanalyze_ratio
                        and agent.replay_buffer.size > 0
                    ):
                        agent.reanalyze_game(inference_model=local_inference_model)
                    else:
                        score, num_steps = agent.play_game(
                            env=worker_env, inference_model=local_inference_model
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
        finally:
            worker_env.close()

    def update_target_model(self):
        """
        Updates self.target_model from self.model.
        Handles:
         1. Stripping `_orig_mod` prefix if self.model is compiled.
         2. Converting to INT8 if quantization is enabled.
         3. Dispatching to update_queues (if multi-process) or updating self.target_model locally.
        """
        # 1. Access the underlying module if compiled
        if hasattr(self.model, "_orig_mod"):
            input_model = self.model._orig_mod
        else:
            input_model = self.model

        # 2. Create a clean CPU copy for conversion/dispatch
        # We use a deepcopy to ensure the worker's model is independent
        temp_model = copy.deepcopy(input_model).cpu()
        temp_model.eval()
        for p in temp_model.parameters():
            p.requires_grad = False

        # 3. Convert to Quantized Model if enabled
        if self.config.qat or self.config.use_quantization:
            # Check if this is the initial update (step 0).
            # If so, we SKIP convert() because observers are empty and it would crash.
            # We must also strip the qconfig/observers so the model is pickleable for workers.
            is_initial_step = (self.training_step == 0) or (
                self.stats is not None and self.stats.get_num_steps() == 0
            )

            if is_initial_step:
                # Sanitize the Float32 model for workers.
                # We replace QuantStub and DeQuantStub with Identity because setting
                # activation_post_process to None causes a TypeError in workers.
                def replace_stubs_with_identity(module):
                    for name, child in module.named_children():
                        if isinstance(
                            child,
                            (
                                torch.ao.quantization.QuantStub,
                                torch.ao.quantization.DeQuantStub,
                            ),
                        ):
                            setattr(module, name, torch.nn.Identity())
                        else:
                            replace_stubs_with_identity(child)

                replace_stubs_with_identity(temp_model)

                for m in temp_model.modules():
                    if hasattr(m, "qconfig"):
                        m.qconfig = None
            else:
                # This converts inplace, fusing observers into quantized layers.
                # It requires the model to have been prepared (prepare_qat or prepare).
                torch.ao.quantization.convert(temp_model, inplace=True)

        # 4. Dispatch or Update Local Target
        if self.config.multi_process:
            # Push to every worker's queue
            # Ensure it's in shared memory if possible
            temp_model.share_memory()
            for q in self.update_queues:
                q.put(temp_model)
        else:
            self.target_model = temp_model.to(self.device)
            # Update inference_model reference
            self.inference_model = self.target_model

    def train(self):
        self._setup_stats()
        if self.config.multi_process:
            stats_client = self.stats.get_client()
            error_queue = mp.Queue()

            workers = [
                mp.Process(
                    target=MuZeroAgent.run_worker,
                    args=(
                        i,
                        self.config,
                        self.update_queues[i],
                        self.replay_buffer,
                        self.device,
                        self.stop_flag,
                        stats_client,
                        error_queue,
                        self.model_name,
                        self.checkpoint_interval,
                        self.num_actions,
                        self._training_step,
                    ),
                )
                for i in range(self.config.num_workers)
            ]
            for w in workers:
                w.start()

        # --- Late Initialization (Learner Setup) ---
        # Now that workers have started with clean models, we can prepare the learner.

        # 1. Quantization Preparation
        if self.config.qat or self.config.use_quantization:
            # Force "qnnpack" for ARM/M-series (or x86 too if consistent)
            torch.backends.quantized.engine = "qnnpack"

            if self.config.qat:
                # Quantization Aware Training (QAT)
                self.model.qconfig = torch.ao.quantization.get_default_qat_qconfig(
                    "qnnpack", version=0
                )
                self.model.fuse_model()
                torch.ao.quantization.prepare_qat(self.model, inplace=True)
            elif self.config.use_quantization:
                # Post Training Quantization (PTQ) - Preparation Phase
                self.model.qconfig = torch.ao.quantization.get_default_qconfig(
                    "qnnpack"
                )
                self.model.fuse_model()
                torch.ao.quantization.prepare(self.model, inplace=True)

        # 1.5 Initialize Optimizer and LR Scheduler
        # This must happen AFTER QAT preparation (fusion) because it swaps modules
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

        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)
        # Re-bind model to loss_pipeline in case it was re-initialized or needs refresh
        if hasattr(self.loss_pipeline, "model"):
            self.loss_pipeline.model = self.model

        # 2. Compilation
        if self.config.compile:
            print("Compiling models in train()...")
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
            # Only compile target_model if NOT multiprocessing (workers have their own compiled copies)
            if not self.config.multi_process:
                self.target_model = torch.compile(
                    self.target_model, mode=self.config.compile_mode
                )

        # 3. Initial Sync
        # Call update_target_model to convert current model and dispatch to workers
        self.update_target_model()

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

        # --- 2. Forward Pass within AMP context ---
        with torch.amp.autocast(
            device_type=self.device.type, enabled=self.config.use_mixed_precision
        ):
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
                targets_tensor["encoder_onehots"] = predictions_tensor[
                    "encoder_onehots"
                ]

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

            # --- 7. Run Modular Loss Pipeline ---
            # NOTE: We run this within autocast as requested.
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
            "latent_root", s0, labels=a0, method=self.config.latent_viz_method
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
            prob_sums = latent_code_probs_tensor.sum(dim=-1)  # (B, K)
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
        unique_codes_all = torch.unique(
            valid_codes
        )  # 1D tensor with sorted unique indices
        num_unique_all = unique_codes_all.numel()
        # Optionally: convert to Python int
        num_unique_all_int = int(num_unique_all)
        self.stats.append("num_codes", num_unique_all_int)

        # Track chance probability statistics (mean over batch and time)
        latent_node_probs = (
            latent_code_probs_tensor  # (B, K+1, NumCodes) or (B, K, NumCodes)
        )

        # Calculate mean probabilities for each code across batch and unroll steps
        # Result shape: (NumCodes,)
        if latent_node_probs.ndim == 3:
            valid_probs = latent_node_probs[mask]  # (N_valid, NumCodes)
            if valid_probs.shape[0] > 0:
                mean_probs = valid_probs.mean(dim=0)
            else:
                mean_probs = torch.zeros(
                    latent_node_probs.shape[-1], device=latent_node_probs.device
                )
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

    def predict_initial_inference(self, states, model):
        if model is None:
            model = self.model
        state_inputs = self.preprocess(states)

        # Safety: Ensure inputs match model device (important for CPU workers)
        # and are Float32 (critical for Quantization which crashes on BFloat16)
        if hasattr(model, "parameters"):
            try:
                param = next(model.parameters())
                if state_inputs.device != param.device:
                    state_inputs = state_inputs.to(param.device)
                if state_inputs.dtype == torch.bfloat16:
                    state_inputs = state_inputs.to(dtype=torch.float32)
            except StopIteration:
                pass

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
        if model is None:
            model = self.model

        # Safety: Ensure inputs match model device/dtype
        if hasattr(model, "parameters"):
            try:
                param = next(model.parameters())
                if states.device != param.device:
                    states = states.to(param.device)
                    actions_or_codes = actions_or_codes.to(param.device)
                if states.dtype == torch.bfloat16:
                    states = states.to(dtype=torch.float32)
            except StopIteration:
                pass

        rewards, states, values, policies, to_play, reward_hidden = (
            model.recurrent_inference(
                states,
                actions_or_codes,
                reward_h_states,
                reward_c_states,
            )
        )

        reward_h_states = reward_hidden[0]
        reward_c_states = reward_hidden[1]

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
        if model is None:
            model = self.model

        # Safety: Ensure inputs match model device/dtype
        if hasattr(model, "parameters"):
            try:
                param = next(model.parameters())
                if hidden_states.device != param.device:
                    hidden_states = hidden_states.to(param.device)
                    actions = actions.to(param.device)
                if hidden_states.dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(dtype=torch.float32)
            except StopIteration:
                pass

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

        root_value, exploratory_policy, target_policy, best_action, search_metadata = (
            self.search.run(
                state, info, to_play, inference_fns, inference_model=inference_model
            )
        )

        return (
            exploratory_policy,
            target_policy,
            root_value,
            best_action,
            search_metadata,
        )

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

            # Disable autocast for buffer storage to prevent "Unexpected floating ScalarType" errors
            # on CPU/MPS during torch.cat operations.
            with torch.amp.autocast(device_type="cpu", enabled=False):
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
            with torch.amp.autocast(device_type="cpu", enabled=False):
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
        self.stats.append(
            "policy_improvement", network_policy.unsqueeze(0), subkey="network"
        )
        self.stats.append(
            "policy_improvement", search_policy.unsqueeze(0), subkey="search"
        )

        # 4. Root Children Values
        if "root_children_values" in search_metadata:
            self.stats.append(
                "root_children_values",
                search_metadata["root_children_values"].unsqueeze(0),
            )

    def __getstate__(self):
        state = super().__getstate__()
        state["stop_flag"] = state["stop_flag"].value
        if "_training_step" in state and hasattr(state["_training_step"], "value"):
            state["_training_step"] = state["_training_step"].value

        # Unconditionally remove learner-specific heavy objects
        for key in [
            "model",
            "loss_pipeline",
            "search",
            "optimizer",
            "scaler",
            "lr_scheduler",
        ]:
            if key in state:
                del state[key]

        # NOTE: We keep "replay_buffer" and "target_model" so workers can use them.

        # Optional: Save model weights if they exist (for checkpointing, not for workers)
        if hasattr(self, "model") and self.model is not None:
            try:
                state["model_state_dict"] = {
                    k: v.cpu() for k, v in self.model.state_dict().items()
                }
            except:
                pass

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
        if hasattr(self, "_training_step") and isinstance(self._training_step, int):
            self._training_step = mp.Value("i", self._training_step)
        self.env = self.config.game.make_env()
        self.test_env = self.config.game.make_env(render_mode="rgb_array")

        # Reconstruct model if we have weights (e.g. from a checkpoint)
        if model_state_dict is not None:
            # self.config, self.observation_dimensions, self.num_actions are in state
            self.model = Network(
                config=self.config,
                num_actions=self.num_actions,
                input_shape=torch.Size(
                    (self.config.minibatch_size,) + self.observation_dimensions
                ),
                channel_first=True,
                world_model_cls=self.config.world_model_cls,
            )
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.target_model = copy.deepcopy(self.model)
        else:
            # Workers operate fine with self.model = None because they use self.target_model
            # Note: We do NOT quantize here anymore; workers handle it locally.
            self.model = None
            self.loss_pipeline = None
            self.search = None
            if not hasattr(self, "replay_buffer"):
                self.replay_buffer = None

        # Re-initialize MCTS if missing (critical for predict() in subprocesses)
        if not hasattr(self, "search") or self.search is None:
            self.search = create_mcts(self.config, self.device, self.num_actions)
