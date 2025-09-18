import datetime
import sys
from time import time

from numpy import save

sys.path.append("../")


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
    process_petting_zoo_obs,
    scale_gradient,
    update_per_beta,
)
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class MuZeroAgent(MARLBaseAgent):
    def __init__(
        self,
        env,
        config: MuZeroConfig,
        name=datetime.datetime.now().timestamp(),
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            # else (
            #     torch.device("mps")
            #     if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else torch.device("cpu")
            # )
        ),
        from_checkpoint=False,
    ):
        super(MuZeroAgent, self).__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )

        # Add learning rate scheduler
        self.model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            action_function=self.config.action_function,
        )

        self.model.to(device)

        self.replay_buffer = MuZeroReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.observation_dtype,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            n_step=self.config.n_step,
            num_unroll_steps=self.config.unroll_steps,
            gamma=self.config.discount_factor,
            has_intermediate_rewards=self.config.game.has_intermediate_rewards,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
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

        if self.config.game.has_intermediate_rewards:
            self.stats = {
                "score": [],
                "policy_loss": [],
                "value_loss": [],
                "reward_loss": [],
                # "l2_loss": [],
                "loss": [],
                "test_score": [],
                "test_score_vs_random": [],
            }
        else:
            self.stats = {
                "score": [],
                "policy_loss": [],
                "value_loss": [],
                # "reward_loss": [],
                # "l2_loss": [],
                "loss": [],
                "test_score": [],
                "test_score_vs_random": [],
            }

        if hasattr(self.env, "spec"):
            self.targets = {
                "score": self.env.spec.reward_threshold,
                # "value_loss": 0,
                # "policy_loss": 0,
                # "l2_loss": 0,
                # "loss": 0,
                "test_score": self.env.spec.reward_threshold,
            }
        else:
            self.targets = {
                # "value_loss": 0,
                # "policy_loss": 0,
                # "reward_loss": 0,
                # "l2_loss": 0,
                # "loss": 0,
            }

    def train(self):
        super().train()
        start_time = time() - self.training_time
        if self.training_step == 0:
            self.print_resume_training()

        while self.training_step < self.config.training_steps:
            if self.training_step % self.config.print_interval == 0:
                self.print_training_progress()
            for training_game in tqdm(range(self.config.games_per_generation)):
                if self.stop_flag:
                    print("Stopping game generation")
                    break

                # print("Training Game ", training_game + 1)
                score, num_steps = self.play_game()
                self.total_environment_steps += num_steps
                self.stats["score"].append({"score": score})  # score for player one
            if self.stop_flag:
                print("Stopping training")
                break

            self.replay_buffer.set_beta(
                update_per_beta(
                    self.replay_buffer.beta,
                    self.config.per_beta_final,
                    self.training_steps,
                    self.config.per_beta,
                )
            )

            # STAT TRACKING
            if self.replay_buffer.size >= self.config.min_replay_buffer_size:
                for minibatch in range(self.config.num_minibatches):
                    value_loss, policy_loss, reward_loss, loss = self.learn()
                    self.stats["value_loss"].append({"loss": value_loss})
                    self.stats["policy_loss"].append({"loss": policy_loss})
                    if self.config.game.has_intermediate_rewards:
                        self.stats["reward_loss"].append({"loss": reward_loss})
                # self.stats["l2_loss"].append(l2_loss)
                self.stats["loss"].append({"loss": loss})
                print("Losses", value_loss, policy_loss, reward_loss, loss)

            self.training_step += 1
            # CHECKPOINTING
            if (
                self.training_step % self.checkpoint_interval == 0
                and self.training_step > 0
            ):
                self.training_time = time() - start_time
                self.save_checkpoint(save_weights=self.config.save_intermediate_weights)

        print("Finished Training")
        self.training_time = time() - start_time
        self.save_checkpoint(save_weights=True)
        # save model to shared storage @Ezra

    def monte_carlo_tree_search(self, env, state, info):
        root = Node(0)
        _, policy, hidden_state = self.predict_single_initial_inference(state, info)
        # print("Initial policy", policy)
        # print("Initial value", _)
        legal_moves = get_legal_moves(info)[0]
        to_play = env.agents.index(env.agent_selection)
        root.expand(legal_moves, to_play, policy, hidden_state, 0)

        if env == self.env:
            root.add_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            to_play = env.agents.index(env.agent_selection)

            # GO UNTIL A LEAF NODE IS REACHED
            while node.expanded():
                action, node = node.select_child(
                    min_max_stats,
                    self.config.pb_c_base,
                    self.config.pb_c_init,
                    self.config.discount_factor,
                )
                # THIS NEEDS TO BE CHANGED FOR GAMES WHERE PLAYER COUNT DECREASES AS PLAYERS GET ELIMINATED, USE agent_selector.next() (clone of the current one)
                to_play = (to_play + 1) % self.config.game.num_players
                search_path.append(node)
            # print(search_path)
            parent = search_path[-2]
            reward, hidden_state, value, policy = (
                self.predict_single_recurrent_inference(parent.hidden_state, action)
            )

            node.expand(
                list(range(self.num_actions)),
                to_play,
                policy,
                hidden_state,
                (
                    reward if self.config.game.has_intermediate_rewards else 0
                ),  # for board games and games with no intermediate rewards
            )

            # print("back propagation of mcts")
            # v = 0, r = 1
            # vs = 0 r = 1 tp = 0, pp0 loser
            # v = 1, pp0
            # vs = -1 r = 0 tp = 0, pp0 winner
            # v = 1, pp0
            # vs = 1 r = 0 tp = 0, pp0 loser
            # v = 1
            # vs = -1 r = 0, tp = 0, pp0 winner ROOT

            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                # print("node value", node.value_sum)
                node.visits += 1
                min_max_stats.update(node.value())
                # print("node reward", node.reward)
                value = (
                    node.reward if node.to_play == to_play else -node.reward
                ) + self.config.discount_factor * value

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]
        return root.value(), visit_counts

    def learn(self):
        samples = self.replay_buffer.sample()
        print("Samples:", samples)
        observations = samples["observations"]
        target_policies = samples["policy"]
        target_values = samples["values"]
        target_rewards = samples["rewards"]
        actions = samples["actions"]
        infos = samples["infos"]
        inputs = self.preprocess(observations)

        for training_iteration in range(self.config.training_iterations):
            loss = 0
            val_loss = 0
            pol_loss = 0
            rew_loss = 0
            priorities = []
            for item in range(self.config.minibatch_size):
                value, policy, hidden_state = self.predict_single_initial_inference(
                    inputs[item], infos[item]
                )

                gradient_scales = [1.0]
                values = [value]
                rewards = [torch.tensor([0.0])]
                policies = [policy]

                for action in actions[item]:
                    if action is None:
                        # self absorbing state, give a random action (legal moves not important as state is not technically valid)
                        # item_player = self.env.agents[infos[item]["player"]]
                        action = self.env.action_space(self.env.agents[0]).sample()
                    reward, hidden_state, value, policy = (
                        self.predict_single_recurrent_inference(hidden_state, action)
                    )
                    gradient_scales.append(1.0 / len(actions[item]))
                    values.append(value)
                    rewards.append(reward)
                    policies.append(policy)

                    hidden_state = scale_gradient(hidden_state, 0.5)

                # Convert to tensors
                if self.config.game.has_intermediate_rewards:
                    print(values, rewards, policies)
                else:
                    print(values, policies)
                values_tensor = torch.stack([v for v in values])
                rewards_tensor = torch.stack([r for r in rewards])
                policies_tensor = torch.stack([p for p in policies])
                gradient_scales_tensor = torch.tensor(gradient_scales)

                assert len(values) == len(target_values[item])
                assert len(rewards) == len(target_rewards[item])
                assert len(policies) == len(target_policies[item])

                for (
                    k,
                    value,
                    reward,
                    policy,
                    target_value,
                    target_reward,
                    target_policy,
                    scale,
                ) in zip(
                    range(len(values)),
                    values_tensor,
                    rewards_tensor,
                    policies_tensor,
                    target_values[item],
                    target_rewards[item],
                    target_policies[item],
                    gradient_scales_tensor,
                ):
                    print("Predicted Value:", value)
                    print("Target Value:", target_value)
                    if self.config.game.has_intermediate_rewards:
                        print("Predicted Reward:", reward)
                        print("Target Reward:", target_reward)
                    print("Predicted Policy:", policy)
                    print("Target Policy:", target_policy)

                    if k == 0:
                        # TODO: ADD FUNCTIONALITY FOR PASSING IN DIFFERENT PRIORITY FUNCTIONS
                        priorities.append(
                            abs(target_value - value).item() + self.config.per_epsilon
                        )  # + abs(target_policy - policy).sum().item()

                    value_loss = (
                        self.config.value_loss_factor
                        * self.config.value_loss_function(target_value, value)
                    )

                    if self.config.game.has_intermediate_rewards:
                        if k != 0:
                            reward_loss = self.config.reward_loss_function(
                                target_reward, reward
                            )
                        else:
                            # NO REWARD ON INITIAL OBSERVATION
                            reward_loss = torch.tensor(0.0)
                    else:
                        # print(
                        #     "Warning: for games with no intermediate rewards (board games) reward_loss is not used"
                        # )
                        reward_loss = torch.tensor(0.0)

                    if target_policy != []:
                        policy_loss = self.config.policy_loss_function(
                            target_policy, policy
                        )
                    else:
                        # THERE SHOULD PROBABLY STILL BE A TARGET POLICY LOSS ON THE UNIFORM DISTRIBUTION, THOUGH MAYBE NOT
                        policy_loss = torch.tensor(0.0)

                    scaled_loss = (
                        scale_gradient(value_loss + reward_loss + policy_loss, scale)
                        * samples["weights"][
                            item
                        ]  # TODO: COULD DO A PRIORITY/WEIGHT FUNCTION THAT INCLUDES THE RECURRENT STEPS AS, SO IT DOESNT JUST MULIPTIY BY samples["weights"][item] but samples["weights"][item][k]
                    )
                    # print("Scaled Loss ", scaled_loss)
                    val_loss += value_loss.item()
                    rew_loss += reward_loss.item()
                    pol_loss += policy_loss.item()
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
            loss.item(),
        )

    def predict_single_initial_inference(self, state, info):
        state_input = self.preprocess(state)
        value, policy, hidden_state = self.model.initial_inference(state_input)
        # should we action mask the priors?
        # legal_moves = get_legal_moves(info)
        # policy = action_mask(policy, legal_moves)
        # policy = policy / torch.sum(policy)  # Normalize policy
        return value[0], policy[0], hidden_state

    def predict_single_recurrent_inference(self, hidden_state, action):
        reward, hidden_state, value, policy = self.model.recurrent_inference(
            hidden_state, action
        )
        return reward[0], hidden_state, value[0], policy[0]

    def predict(
        self, state, info: dict = None, env=None, temperature=1.0, *args, **kwargs
    ):
        value, visit_counts = self.monte_carlo_tree_search(env, state, info)
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
        # print("Temperature probs ", probs)
        # print("Temperature probs sum to 1", torch.sum(probs))
        action = torch.multinomial(probs, 1)
        # action = np.random.choice(predictions[2], p=predictions[0].cpu().numpy())
        # print("Selected Action", action)
        return action

    def play_game(self):
        # print("Playing game")
        with torch.no_grad():
            self.env.reset()
            state, reward, terminated, truncated, info = self.env.last()
            agent_id = self.env.agent_selection
            current_player = self.env.agents.index(agent_id)
            state, info = process_petting_zoo_obs(state, info, current_player)
            game = Game(self.config.game.num_players)

            game.append(state, info)

            done = False
            while not done:
                if (
                    len(game) <= self.config.num_sampling_moves
                ):  # and (not self.is_test)
                    # print(
                    #     "Using exploration temperature",
                    #     self.config.exploration_temperature,
                    # )
                    temperature = self.config.exploration_temperature
                else:
                    # print(
                    #     "Using exploitation temperature",
                    #     self.config.exploitation_temperature,
                    # )
                    temperature = self.config.exploitation_temperature
                prediction = self.predict(
                    state, info, env=self.env, temperature=temperature
                )
                # print("State", state)
                # print("Current Player:", current_player)
                # print("Target Policy", prediction[1])
                # print("Temperature Policy ", prediction[0])
                # print("Value ", prediction[3])
                # print("Info", info)
                action = self.select_actions(prediction).item()

                # print("Action ", action)
                self.env.step(action)
                next_state, _, terminated, truncated, next_info = self.env.last()
                reward = self.env.rewards[self.env.agents[current_player]]
                agent_id = self.env.agent_selection
                current_player = self.env.agents.index(agent_id)
                next_state, next_info = process_petting_zoo_obs(
                    next_state, next_info, current_player
                )
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
                state = next_state
                info = next_info

            # game.set_rewards()
            self.replay_buffer.store(game)
        return self.env.rewards[self.env.agents[0]], game.length
