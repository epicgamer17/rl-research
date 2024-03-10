import copy
import numpy as np
import tensorflow as tf
from replay_buffers.alphazero_replay_buffer import ReplayBuffer, Game
import math
from alphazero.alphazero_mcts import Node
from alphazero.alphazero_network import Network
import matplotlib.pyplot as plt
import gymnasium as gym


class AlphaZeroAgent:
    def __init__(self, env, name, config):
        self.model_name = name

        self.env = env
        self.test_env = copy.deepcopy(env)
        self.num_actions = env.action_space.n
        self.observation_dimensions = env.observation_space.shape

        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.adam_epsilon = config["adam_epsilon"]
        self.clipnorm = config["clipnorm"]
        # LEARNING RATE SCHEDULE
        self.value_loss_factor = config["value_loss_factor"]
        self.weight_decay = config["weight_decay"]

        self.training_steps = config["training_steps"]
        self.games_per_generation = config["games_per_generation"]
        self.checkpoint_interval = 5

        self.model = Network(config, self.observation_dimensions, self.num_actions)

        self.replay_batch_size = config["replay_batch_size"]
        self.replay_buffer = ReplayBuffer(
            config["replay_buffer_size"], self.replay_batch_size
        )

        self.root_dirichlet_alpha = config["root_dirichlet_alpha"]
        self.root_exploration_fraction = config["root_exploration_fraction"]
        self.num_simulations = config["num_simulations"]
        self.num_sampling_moves = config["num_sampling_moves"]
        self.initial_temperature = config["initial_temperature"]
        self.exploitation_temperature = config["exploitation_temperature"]

        self.pb_c_base = config["pb_c_base"]
        self.pb_c_init = config["pb_c_init"]

        self.is_test = False

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train(self):
        state, info = self.env.reset()
        game = Game()
        training_step = 0
        legal_moves = (
            info["legal_moves"] if "legal_moves" in info else range(self.num_actions)
        )
        while training_step < self.training_steps:
            print("Training Step ", training_step + 1)
            for _ in range(self.games_per_generation):
                done = False
                while not done:
                    visit_counts = self.monte_carlo_tree_search(
                        self.env, state, legal_moves
                    )
                    actions = [action for _, action in visit_counts]
                    visit_counts = np.array(
                        [count for count, _ in visit_counts], dtype=np.float32
                    )
                    if game.length < self.num_sampling_moves:
                        temperature = self.initial_temperature
                    else:
                        temperature = self.exploitation_temperature

                    visit_counts = np.power(visit_counts, 1 / temperature)
                    visit_counts /= np.sum(visit_counts)
                    action = np.random.choice(actions, p=visit_counts)

                    # if game.length < self.num_sampling_moves:
                    #     action = np.random.choice(
                    #         actions,
                    #         p=tf.nn.softmax(visit_counts).numpy()
                    #     )
                    # else:
                    #     action = actions[np.argmax(visit_counts)]

                    next_state, reward, terminated, truncated, info = self.step(action)
                    done = terminated or truncated
                    legal_moves = (
                        info["legal_moves"]
                        if "legal_moves" in info
                        else range(self.num_actions)
                    )
                    policy = np.zeros(self.num_actions)
                    policy[actions] = visit_counts / np.sum(visit_counts)
                    print("Target Policy", policy)
                    game.append(state, reward, policy)
                    state = next_state
                game.set_rewards()
                self.replay_buffer.store(game)
                game = Game()
                state, info = self.env.reset()
                legal_moves = (
                    info["legal_moves"]
                    if "legal_moves" in info
                    else range(self.num_actions)
                )
            loss = self.experience_replay()
            training_step += 1
            if training_step % self.checkpoint_interval == 0:
                self.test(num_trials=1)
                self.save_checkpoint()
        self.save_checkpoint()
        # save model to shared storage @Ezra

    def monte_carlo_tree_search(self, env, state, legal_moves):
        root = Node(0, env, state, legal_moves)
        value, policy = self.predict_single(state)
        print("Predicted Policy ", policy)
        print("Predicted Value ", value)
        root.to_play = int(
            state[2][0][0]
        )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT
        policy = {a: policy[a] for a in root.legal_moves}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            child_env = copy.deepcopy(env)
            child_state, reward, terminated, truncated, info = child_env.step(action)
            child_legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            root.children[action] = Node(
                p / policy_sum, child_env, child_state, child_legal_moves
            )

        actions = root.children.keys()
        noise = np.random.gamma(self.root_dirichlet_alpha, 1, len(actions))
        for a, n in zip(actions, noise):
            root.children[a].prior_policy = (
                1 - self.root_exploration_fraction
            ) * root.children[a].prior_policy + self.root_exploration_fraction * n

        for _ in range(self.num_simulations):
            node = root
            mcts_env = copy.deepcopy(env)
            search_path = [node]

            while node.expanded():
                _, action, node = max(
                    (self.ucb_score(node, child), action, child)
                    for action, child in node.children.items()
                )
                _, reward, terminated, truncated, _ = mcts_env.step(action)
                search_path.append(node)
            value, policy = self.predict_single(node.state)
            if terminated or truncated:  ###
                value = reward  ###

            node.to_play = int(
                node.state[2][0][0]
            )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT
            policy = {a: policy[a] for a in node.legal_moves}
            policy_sum = sum(policy.values())
            for action, p in policy.items():
                child_state, reward, terminated, truncated, info = mcts_env.step(action)
                child_legal_moves = (
                    info["legal_moves"]
                    if "legal_moves" in info
                    else range(self.num_actions)
                )
                node.children[action] = Node(
                    p / policy_sum, mcts_env, child_state, child_legal_moves
                )

            for node in search_path:
                node.value_sum += value if node.to_play == root.to_play else -value
                node.visits += 1

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]
        return visit_counts

    def ucb_score(self, parent, child):
        pb_c = (
            math.log((parent.visits + self.pb_c_base + 1) / self.pb_c_base)
            + self.pb_c_init
        )
        pb_c *= math.sqrt(parent.visits) / (child.visits + 1)

        prior_score = (
            pb_c * child.prior_policy * math.sqrt(parent.visits) / (child.visits + 1)
        )
        value_score = child.value()
        return prior_score + value_score

    def experience_replay(self):
        samples = self.replay_buffer.sample()
        observations = samples["observations"]
        target_policies = samples["policy"]
        target_values = samples["rewards"]
        inputs = self.prepare_states(observations)
        with tf.GradientTape() as tape:
            values, policies = self.model(inputs)
            # print("Values ", values)
            # print("Target Values ", target_values)
            # print("Policies ", policies)
            # print("Tartget Policies ", target_policies)
            value_loss = self.value_loss_factor * tf.losses.MSE(target_values, values)
            policy_loss = tf.losses.categorical_crossentropy(target_policies, policies)
            l2_loss = sum(self.model.losses)
            loss = (value_loss + policy_loss) + l2_loss
            loss = tf.reduce_mean(loss)

        print("Value Loss ", value_loss)
        print("Policy Loss ", policy_loss)
        print("L2 Loss ", l2_loss)

        print("Loss ", loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print("Gradients ", gradients)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        return loss

    def prepare_states(self, state):
        state = np.array(state)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        value, policy = self.model(inputs=state_input)
        return value.numpy()[0], policy.numpy()[0]

    def save_checkpoint(self, episode=-1, best_model=False):
        if episode != -1:
            path = "./{}_{}_episodes.keras".format(
                self.model_name, episode + self.start_episode
            )
        else:
            path = "./{}.keras".format(self.model_name)

        if best_model:
            path = "./best_model.keras"

        self.model.save(path)

    def plot_graph(self, score, loss, test_score, step):
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 5))
        ax1.plot(score, linestyle="solid")
        ax1.set_title("Frame {}. Score: {}".format(step, np.mean(score[-10:])))
        ax2.plot(loss, linestyle="solid")
        ax2.set_title("Frame {}. Loss: {}".format(step, np.mean(loss[-10:])))
        ax3.plot(test_score, linestyle="solid")
        # ax3.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
        ax3.set_title(
            "Frame {}. Test Score: {}".format(step, np.mean(test_score[-10:]))
        )
        plt.savefig("./{}.png".format(self.model_name))
        plt.close(fig)

    def test(self, num_trials=100, video_folder="") -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0

        state, info = self.test_env.reset()
        legal_moves = (
            info["legal_moves"] if "legal_moves" in info else range(self.num_actions)
        )
        for trials in range(num_trials):
            done = False
            score = 0
            test_game_moves = []
            while not done:
                visit_counts = self.monte_carlo_tree_search(
                    self.test_env, state, legal_moves
                )
                actions = [action for _, action in visit_counts]
                visit_counts = np.array(
                    [count for count, _ in visit_counts], dtype=np.float32
                )
                action = actions[np.argmax(visit_counts)]
                test_game_moves.append(action)
                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                legal_moves = (
                    info["legal_moves"]
                    if "legal_moves" in info
                    else range(self.num_actions)
                )
                state = next_state
                score += reward
            state, info = self.test_env.reset()
            legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            average_score += score
            print("score: ", score)

        if video_folder == "":
            video_folder = "./videos/{}".format(self.model_name)

        video_test_env = copy.deepcopy(self.test_env)
        video_test_env.reset()
        video_test_env = gym.wrappers.RecordVideo(video_test_env, video_folder)
        for move in test_game_moves:
            video_test_env.step(move)
        video_test_env.close()

        # reset
        self.is_test = False
        average_score /= num_trials
        return average_score
