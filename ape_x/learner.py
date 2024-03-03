import sys

sys.path.append("../")
import rainbow.rainbow_dqn as dqn

default_learner_config = {
    "remove_old_experiences_interval": 100,  # number of learning steps between removing old experiences from remote replay buffer
}


class LearnerBase(dqn.RainbowDQN):
    def __init__(self, env, config, learner_config=default_learner_config):
        super().__init__(model_name="learner", env=env, config=config)
        self.graph_interval = 200
        self.remove_old_experiences_interval = learner_config[
            "remove_old_experiences_interval"
        ]
        self.running = False

    def sample_experiences_from_remote_replay_buffer(self):
        pass

    def update_remote_replay_buffer_priorities(self, indices, priorities):
        pass

    def remove_old_experiences_from_remote_replay_buffer(self):
        pass

    def run(self, graph_interval=200):
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        self.fill_memory()
        num_trials_truncated = 0
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        training_step = 0
        step = 0
        while training_step < self.num_training_steps:
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if (step % self.replay_period) == 0 and (
                len(self.memory) >= self.replay_batch_size
            ):
                model_update_count += 1
                loss = self.experience_replay()
                training_step += 1
                stat_loss.append(loss)

                self.update_target_model(model_update_count)

            if training_step % graph_interval == 0 and training_step > 0:
                self.export()
                # stat_test_score.append(self.test())
                self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
            step += 1

        self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
        self.export()
        self.env.close()
        return num_trials_truncated / self.num_training_steps


class SingleMachineLearner(LearnerBase):
    def __init__(self, env, config, learner_config=default_learner_config):
        super().__init__(env=env, config=config, learner_config=learner_config)

    def sample_experiences_from_remote_replay_buffer(self):
        pass

    def update_remote_replay_buffer_priorities(self, indices, priorities):
        pass

    def remove_old_experiences_from_remote_replay_buffer(self):
        pass

    def get_weights(self):
        return self.model.get_weights()
