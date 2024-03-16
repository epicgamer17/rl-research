import sys

sys.path.append("../")
from rainbow.rainbow_agent import RainbowAgent


class LearnerBase(RainbowAgent):
    def __init__(self, env, config):
        super().__init__(model_name="learner", env=env, config=config)
        self.graph_interval = 200
        self.remove_old_experiences_interval = config["remove_old_experiences_interval"]
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
        self.fill_replay_buffer()
        num_trials_truncated = 0
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        training_step = 0
        step = 0
        while training_step < self.num_training_steps:
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if (step % self.replay_period) == 0 and (
                len(self.replay_buffer) >= self.replay_batch_size
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
    def __init__(self, env, config):
        super().__init__(env=env, config=config)

    def sample_experiences_from_remote_replay_buffer(self):
        return self.replay_buffer.sample()

    def update_remote_replay_buffer_priorities(self, indices, priorities):
        pass

    def remove_old_experiences_from_remote_replay_buffer(self):
        pass

    def get_weights(self):
        return self.model.get_weights()
