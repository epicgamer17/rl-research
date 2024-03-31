from base_agent.agent import BaseAgent
from agent_configs import Config, ConfigBase
from gymnasium import Env


class ActorAgent(BaseAgent):
    """
    Agent base class for all distributed agent/learner training setups
    """

    def __init__(self, env: Env, config: Config, name):
        super().__init__(env, config, name)

    def collect_experience(self):
        raise NotImplementedError

    def send_experience_batch(self):
        raise NotImplementedError

    def should_send_experience_batch(self, training_step: int):
        raise NotImplementedError

    def update_params(self):
        raise NotImplementedError

    def should_update_params(self, training_step: int):
        raise NotImplementedError

    def on_run_start(self):
        pass

    def on_run_end(self):
        pass

    def on_training_step_start(self):
        pass

    def on_training_step_end(self):
        pass

    def run(self):
        self.on_run_start()

        for training_step in range(self.config.training_steps + 1):
            self.on_training_step_start()
            self.collect_experience()

            if self.should_send_experience_batch(training_step):
                self.send_experience_batch()

            if self.should_update_params(training_step):
                self.update_params()

            self.on_training_step_end()

        self.on_run_end()


# Config class for PollingActor
class PollingActorConfig(ConfigBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polling_interval = self.parse_field("poll_params_interval", 10)


# ActorAgent mixin for agent that poll for new network parameters after a fixed amount of training setups
class PollingActor:
    def __init__(self, config: PollingActorConfig):
        self.config = config

    def should_update_params(self, training_step: int):
        return training_step > 0 and training_step % self.config.polling_interval == 0


# Config class for LearnerAgent
class LearnerAgent(BaseAgent):
    """
    Learner base class for all distributed actor/learner training setups
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_run_start(self):
        pass

    def on_run_end(self):
        pass

    def run_training_step(self, training_step: int):
        raise NotImplementedError

    def run(self):
        self.on_run_start()

        for training_step in range(self.config.training_steps + 1):
            self.run_training_step(training_step)

        self.on_run_end()
