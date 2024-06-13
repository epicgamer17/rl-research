from typing import NamedTuple, Any
import torch

from base_agent.agent import BaseAgent
from agent_configs import Config, ConfigBase
from gymnasium import Env
from utils import get_legal_moves
import logging

logger = logging.getLogger(__name__)



class DistreteTransition(NamedTuple):
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    legal_moves: list


class ActorAgent(BaseAgent):
    """
    Agent base class for all distributed agent/learner training setups
    """

    def __init__(self, env: Env, config: Config, name):
        super().__init__(env, config, name)
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def setup(self):
        """This method is called before starting of the training/collecting experiences loop.
        It is used for things such as fetching the initial weights from the learner.
        """
        pass

    def cleanup(self, failed: bool):
        """This method is called after the training/collecting experiences loop finishes or errors.
        It is used for any cleanup that may be necessary.
        """
        pass

    def collect_experience(self, state, info) -> tuple[DistreteTransition, Any]:
        legal_moves = None # sget_legal_moves(info)
        values = self.predict(state)
        action = self.select_actions(values).item()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated

        t = (state, action, reward, next_state, done, legal_moves)
        return DistreteTransition(*t), info

    def on_experience_collected(self, t):
        pass

    def on_training_step_start(self, training_step: int):
        pass

    def on_training_step_end(self, training_step: int):
        pass

    def learn(self):
        with torch.no_grad():
            failed = False
            try:
                self.setup()
                state, info = self.env.reset()

                for training_step in range(self.config.training_steps + 1):
                    if self.stop_flag:
                        break
                    t, info = self.collect_experience(state, info)
                    self.on_experience_collected(t)

                    if t.done:
                        state, info = self.env.reset()

                    self.on_training_step_end(training_step)
            except Exception as e:
                logger.exception(e)
                failed = True
                pass
            finally:
                self.env.close()
                self.cleanup(failed)


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
