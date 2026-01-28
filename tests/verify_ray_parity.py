import ray
import torch
import numpy as np
import random
import unittest
import sys
import os
import gymnasium as gym

# Add project root to path
sys.path.append(os.getcwd())

from agents.muzero import MuZeroWorker
from agents.muzero_tmp import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig


# Mock Environment for Determinism
class MockEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, 32, 32), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.agents = ["player_0"]
        self.possible_agents = ["player_0"]
        self.steps = 0
        self.max_steps = 2

    def reset(self):
        self.steps = 0
        return np.zeros((3, 32, 32), dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False
        reward = 1.0
        obs = np.random.randn(3, 32, 32).astype(np.float32)  # Random but seeded
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


# Mock Config
class MockGame:
    def __init__(self):
        self.num_players = 1
        self.observation_space = (3, 32, 32)
        self.action_space = 4
        self.is_discrete = True
        self.is_image = True
        self.has_legal_moves = False
        self.is_deterministic = True

    def make_env(self):
        return MockEnv()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from modules.world_models.muzero_world_model import MuzeroWorldModel


class TestRayParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_parity(self):
        seed = 42
        set_seed(seed)

        # Config
        config = MuZeroConfig(
            config_dict={"world_model_cls": MuzeroWorldModel}, game_config={}
        )
        config.game = MockGame()
        config.self_play_delay = 0
        config.training_delay = 0
        config.minibatch_size = 2  # Small batch
        config.reanalyze_ratio = 0.0  # No reanalyze for simple play test
        config.use_quantization = False
        config.qat = False
        config.compile = False  # Disable compile for simplicity
        # Ensure deterministic selection
        config.temperatures = [0.0]
        config.temperature_updates = []
        config.temperature_with_training_steps = False
        config.num_simulations = 1
        config.root_exploration_fraction = 0.0  # Critical for determinism
        config.add_exploration_noise = False

        # 1. Instantiate MuZeroAgent (Legacy)
        set_seed(seed)  # Reset seed before init
        agent = MuZeroAgent(
            env=MockEnv(), config=config, device="cpu", name="legacy_agent"
        )

        # 2. Instantiate MuZeroWorker (Ray)
        set_seed(
            seed
        )  # Reset seed before init (though worker init happens in remote process)
        # Note: Worker initializes its own network remotely. We MUST check/sync weights.
        worker = MuZeroWorker.remote(
            config=config,
            worker_id=0,
            model_name="ray_worker",
            checkpoint_interval=100,
            device="cpu",
        )

        # 3. Sync Weights (Critical for parity)
        weights = agent.model.state_dict()
        # Transfer weights to worker
        ray.get(worker.set_weights.remote(weights, 0))

        # 4. Run Play Game
        # We need to ensure the environment interaction is identical.
        # Since MockEnv uses np.random, we need to seed the ENV inside the worker process logic
        # or rely on the fact that if inputs (obs) are identical, deterministic model outputs identical actions.
        # But MockEnv generates random observations.
        # FIX: The MockEnv inside the worker will have a different random state than the MockEnv here.
        # To strictly verify parity, we should supply a fixed environment or mocked observations?
        # Actually, if we set the seed inside the worker via a remote call or assume Ray isn't actively reseeding...
        # Let's try to trust the inputs.
        # Alternatively, we can pass `env` to play_game?
        # MuZeroAgent.play_game(env=...)
        # MuZeroWorker.play_game(env=...)

        # Let's use a Deterministic Mock Env
        class DeterministicMockEnv(MockEnv):
            def step(self, action):
                self.steps += 1
                terminated = self.steps >= self.max_steps
                obs = np.full((3, 32, 32), 0.1 * self.steps, dtype=np.float32)
                return obs, 1.0, terminated, False, {}

            def reset(self):
                self.steps = 0
                return np.zeros((3, 32, 32), dtype=np.float32), {}

        det_env_agent = DeterministicMockEnv()

        # Run Agent
        print("Running Agent...")
        score_agent, steps_agent = agent.play_game(env=det_env_agent)

        # Run Worker
        # We need to pass the SAME environment logic. MuZeroWorker creates its own env in __init__.
        # But play_game accepts an `env` argument.
        # However, we can't pass a local object (det_env) to a remote actor easily if it's not picklable or if we want state sharing.
        # But we CAN pass a copy of it.
        # Better: let's rely on the worker's internal env IF we can make it deterministic.
        # The worker uses `config.game.make_env()`.
        # We overrode config.game with MockGame which returns MockEnv.
        # Let's override MockGame to return DeterministicMockEnv.

        config.game.make_env = lambda: DeterministicMockEnv()

        # Re-init worker to get the new env? Or just pass env to play_game.
        # Passing env to remote play_game works because it's pickled.
        det_env_worker = DeterministicMockEnv()
        print("Running Worker...")
        # Note: play_game returns (score, steps, game)
        future = worker.play_game.remote(env=det_env_worker)
        score_worker, steps_worker, game_worker = ray.get(future)

        print(f"Agent: Score={score_agent}, Steps={steps_agent}")
        print(f"Worker: Score={score_worker}, Steps={steps_worker}")

        self.assertEqual(score_agent, score_worker, "Scores should match")
        self.assertEqual(steps_agent, steps_worker, "Steps should match")

        # Check actions
        # MuZeroAgent stores actions in its replay buffer tensors.
        # Since this is a fresh agent and one game, actions are at indices 0 to steps_agent.
        agent_actions_tensor = agent.replay_buffer.buffers["actions"]
        # Note: buffer stores floats/ints depending on config, usually float for actions
        agent_actions = agent_actions_tensor[0:steps_agent].flatten().cpu().numpy()

        # Worker returns the game object which has actions list
        worker_actions = np.array(game_worker.action_history)

        print(f"Agent Actions: {agent_actions}")
        print(f"Worker Actions: {worker_actions}")

        # Verify length
        self.assertEqual(
            len(agent_actions), len(worker_actions), "Action counts should match"
        )

        # Compare values (allow for minor type diffs e.g. float vs int)
        np.testing.assert_allclose(
            agent_actions,
            worker_actions,
            rtol=1e-5,
            err_msg="Actions should be identical",
        )


if __name__ == "__main__":
    unittest.main()
