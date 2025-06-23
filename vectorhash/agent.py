import torch
from gymnasium import Env
from vectorhash import VectorHaSH
from agent_history import (
    VectorhashAgentHistory,
    VectorhashAgentKidnappedHistory,
)
from preprocessing_cnn import PreprocessingCNN, Preprocessor


class TrueData:
    def __init__(self, pos: torch.Tensor):
        self.start_position = pos
        self.true_position = pos

    def get_relative_true_pos(self):
        return self.true_position - self.start_position


class VectorhashAgent:
    def __init__(
        self,
        vectorhash: VectorHaSH,
        env: Env,
        hard_store: bool = True,
        store_new: bool = True,
        shift_method="additive",
        preprocessor: Preprocessor = None,
    ):
        assert shift_method in [
            "additive",
            "multiplicative",
        ], f"got invalid shift method {shift_method}"

        self.env = env
        self.vectorhash = vectorhash
        self.device = vectorhash.scaffold.device
        self.hard_store = hard_store
        self.store_new = store_new
        self.shift_method = shift_method
        if preprocessor == None:
            self.preprocessor = PreprocessingCNN(
                device=self.device,
                latent_dim=128,
                input_channels=3,
                target_size=(84, 84),
                model_path="resnet18_adapter.pth",
            )
        else:
            self.preprocessor = preprocessor

        self.world_size = self._get_world_size(env)
        start_img, start_pos = self._env_reset(env)
        self.vectorhash.store_memory(self.preprocessor.encode(start_img))
        self.previous_stored_position = self.vectorhash.scaffold.get_mean_positions()

        self.true_data = TrueData(start_pos)
        pass

    def _get_world_size(self, env: Env):
        """Get the world size of the environment"""
        pass

    def _env_reset(self, env: Env) -> tuple[torch.Tensor, torch.Tensor]:
        """Do environment-specific work to reset the environment

        Returns a tuple `(initial_image, initial_position)`
        """
        pass

    def _obs_postpreprocess(
        self, step_tuple, action
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Do environment-specific work to postprocess the tuple returned by env.step()

        Returns a tuple `(new_image, new_position)`
        """
        pass

    def step(
        self, action: int, limits, noise_dist: torch.distributions.Distribution = None
    ):
        """Take an environment step and apply the SLAM algorithm accordingly. Also updates internal true position attribute.

        Returns a tuple `(new_img, odometry_certainty, sensory_certainty)`
        """
        ### env-specific observation processing
        step_tuple = self.env.step(action)

        ### this is the sensory input not flattened yet
        new_img, new_p = self._obs_postpreprocess(step_tuple, action)

        ### calculation of noisy input
        dp = new_p - self.true_data.true_position
        self.true_data.true_position = new_p
        noisy_dp = new_p
        if noise_dist != None:
            noisy_dp += noise_dist.sample(3)

        dt = 1
        v = dp / dt

        ### aliases
        scaffold = self.vectorhash.scaffold
        hs_layer = self.vectorhash.hippocampal_sensory_layer

        ### odometry update and estimate certainty
        scaffold.shift(v)
        g_o = scaffold.denoise(scaffold.g, onehot=False)[0]
        odometry_certainty = scaffold.estimate_certainty(limits=limits, g=g_o)
        scaffold.modules = scaffold.modules_from_g(g_o)

        ### estimate sensory certainty

        ### the reason why we take the first element because the hippocampal from sensory fucntion takes a batch of input B and outputs a batch in this case there is no batch so the function will add a batch dimension which is 1
        g_s = scaffold.denoise(
            scaffold.grid_from_hippocampal(
                hs_layer.hippocampal_from_sensory(self.preprocessor.encode(new_img))
            )[0]
        )[0]
        sensory_certainty = scaffold.estimate_certainty(limits=limits, g=g_s)

        ### get new position distribution
        new_positions = scaffold.get_mean_positions()
        print("new positions:", new_positions)
        lims = torch.Tensor([0.5, 0.5, 6])
        if self.store_new:
            new = False
            for i in range(len(scaffold.modules[0].shape)):
                if (
                    torch.abs(new_positions[i] - self.previous_stored_position[i])
                    > lims[i]
                ):
                    new = True

        else:
            new = True

        ## sensory update
        if new:
            if self.shift_method == "additive":
                scaffold.additive_shift(new_g=g_s)
            else:
                scaffold.multiplicative_shift(new_g=g_s)

            image_scaffold = [0] * len(scaffold.modules)
            for i in range(len(scaffold.modules)):
                if sensory_certainty[i] > odometry_certainty[i]:
                    image_scaffold[i] = scaffold.modules_from_g(g_s)[i]
                else:
                    image_scaffold[i] = scaffold.modules_from_g(g_o)[i]

            self.vectorhash.store_memory(
                self.preprocessor.encode(new_img), hard=self.hard_store
            )
            self.previous_stored_position = scaffold.get_mean_positions()

        return new_img, odometry_certainty, sensory_certainty

    def calculate_position_err(self):
        estimated_relative_pos = self.vectorhash.scaffold.get_mean_positions().to("cpu")
        d = torch.abs(
            estimated_relative_pos - self.true_data.get_relative_true_pos().to("cpu")
        )
        return torch.where(d < self.world_size / 2, d, self.world_size - d)

    def close(self):
        self.env.close()


def path_test(
    agent: VectorhashAgent,
    path: torch.Tensor,
    limits: torch.Tensor,
    noise_dist: torch.distributions.Distribution = None,
):
    history = VectorhashAgentHistory()
    ## reset states
    agent.vectorhash.reset()
    agent.previous_stored_position = agent.vectorhash.scaffold.get_mean_positions()

    ## store initial observations
    start_img, start_pos = agent._env_reset(agent.env)
    agent.vectorhash.store_memory(agent.preprocessor.encode(start_img))
    agent.true_data = TrueData(start_pos)

    ## aliases
    def s_from_h_from_g(g):
        return agent.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
            agent.vectorhash.scaffold.hippocampal_from_grid(
                agent.vectorhash.scaffold.denoise(g)[0]
            )[0]
        )[0].reshape(16, 8)

    def g_from_h_from_s(s):
        agent.vectorhash.scaffold.denoise(
            agent.vectorhash.scaffold.grid_from_hippocampal(
                agent.vectorhash.hippocampal_sensory_layer.hippocampal_from_sensory(s)
            ),
        )[0]

    def translate_world_to_grid_pos(p):
        return (
            p * agent.vectorhash.scaffold.scale_factor
        ) % agent.vectorhash.scaffold.grid_limits

    ## initial history store
    est_img = s_from_h_from_g(agent.vectorhash.scaffold.g)
    certainty_o = agent.vectorhash.scaffold.estimate_certainty(
        limits, g=agent.vectorhash.scaffold.g
    )
    certainty_s = agent.vectorhash.scaffold.estimate_certainty(
        limits, g=g_from_h_from_s(agent.preprocessor.encode(start_img))
    )

    history.append(
        true_image=agent.preprocessor.encode(start_img).reshape(16, 8),
        estimated_image=est_img,
        certainty_odometry=certainty_o,
        certainty_sensory=certainty_s,
        x_distribution=agent.vectorhash.scaffold.expand_distribution(0),
        y_distribution=agent.vectorhash.scaffold.expand_distribution(1),
        theta_distribution=agent.vectorhash.scaffold.expand_distribution(2),
        true_position=translate_world_to_grid_pos(
            agent.true_data.get_relative_true_pos()
        ),
    )

    errs = [agent.calculate_position_err()]

    for i, action in enumerate(path):
        new_img, odometry_certainty, sensory_certainty = agent.step(
            action, limits, noise_dist
        )
        est_img = s_from_h_from_g(agent.vectorhash.scaffold.g)
        history.append(
            true_image=agent.preprocessor.encode(new_img).reshape(16, 8),
            estimated_image=est_img,
            certainty_odometry=odometry_certainty,
            certainty_sensory=sensory_certainty,
            x_distribution=agent.vectorhash.scaffold.expand_distribution(0),
            y_distribution=agent.vectorhash.scaffold.expand_distribution(1),
            theta_distribution=agent.vectorhash.scaffold.expand_distribution(2),
            true_position=translate_world_to_grid_pos(
                agent.true_data.get_relative_true_pos()
            ),
        )

    return errs, history, path


def kidnapping_test(
    agent: VectorhashAgent,
    path: torch.Tensor,
    visibles: torch.Tensor,
    limits: torch.Tensor,
    noise_dist: torch.distributions.Distribution = None,
):
    ## reset states
    history = VectorhashAgentKidnappedHistory()
    agent.vectorhash.reset()
    agent.previous_stored_position = agent.vectorhash.scaffold.get_mean_positions()

    ## store initial observations
    start_img, start_pos = agent._env_reset(agent.env)
    agent.vectorhash.store_memory(agent.preprocessor.encode(start_img))
    agent.true_data = TrueData(start_pos)

    ## aliases
    def s_from_h_from_g(g):
        return agent.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
            agent.vectorhash.scaffold.hippocampal_from_grid(
                agent.vectorhash.scaffold.denoise(g)[0]
            )[0]
        )[0].reshape(16, 8)

    def g_from_h_from_s(s):
        agent.vectorhash.scaffold.denoise(
            agent.vectorhash.scaffold.grid_from_hippocampal(
                agent.vectorhash.hippocampal_sensory_layer.hippocampal_from_sensory(s)
            ),
        )[0]

    def translate_world_to_grid_pos(p):
        return (
            p * agent.vectorhash.scaffold.scale_factor
        ) % agent.vectorhash.scaffold.grid_limits

    ## initial history store
    est_img = s_from_h_from_g(agent.vectorhash.scaffold.g)
    certainty_o = agent.vectorhash.scaffold.estimate_certainty(
        limits, g=agent.vectorhash.scaffold.g
    )
    certainty_s = agent.vectorhash.scaffold.estimate_certainty(
        limits, g=g_from_h_from_s(agent.preprocessor.encode(start_img))
    )
    history.append(
        true_image=start_img,
        estimated_image=est_img,
        certainty_odometry=certainty_o,
        certainty_sensory=certainty_s,
        x_distribution=agent.vectorhash.scaffold.expand_distribution(0),
        y_distribution=agent.vectorhash.scaffold.expand_distribution(1),
        theta_distribution=agent.vectorhash.scaffold.expand_distribution(2),
        true_position=translate_world_to_grid_pos(
            agent.true_data.get_relative_true_pos()
        ),
        seen=True,
    )

    for i, [action, visible] in enumerate(zip(path, visibles)):
        if visible:
            true_img, c_o, c_s = agent.step(action, limits, noise_dist)
            est_img = s_from_h_from_g(agent.vectorhash.scaffold.g)
            history.append(
                true_image=true_img,
                estimated_image=est_img,
                certainty_odometry=c_o,
                certainty_sensory=c_s,
                true_position=translate_world_to_grid_pos(
                    agent.true_data.get_relative_true_pos()
                ),
                x_distribution=agent.vectorhash.scaffold.expand_distribution(0),
                y_distribution=agent.vectorhash.scaffold.expand_distribution(1),
                theta_distribution=agent.vectorhash.scaffold.expand_distribution(2),
                seen=True,
            )
        else:
            step_tuple = agent.env.step(action)
            new_img, new_p = agent._obs_postpreprocess(step_tuple, action)
            agent.true_data.true_position = new_p

            history.append(
                true_image=new_img,
                estimated_image=None,
                true_position=translate_world_to_grid_pos(
                    agent.true_data.get_relative_true_pos()
                ),
                x_distribution=None,
                y_distribution=None,
                theta_distribution=None,
                certainty_odometry=None,
                certainty_sensory=None,
                seen=False,
            )
    return history
