import torch
import numpy as np
from gymnasium import Env

from fourier_scaffold import FourierScaffold
from hippocampal_sensory_layers import HippocampalSensoryLayer
from preprocessing_cnn import PreprocessingCNN, Preprocessor
from agent_history import FourierVectorhashAgentHistory
from agent import TrueData

import copy


class CombineMethod:
    def __init__(self):
        pass

    def combine(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass

    def __str__(self) -> str:  # type:ignore
        pass


class AdditiveCombine(CombineMethod):
    def __init__(self, alpha: float) -> None:
        """ alpha = how much weight to put on sensory (second arg) """
        self.alpha = alpha

    def combine(self, P1, P2) -> torch.Tensor:  # type: ignore
        return (1 - self.alpha) * P1 + self.alpha * P2

    def __str__(self) -> str:
        return f"additive (alpha={self.alpha})"


class MultiplicativeCombine(CombineMethod):
    def __init__(self) -> None:
        pass

    def combine(self, P1, P2) -> torch.Tensor:  # type: ignore
        S = (P1 * P2.conj()).sum().abs()
        P = P1 @ P2.H
        return P / S

    def __str__(self) -> str:
        return f"multiplicative"


class FourierVectorHaSH:
    def __init__(
        self,
        scaffold: FourierScaffold,
        hippocampal_sensory_layer: HippocampalSensoryLayer,
        eps_H: float,
        eps_v: float,
        combine: CombineMethod,
    ):
        self.scaffold = scaffold
        self.hippocampal_sensory_layer = hippocampal_sensory_layer
        self.combine_method = combine
        self._layer_copy = copy.deepcopy(hippocampal_sensory_layer)

        self.eps_H = eps_H
        self.eps_v = eps_v

    def reset(self):
        self.hippocampal_sensory_layer = copy.deepcopy(self._layer_copy)
        self.scaffold.P = self.scaffold.zero()

    def store(self, P: torch.Tensor, s: torch.Tensor):
        h = self.scaffold.g_avg()

        self.hippocampal_sensory_layer.learn(h, s)

    def stored(self, s: torch.Tensor):
        g = self.hippocampal_sensory_layer.hippocampal_from_sensory(s)[0]
        P = torch.einsum("i,j->ij", g, g.conj())
        H = P.norm() ** 2
        print(f"H = {H:.2f}")
        return H < self.eps_H

    def combine(self, P1: torch.Tensor, P2: torch.Tensor):
        return self.combine_method.combine(P1, P2)


class FourierVectorHaSHAgent:
    def __init__(
        self,
        vectorhash: FourierVectorHaSH,
        env: Env,
        preprocessor: Preprocessor | None = None,
    ):

        self.env = env
        self.device = vectorhash.scaffold.device
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

        self.vectorhash = vectorhash
        self.world_size = self._get_world_size(env)
        start_img, start_pos = self._env_reset(env)
        self.true_data = TrueData(start_pos)
        self.vectorhash.hippocampal_sensory_layer.learn(
            self.vectorhash.scaffold.g_avg(),
            self.preprocessor.encode(start_img),
        )

        self.v = torch.zeros(self.vectorhash.scaffold.d, device=self.device)

    def _get_world_size(self, env: Env):
        """Get the world size of the environment"""
        pass

    def _env_reset(self, env: Env) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """Do environment-specific work to reset the environment

        Returns a tuple `(initial_image, initial_position)`
        """
        pass

    def _obs_postpreprocess(
        self, step_tuple, action
    ) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """Do environment-specific work to postprocess the tuple returned by env.step()

        Returns a tuple `(new_image, new_position)`
        """
        pass

    def error(self, true_pos):
        pass

    def step(self, action, noise_dist: torch.distributions.Distribution | None):
        ### env-specific observation processing
        step_tuple = self.env.step(action)

        ### this is the sensory input not flattened yet
        new_img, new_pos = self._obs_postpreprocess(step_tuple, action)

        ### calculation of noisy input
        dp = new_pos - self.true_data.true_position
        self.true_data.true_position = new_pos
        noisy_dp = new_pos
        if noise_dist != None:
            noisy_dp += noise_dist.sample(noisy_dp.shape).to(self.device)

        dt = 1
        v = (dp / dt) * self.vectorhash.scaffold.scale_factor

        self.v = self.v + v

        return new_pos, new_img, self.v

    def reset_v(self):
        self.v = torch.zeros(self.vectorhash.scaffold.d, device=self.device)


def path_test(
    agent: FourierVectorHaSHAgent,
    path: torch.Tensor,
    noise_dist: torch.distributions.Distribution | None = None,
    reshape_img_size=(30, 40),
):
    ## aliases
    scaffold = agent.vectorhash.scaffold
    hs_layer = agent.vectorhash.hippocampal_sensory_layer

    ## reset states
    history = FourierVectorhashAgentHistory()
    agent.vectorhash.reset()

    ## store initial observations
    start_img, start_pos = agent._env_reset(agent.env)

    g_avg = scaffold.g_avg()
    hs_layer.learn(h=g_avg, s=agent.preprocessor.encode(start_img))
    agent.true_data = TrueData(start_pos)

    def s_from_P(P):
        g_avg = torch.einsum("ijm,ij->m", scaffold.T_s, P)
        return hs_layer.sensory_from_hippocampal(g_avg)[0].reshape(reshape_img_size)

    def P_from_s(s):
        encoded = agent.preprocessor.encode(s)
        g = hs_layer.hippocampal_from_sensory(encoded)[0]
        P = torch.einsum("i,j->ij", g, g.conj())
        return P

    def grid_vector_from_world_vector(v):
        return (v * scaffold.scale_factor) % scaffold.grid_limits

    ## initial history store
    est_img = s_from_P(scaffold.P)
    H_o = scaffold.entropy(agent.vectorhash.scaffold.P).item()
    H_s = scaffold.entropy(P_from_s(start_img)).item()

    history.append(
        P=scaffold.P,
        true_image=agent.preprocessor.encode(start_img).reshape(reshape_img_size),
        estimated_image=est_img,
        entropy_odometry=H_o,
        entropy_sensory=H_s,
        true_position=grid_vector_from_world_vector(
            agent.true_data.get_relative_true_pos()
        ),
        scaffold=scaffold,
    )

    for i, action in enumerate(path):
        new_pos, new_img, v = agent.step(action, noise_dist)
        if v.norm(p=float("inf")) < agent.vectorhash.eps_v:
            history.append(
                P=None,
                estimated_image=None,
                entropy_odometry=None,
                entropy_sensory=None,
                true_position=grid_vector_from_world_vector(
                    agent.true_data.get_relative_true_pos()
                ),
                true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
                scaffold=scaffold,
            )

            continue

        print(f"t={i}, shift={v}")
        scaffold.velocity_shift(v)
        scaffold.smooth()
        H_o = scaffold.entropy(scaffold.P).item()
        P_s = P_from_s(new_img)
        H_s = scaffold.entropy(P_s).item()
        if H_s > 0.5 and H_s < agent.vectorhash.eps_H:
            print(f"t={i}, combine, H_s={H_s:.3f}")
            # we have been here before
            scaffold.P = agent.vectorhash.combine(scaffold.P, P_from_s(new_img))

        scaffold.sharpen()
        hs_layer.learn(h=scaffold.g_avg(), s=agent.preprocessor.encode(new_img))
        agent.reset_v()

        history.append(
            P=scaffold.P,
            true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
            estimated_image=s_from_P(scaffold.P),
            entropy_odometry=H_o,
            entropy_sensory=H_s,
            true_position=grid_vector_from_world_vector(
                agent.true_data.get_relative_true_pos()
            ),
            scaffold=scaffold,
        )

    return history, path


def kidnap_test(
    agent: FourierVectorHaSHAgent,
    pre_kidnap_path: torch.Tensor,
    post_kidnap_path: torch.Tensor,
    kidnap_pos: list[float],
    kidnap_dir: float,
    noise_dist: torch.distributions.Distribution | None = None,
    reshape_img_size=(30, 40),
):
    ## aliases
    scaffold = agent.vectorhash.scaffold
    hs_layer = agent.vectorhash.hippocampal_sensory_layer

    ## reset states
    history = FourierVectorhashAgentHistory()
    agent.vectorhash.reset()

    ## store initial observations
    start_img, start_pos = agent._env_reset(agent.env)

    hs_layer.learn(h=scaffold.g_avg(), s=agent.preprocessor.encode(start_img))
    agent.true_data = TrueData(start_pos)

    def kidnap():
        env_agent = agent.env.get_wrapper_attr("agent")
        agent_copy = copy.deepcopy(env_agent)
        agent_copy.pos = np.array(kidnap_pos)
        agent_copy.dir = kidnap_dir
        agent.env.set_wrapper_attr("agent", agent_copy)

    def s_from_P(P):
        g_avg = torch.einsum("ijm,ij->m", scaffold.T_s, P)
        return hs_layer.sensory_from_hippocampal(g_avg)[0].reshape(reshape_img_size)

    def P_from_s(s):
        encoded = agent.preprocessor.encode(s)
        g = hs_layer.hippocampal_from_sensory(encoded)[0]
        P = torch.einsum("i,j->ij", g, g.conj())
        return P

    def grid_vector_from_world_vector(v):
        return (v * scaffold.scale_factor) % scaffold.grid_limits

    ## initial history store
    est_img = s_from_P(scaffold.P)
    H_o = scaffold.entropy(agent.vectorhash.scaffold.P).item()
    H_s = scaffold.entropy(P_from_s(start_img)).item()

    history.append(
        P=agent.vectorhash.scaffold.P,
        true_image=agent.preprocessor.encode(start_img).reshape(reshape_img_size),
        estimated_image=est_img,
        entropy_odometry=H_o,
        entropy_sensory=H_s,
        true_position=grid_vector_from_world_vector(
            agent.true_data.get_relative_true_pos()
        ),
        scaffold=scaffold,
    )

    for i, action in enumerate(pre_kidnap_path):
        new_pos, new_img, v = agent.step(action, noise_dist)
        if v.norm() < agent.vectorhash.eps_v:
            history.append(
                P=None,
                estimated_image=None,
                entropy_odometry=None,
                entropy_sensory=None,
                true_position=grid_vector_from_world_vector(
                    agent.true_data.get_relative_true_pos()
                ),
                true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
                scaffold=scaffold,
            )

            continue

        print(f"t={i}, shift={v}")
        scaffold.velocity_shift(v)
        scaffold.smooth()
        H_o = scaffold.entropy(scaffold.P).item()
        P_s = P_from_s(new_img)
        H_s = scaffold.entropy(P_s).item()
        if H_s > 0.5 and H_s < agent.vectorhash.eps_H:
            print(f"t={i}, combine, H_s={H_s:.3f}")
            # we have been here before
            scaffold.P = agent.vectorhash.combine(scaffold.P, P_from_s(new_img))

        scaffold.sharpen()
        hs_layer.learn(h=scaffold.g_avg(), s=agent.preprocessor.encode(new_img))
        agent.reset_v()

        history.append(
            P=scaffold.P,
            true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
            estimated_image=s_from_P(scaffold.P),
            entropy_odometry=H_o,
            entropy_sensory=H_s,
            true_position=grid_vector_from_world_vector(
                agent.true_data.get_relative_true_pos()
            ),
            scaffold=scaffold,
        )

    kidnap()

    for i, action in enumerate(post_kidnap_path):
        new_pos, new_img, v = agent.step(action, noise_dist)
        if v.norm(p=float("inf")) < agent.vectorhash.eps_v:
            history.append(
                P=None,
                estimated_image=None,
                entropy_odometry=None,
                entropy_sensory=None,
                true_position=grid_vector_from_world_vector(
                    agent.true_data.get_relative_true_pos()
                ),
                true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
                scaffold=scaffold,
            )

            continue

        print(f"t={i}, shift={v}")
        scaffold.velocity_shift(v)
        scaffold.smooth()
        H_o = scaffold.entropy(scaffold.P).item()
        P_s = P_from_s(new_img)
        H_s = scaffold.entropy(P_s).item()
        if H_s > 0.5 and H_s < agent.vectorhash.eps_H:
            print(f"t={i}, combine, H_s={H_s:.3f}")
            # we have been here before
            scaffold.P = agent.vectorhash.combine(scaffold.P, P_from_s(new_img))

        scaffold.sharpen()
        hs_layer.learn(h=scaffold.g_avg(), s=agent.preprocessor.encode(new_img))
        agent.reset_v()

        history.append(
            P=scaffold.P,
            true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
            estimated_image=s_from_P(scaffold.P),
            entropy_odometry=H_o,
            entropy_sensory=H_s,
            true_position=grid_vector_from_world_vector(
                agent.true_data.get_relative_true_pos()
            ),
            scaffold=scaffold,
        )

    return history, pre_kidnap_path, post_kidnap_path


def trajectory_test(
    agent: FourierVectorHaSHAgent,
    velocities: torch.Tensor,
    reshape_img_size=(30, 40),
    noise_dist: torch.distributions.Distribution | None = None,
):
    ## aliases
    scaffold = agent.vectorhash.scaffold
    hs_layer = agent.vectorhash.hippocampal_sensory_layer

    ## reset states
    history = FourierVectorhashAgentHistory()
    agent.vectorhash.reset()

    ## store initial observations
    start_img, start_pos = agent._env_reset(agent.env)

    g_avg = scaffold.g_avg()
    hs_layer.learn(h=g_avg, s=agent.preprocessor.encode(start_img))
    agent.true_data = TrueData(start_pos)

    def s_from_P(P):
        g_avg = torch.einsum("ijm,ij->m", scaffold.T_s, P)
        return hs_layer.sensory_from_hippocampal(g_avg)[0].reshape(reshape_img_size)

    def P_from_s(s):
        encoded = agent.preprocessor.encode(s)
        g = hs_layer.hippocampal_from_sensory(encoded)[0]
        P = torch.einsum("i,j->ij", g, g.conj())
        return P

    def grid_vector_from_world_vector(v):
        return (v * scaffold.scale_factor) % scaffold.grid_limits

    ## initial history store
    est_img = s_from_P(scaffold.P)
    H_o = scaffold.entropy(agent.vectorhash.scaffold.P).item()
    H_s = scaffold.entropy(P_from_s(start_img)).item()

    noisy_vels = velocities.clone()
    if noise_dist is not None:
        noisy_vels += noise_dist.sample(noisy_vels.shape)

    history.append(
        P=scaffold.P,
        true_image=agent.preprocessor.encode(start_img).reshape(reshape_img_size),
        estimated_image=est_img,
        entropy_odometry=H_o,
        entropy_sensory=H_s,
        true_position=grid_vector_from_world_vector(
            agent.true_data.get_relative_true_pos()
        ),
        scaffold=scaffold,
    )

    for i, vel in enumerate(noisy_vels):
        new_pos, new_img, v = agent.step(vel, None)
        if v.norm(p=float("inf")) < agent.vectorhash.eps_v:
            history.append(
                P=None,
                estimated_image=None,
                entropy_odometry=None,
                entropy_sensory=None,
                true_position=grid_vector_from_world_vector(
                    agent.true_data.get_relative_true_pos()
                ),
                true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
                scaffold=scaffold,
            )

            continue

        print(f"t={i}, shift={v}")
        scaffold.velocity_shift(v)
        scaffold.smooth()
        H_o = scaffold.entropy(scaffold.P).item()
        P_s = P_from_s(new_img)
        H_s = scaffold.entropy(P_s).item()
        if H_s > 0.5 and H_s < agent.vectorhash.eps_H:
            print(f"t={i}, combine, H_s={H_s:.3f}")
            # we have been here before
            scaffold.P = agent.vectorhash.combine(scaffold.P, P_from_s(new_img))

        scaffold.sharpen()
        hs_layer.learn(h=scaffold.g_avg(), s=agent.preprocessor.encode(new_img))
        agent.reset_v()

        history.append(
            P=scaffold.P,
            true_image=agent.preprocessor.encode(new_img).reshape(reshape_img_size),
            estimated_image=s_from_P(scaffold.P),
            entropy_odometry=H_o,
            entropy_sensory=H_s,
            true_position=grid_vector_from_world_vector(
                agent.true_data.get_relative_true_pos()
            ),
            scaffold=scaffold,
        )

    return history, noisy_vels
