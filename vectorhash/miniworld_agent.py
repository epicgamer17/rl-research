import numpy as np
import math
import torch
from agent import VectorhashAgent
from skimage import color
from miniworld.entity import Agent
import copy


class MiniworldVectorhashAgent(VectorhashAgent):
    """
    Actions:
    - 0: turn left
    - 1: turn right
    - 2: forward
    - 3: back
    - 4-7: object interaction, not used
    """

    def postprocess_img(self, image):
        # rescaled = image / 255
        # grayscale_img = color.rgb2gray(rescaled)
        # torch_img = torch.from_numpy(grayscale_img)
        return image

    def get_true_pos(self, env):
        p_x, p_y, p_z = env.get_wrapper_attr("agent").pos
        angle = env.get_wrapper_attr("agent").dir
        p = torch.tensor([p_x, p_z, angle]).float().to(self.device)
        return p

    def set_agent_pos(self, pos):
        agent: Agent = self.env.get_wrapper_attr("agent")
        agent_copy = copy.deepcopy(agent)
        agent_copy.pos = np.array([pos[0], 0, pos[1]])
        agent_copy.dir = pos[2]
        self.env.set_wrapper_attr("agent", agent_copy)

    def _get_world_size(self, env):
        min_x = env.get_wrapper_attr("min_x")
        max_x = env.get_wrapper_attr("max_x")
        min_z = env.get_wrapper_attr("min_z")
        max_z = env.get_wrapper_attr("max_z")

        return torch.tensor([max_x - min_x, max_z - min_z, 2 * math.pi]).float()

    def _env_reset(self, env):
        obs, info = env.reset()
        img = self.postprocess_img(obs)
        p = self.get_true_pos(env)
        return img, p

    def _obs_postpreprocess(self, step_tuple, action):
        obs, reward, terminated, truncated, info = step_tuple
        img = self.postprocess_img(obs)
        p = self.get_true_pos(self.env)
        return img, p
