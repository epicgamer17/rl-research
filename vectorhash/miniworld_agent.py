import math
import torch
from agent import VectorhashAgent
from skimage import color


class MiniworldVectorhashAgent(VectorhashAgent):
    def postprocess_img(self, image):
        rescaled = image / 255
        grayscale_img = color.rgb2gray(rescaled)
        torch_img = torch.from_numpy(grayscale_img)
        return torch_img

    def get_true_pos(env):
        p_x, p_y, p_z = env.get_wrapper_attr("agent").pos
        angle = env.get_wrapper_attr("agent").dir
        p = torch.tensor([p_x, p_z, angle])
        return p

    def _get_world_size(self, env):
        min_x = env.get_wrapper_attr("min_x")
        max_x = env.get_wrapper_attr("max_x")
        min_z = env.get_wrapper_attr("min_z")
        max_z = env.get_wrapper_attr("max_z")

        return torch.tensor([max_x - min_x, max_z - min_z, 2 * math.pi])

    def _env_reset(self, env):
        obs, info = env.reset()
        img = self.postprocess_img(obs)
        p = self.get_true_pos()
        return img, p

    def _obs_postpreprocess(self, step_tuple, action):
        obs, reward, terminated, truncated, info = step_tuple
        img = self.postprocess_img(obs)
        p = self.get_true_pos()
        return img, p
