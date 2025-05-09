import math
import torch
from gymnasium import Env
from vectorhash import VectorHaSH
from skimage import color
from animalai_agent_history import (
    VectorhashAgentHistoryWithCertainty,
    VectorhashAgentKidnappedHistoryWithCertainty,
)


class MiniworldVectorhashAgent:
    def __init__(
        self,
        vectorhash: VectorHaSH,
        env: Env,
        world_size: torch.Tensor = torch.Tensor([40, 40, 360]),
        hard_store: bool = True,
        store_new: bool = True,
        additive_shift: bool = True,
    ):
        self.env = env
        self.vectorhash = vectorhash
        self.device = self.vectorhash.scaffold.device
        self.hard = hard_store
        self.world_size = world_size
        obs, info = self.env.reset()
        image, p, theta = self.postprocess_obs(obs)
        self.store_new = store_new
        self.additive_shift = additive_shift

        self.true_data = {
            "exact_angle": theta,
            "exact_position": p,
            "start_position": p,
            "start_angle": theta,
        }

        print(image.shape)
        print(image.flatten().shape)

        self.vectorhash.store_memory(image.flatten().to(self.device))
        self.previous_stored_postition = self.vectorhash.scaffold.get_mean_positions()
        self.history = VectorhashAgentHistoryWithCertainty()

    def postprocess_img(self, image):
        rescaled = image / 255
        grayscale_img = color.rgb2gray(rescaled)
        torch_img = torch.from_numpy(grayscale_img)
        return torch_img

    def postprocess_obs(self, obs):
        image = self.postprocess_obs(obs[0])
        p_x, p_y, p_z = self.env.get_wrapper_attr("agent").pos
        theta = self.env.get_wrapper_attr("agent").dir % (2 * math.pi)
        p = torch.Tensor([p_x, p_z]).to(self.device)
        return image, p, theta

    def step(self, action, noise=[]):
        """
        0 - Turn left
        1 - Turn right
        2 - Move forward
        3 - Move back
        4 to 7 - object interaction (non relevant)
        """
        obs, _, _, _, _ = self.env.step(action)
        image, p, theta = self.postprocess_obs(obs)
        if noise != []:
            if noise[2] == "normal":
                noise = torch.distributions.normal.Normal(loc=noise[0], scale=noise[1])
