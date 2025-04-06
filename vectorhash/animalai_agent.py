import torch
import numpy as np
from gymnasium import Env
from vectorhash import VectorHaSH
from skimage import color
from clean_scaffold import get_dim_distribution_from_g

_epsilon = 1e-8
def categorical_crossentropy(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    # print(predicted)
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    # print(predicted)
    predicted = torch.clamp(predicted, _epsilon, 1.0 - _epsilon)
    # print(predicted)
    log_prob = torch.log(predicted)
    return -torch.sum(log_prob * target, axis=axis)

class AnimalAIVectorhashAgent:
    def __init__(self, vectorhash: VectorHaSH, env: Env):
        self.env = env
        self.vectorhash = vectorhash
        self.device = self.vectorhash.scaffold.device

        obs, info = self.env.reset()
        image, p, v = self.postprocess_obs(obs)

        self.animal_ai_data = {
            "exact_angle": 0,
            "exact_position": p,
            "start_position": p,
            "start_angle": 0,
        }

        self.vectorhash.store_memory()

    def postprocess_obs(self, obs):
        image = obs[0]
        health, velocity, position = obs[1][1], obs[1][1:4], obs[1][4:7]
        p_x, p_y, p_z = (
            position  # x,z dimensions are typical forward/back/left/right, y dimension is up/down
        )
        v_x, v_y, v_z = velocity

        p = np.array([p_x, p_z])
        v = np.array([v_x, v_z])

        grayscale_img = color.rgb2gray(image)
        torch_img = torch.from_numpy(grayscale_img)
        return torch_img, p, v

    def step(self, action):
        # 0 - nothing
        # 1 - rotate right by 6 degrees
        # 2 - rotate left by 6 degrees
        # 3 - accelerate forward
        # 4 - accelerate forward and rotate CW by 6 degrees
        # 5 - accelerate forward and rotate CCW by 6 degrees
        # 6 - accelerate backward
        # 7 - accelerate backward and rotate CW by 6 degrees
        # 8 - accelerate backward and rotate CCW by 6 degrees
        obs, reward, done, info = self.env.step(action)
        image, p, v = self.postprocess_obs(obs)

        dtheta = 0
        if action == 1 or action == 4 or action == 4:
            dtheta = 6
        elif action == 2 or action == 7 or action == 8:
            dtheta = 6
        noisy_dtheta = dtheta  # + random_noise

        dp = p - self.animal_ai_data["exact_position"]
        noisy_dp = dp  # + random noise

        self.vectorhash.scaffold.shift(
            torch.tensor([noisy_dp[0], noisy_dp[1], noisy_dtheta], device=self.device)
        )

        certainty = self.vectorhash.scaffold.estimate_certainty(k=5)
        if certainty >= self.vectorhash.certainty:
            self.vectorhash.store_memory(image)
        else:
            print(
                f"Certainty {certainty.round(2)}<{self.vectorhash.certainty}, not storing memory."
            )

        # obs (84x84x3), in [0,1]

        self.animal_ai_data["exact_position"] = p
        self.animal_ai_data["exact_angle"] += dtheta

    def calculate_position_err(self):
        x_dist, y_dist, theta_dist = [
            self.vectorhash.scaffold.expand_distribution(i) for i in [0, 1, 2]
        ]

        coordinates = torch.zeros(3, device=self.vectorhash.scaffold.device)
        coordinates[0] = (
            self.animal_ai_data["exact_position"][0]
            + self.animal_ai_data["start_position"][0]
        )
        coordinates[1] = (
            self.animal_ai_data["exact_position"][1]
            + self.animal_ai_data["start_position"][1]
        )
        coordinates[2] = (
            self.animal_ai_data["exact_angle"] + self.animal_ai_data["start_angle"]
        )

        g = self.vectorhash.scaffold.grid_state_from_cartesian_coordinates(coordinates)
        x_true_dist, y_true_dist, theta_true_dist = [
            get_dim_distribution_from_g(self.vectorhash.scaffold, g, dim)
            for dim in [0, 1, 2]
        ]

        return categorical_crossentropy(x_dist, x_true_dist), categorical_crossentropy(y_dist, y_true_dist), categorical_crossentropy(theta_dist, theta_true_dist)

    def close(self):
        self.env.close()
