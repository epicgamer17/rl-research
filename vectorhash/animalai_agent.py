import torch
import numpy as np
from gymnasium import Env
from vectorhash import VectorHaSH
from skimage import color
from clean_scaffold import get_dim_distribution_from_g
from matplotlib import animation
import matplotlib.pyplot as plt
from functools import partial

_epsilon = 1e-8


def categorical_crossentropy(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    # print(predicted)
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    # print(predicted)
    predicted = torch.clamp(predicted, _epsilon, 1.0 - _epsilon)
    # print(predicted)
    log_prob = torch.log(predicted)
    return -torch.sum(log_prob * target, axis=axis)


class VectorhashAgentHistory:
    def __init__(self):
        self._true_positions = []
        self._estimated_positions = []
        self._true_images = []
        self._estimated_images = []

    def append(self, true_position, estimated_position, true_image, estimated_image):
        self._true_positions.append(true_position)
        self._estimated_positions.append(estimated_position)
        self._true_images.append(true_image)
        self._estimated_images.append(estimated_image)

    def make_image_video(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 8), dpi=600)

        text = fig.suptitle('t=0')
        a1 = ax1.imshow(self._true_images[0])
        a2 = ax2.imshow(self._estimated_images[0])

        def frames_func():
            return zip(self._true_images, self._estimated_images)

        def plot_func(frame, *fargs):
            true_img, estimated_img = fargs
            a1.set_data(true_img)
            a2.set_data(estimated_img)
            text.set_text(f't={frame}')
            return a1, a2, text
        

        self.ani = animation.FuncAnimation(fig, plot_func, frames_func, blit=False)

        return self.ani


class AnimalAIVectorhashAgent:
    def __init__(self, vectorhash: VectorHaSH, env: Env, hard_store: bool = True):
        self.env = env
        self.vectorhash = vectorhash
        self.device = self.vectorhash.scaffold.device
        self.hard = hard_store
        obs, info = self.env.reset()
        image = self.postprocess_image(obs)
        p, v = self.postprocess_health_pos_vel(info)

        self.animal_ai_data = {
            "exact_angle": 0,
            "exact_position": p,
            "start_position": p,
            "start_angle": 0,
        }

        print(image.shape)
        print(image.flatten().shape)

        self.vectorhash.store_memory(image.flatten())
        self.history = VectorhashAgentHistory()

    def postprocess_image(self, image):
        grayscale_img = color.rgb2gray(image)
        torch_img = torch.from_numpy(grayscale_img)
        return torch_img

    def postprocess_health_pos_vel(self, data):
        health, velocity, position = data[1], data[1:4], data[4:7]
        p_x, p_y, p_z = (
            position  # x,z dimensions are typical forward/back/left/right, y dimension is up/down
        )
        v_x, v_y, v_z = velocity

        p = torch.Tensor([p_x, p_z], device=self.device)
        v = torch.Tensor([v_x, v_z], device=self.device)

        return p, v

    def postprocess_obs(self, obs):
        image = self.postprocess_image(obs[0])
        p, v = self.postprocess_health_pos_vel(obs[1])
        return image, p, v

    def step(self, action):
        """
        0 - nothing

        1 - rotate right by 6 degrees

        2 - rotate left by 6 degrees

        3 - accelerate forward

        4 - accelerate forward and rotate CW by 6 degrees

        5 - accelerate forward and rotate CCW by 6 degrees

        6 - accelerate backward

        7 - accelerate backward and rotate CW by 6 degrees

        8 - accelerate backward and rotate CCW by 6 degrees
        """
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
        current_g = self.vectorhash.scaffold.g
        current_g_certainty = self.vectorhash.scaffold.estimate_certainty(k=5)
        sensory_g = self.vectorhash.scaffold.grid_from_hippocampal(
            self.vectorhash.hippocampal_sensory_layer.hippocampal_from_sensory(
                image.flatten()
            )
        )
        self.vectorhash.scaffold.g = sensory_g
        certainty = self.vectorhash.scaffold.estimate_certainty(k=5)
        if torch.sum(current_g_certainty) >= torch.sum(certainty):
            self.vectorhash.scaffold.g = self.vectorhash.scaffold.denoise(current_g)
            self.vectorhash.store_memory(image.flatten(), hard=self.hard)
        else:
            print(
                f"Certainty {certainty.round(decimals=2)}<{self.vectorhash.certainty}, not storing memory."
            )

            self.vectorhash.store_memory(image.flatten(), hard=True)
        # obs (84x84x3), in [0,1]
        print("HIII")
        print(self.vectorhash.scaffold.g)
        self.animal_ai_data["exact_position"] = p
        self.animal_ai_data["exact_angle"] += dtheta

    def calculate_position_err(self):
        x_dist, y_dist, theta_dist = [
            self.vectorhash.scaffold.expand_distribution(i) for i in [0, 1, 2]
        ]

        coordinates = torch.zeros(3, device=self.vectorhash.scaffold.device)
        coordinates[0] = (
            self.animal_ai_data["exact_position"][0]
            - self.animal_ai_data["start_position"][0]
        )
        coordinates[1] = (
            self.animal_ai_data["exact_position"][1]
            - self.animal_ai_data["start_position"][1]
        )
        coordinates[2] = (
            self.animal_ai_data["exact_angle"] + self.animal_ai_data["start_angle"]
        )

        g = self.vectorhash.scaffold.grid_state_from_cartesian_coordinates(coordinates)
        x_true_dist, y_true_dist, theta_true_dist = [
            get_dim_distribution_from_g(self.vectorhash.scaffold, g, dim)
            for dim in [0, 1, 2]
        ]

        return (
            categorical_crossentropy(x_dist, x_true_dist),
            categorical_crossentropy(y_dist, y_true_dist),
            categorical_crossentropy(theta_dist, theta_true_dist),
        )

    def close(self):
        self.env.close()
