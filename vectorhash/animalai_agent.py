import torch
from gymnasium import Env
from vectorhash import VectorHaSH
from skimage import color
from clean_scaffold import get_dim_distribution_from_g
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from graph_utils import plot_path, plot_probability_distribution_on_ax

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
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []

    def append(
        self,
        true_position,
        true_angle,
        x_distribution,
        y_distribution,
        theta_distribution,
        true_image,
        estimated_image,
    ):
        self._true_positions.append(true_position)
        self._x_distributions.append(x_distribution)
        self._y_distributions.append(y_distribution)
        self._theta_distributions.append(theta_distribution)
        self._true_images.append(true_image)
        self._estimated_images.append(estimated_image)

    def make_image_video(self):
        fig = plt.figure(layout="constrained", figsize=(8, 8), dpi=300)
        gs = GridSpec(6, 6, figure=fig)

        text = fig.suptitle("t=0")

        im_true_ax = fig.add_subplot(gs[0:3, 0:3])
        im_pred_ax = fig.add_subplot(gs[0:3, 3:6])
        x_dist_ax = fig.add_subplot(gs[4, 0:4])
        y_dist_ax = fig.add_subplot(gs[4, 0:4])
        theta_dist_ax = fig.add_subplot(gs[4, 0:4])
        im_true_ax.set_title("true image")
        im_pred_ax.set_title("predicted image")
        x_dist_ax.set_title("x dist")
        y_dist_ax.set_title("y dist")
        theta_dist_ax.set_title("θ dist")

        im_true_artist = im_true_ax.imshow(self._true_images[0])
        im_pred_artist = im_pred_ax.imshow(self._estimated_images[0])
        x_dist_artist = plot_probability_distribution_on_ax(
            self._x_distributions[0], x_dist_ax
        )
        y_dist_artist = plot_probability_distribution_on_ax(
            self._y_distributions[0], y_dist_ax
        )
        theta_dist_artist = plot_probability_distribution_on_ax(
            self._theta_distributions[0], theta_dist_ax
        )

        def plot_func(frame):
            im_true_artist.set_data(self._true_images[frame])
            im_pred_artist.set_data(self._estimated_images[frame])

            x_dist_artist.set_data(values=self._x_distributions[frame], edges=None)
            y_dist_artist.set_data(values=self._y_distributions[frame], edges=None)
            theta_dist_artist.set_data(
                values=self._theta_distributions[frame], edges=None
            )

            text.set_text(f"t={frame}")
            return (
                im_true_artist,
                im_pred_artist,
                x_dist_artist,
                y_dist_artist,
                theta_dist_artist,
                text,
            )

        self.ani = animation.FuncAnimation(
            fig, plot_func, len(self._estimated_images) - 1, blit=False
        )

        return self.ani

    def reset(self):
        self.ani = None
        self._true_positions = []
        self._true_angles = []
        self._x_distributions = []
        self._y_distributions = []
        self._theta_distributions = []
        self._true_images = []
        self._estimated_images = []


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

        self.vectorhash.store_memory(image.flatten().to(self.device))
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

        p = torch.Tensor([p_x, p_z]).to(self.device)
        v = torch.Tensor([v_x, v_z]).to(self.device)

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
                image.flatten().to(self.device)
            )
        )
        self.vectorhash.scaffold.g = sensory_g
        certainty = self.vectorhash.scaffold.estimate_certainty(k=5)
        if torch.sum(current_g_certainty) >= torch.sum(certainty):
            self.vectorhash.scaffold.g = self.vectorhash.scaffold.denoise(current_g)[0]
            self.vectorhash.store_memory(
                image.flatten().to(self.device), hard=self.hard
            )
        else:
            print(
                f"Certainty {certainty.round(decimals=2)}<{self.vectorhash.certainty}, not storing memory."
            )

            # self.vectorhash.store_memory(image.flatten().to(self.device), hard=True)
        # obs (84x84x3), in [0,1]
        print("HIII")
        print(self.vectorhash.scaffold.g)
        self.animal_ai_data["exact_position"] = p
        self.animal_ai_data["exact_angle"] += dtheta

        return image, p, self.animal_ai_data["exact_angle"]

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

    def test_path(self, path):
        self.history.reset()
        state, info = self.env.reset()
        img = self.postprocess_image(state)
        p, v = self.postprocess_health_pos_vel(info)

        # self.vectorhash.reset()
        self.vectorhash.store_memory(img.flatten().to(self.device))
        self.history.append(
            true_image=torch.clone(img).cpu(),
            true_angle=self.animal_ai_data["exact_angle"]
            - self.animal_ai_data["start_angle"],
            estimated_image=torch.clone(
                self.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
                    self.vectorhash.scaffold.hippocampal_from_grid(
                        self.vectorhash.scaffold.g
                    )[0]
                )[0]
            )
            .reshape(84, 84)
            .cpu(),
            true_position=torch.clone(p - self.animal_ai_data["start_position"]).cpu(),
            x_distribution=torch.clone(
                self.vectorhash.scaffold.expand_distribution(0)
            ).cpu(),
            y_distribution=torch.clone(
                self.vectorhash.scaffold.expand_distribution(1)
            ).cpu(),
            theta_distribution=torch.clone(
                self.vectorhash.scaffold.expand_distribution(2)
            ).cpu(),
        )

        errs = [self.calculate_position_err()]
        for i in range(len(path)):
            action = path[i]
            true_img, true_p, true_ang = self.step(action)
            estimated_img = (
                torch.clone(
                    self.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
                        self.vectorhash.scaffold.hippocampal_from_grid(
                            self.vectorhash.scaffold.g
                        )[0]
                    )[0]
                )
                .reshape(84, 84)
                .cpu()
            )
            self.history.append(
                true_image=torch.clone(true_img).cpu(),
                estimated_image=estimated_img,
                true_position=torch.clone(true_p).cpu(),
                true_angle=true_ang,
                x_distribution=torch.clone(
                    self.vectorhash.scaffold.expand_distribution(0)
                ).cpu(),
                y_distribution=torch.clone(
                    self.vectorhash.scaffold.expand_distribution(1)
                ).cpu(),
                theta_distribution=torch.clone(
                    self.vectorhash.scaffold.expand_distribution(2)
                ).cpu(),
            )
            if i % 100 == 0:
                print(f"Step {i}: {self.calculate_position_err()}")
        print("Final position error: ", self.calculate_position_err())
        return errs, path

    def agent_plot_path(self, path, beliefs):
        plot_path(path, beliefs, out="animalai_path.png", title="AnimalAI Path")

    def close(self):
        self.env.close()
