import torch
from gymnasium import Env
from vectorhash import VectorHaSH
from skimage import color
from animalai_agent_history import (
    VectorhashAgentHistoryWithCertainty,
    VectorhashAgentKidnappedHistoryWithCertainty,
)


class AnimalAIVectorhashAgent:
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
        image = self.postprocess_image(obs)
        p, v = self.postprocess_health_pos_vel(info)
        self.store_new = store_new
        self.additive_shift = additive_shift

        self.animal_ai_data = {
            "exact_angle": 0,
            "exact_position": p,
            "start_position": p,
            "start_angle": 0,
        }

        print(image.shape)
        print(image.flatten().shape)

        self.vectorhash.store_memory(image.flatten().to(self.device))
        self.history = VectorhashAgentHistoryWithCertainty()

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

    def step(self, action, noise=[]):
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
        if noise != []:
            if noise[2] == "normal":
                noise = torch.distributions.normal.Normal(loc=noise[0], scale=noise[1])
        ### calculation of noisy input
        dtheta = 0
        if action == 1 or action == 4 or action == 4:
            dtheta = 6
        elif action == 2 or action == 7 or action == 8:
            dtheta = -6
        noisy_dtheta = dtheta  # + random_noise

        dp = p - self.animal_ai_data["exact_position"]
        noisy_dp = dp  # + random noise
        if noise != []:
            # add gaussian noise to dtheta and dp
            noisy_dtheta = dtheta + noise.sample().item()
            noisy_dp = dp + noise.sample((2,)).tolist()
        v = torch.tensor([noisy_dp[0], noisy_dp[1], noisy_dtheta], device=self.device)

        ### aliases
        scaffold = self.vectorhash.scaffold
        hs_layer = self.vectorhash.hippocampal_sensory_layer

        ### get previous position distribution
        old_positions = scaffold.get_mean_positions()
        print("old positions:", old_positions)

        ### odometry update and estimate certainty
        scaffold.shift(v)
        g_o = scaffold.denoise(scaffold.g, onehot=False)[0]
        odometry_certainty = scaffold.estimate_certainty(
            limits=torch.Tensor([2, 2, 30]), g=g_o
        )
        scaffold.modules = scaffold.modules_from_g(g_o)

        ### estimate sensory certainty
        g_s = scaffold.denoise(
            scaffold.grid_from_hippocampal(
                hs_layer.hippocampal_from_sensory(image.flatten().to(self.device))[0]
            )[0]
        )[0]
        sensory_certainty = scaffold.estimate_certainty(
            limits=torch.Tensor([2, 2, 30]), g=g_s
        )

        ### get new position distribution
        new_positions = scaffold.get_mean_positions()
        print("new positions:", new_positions)
        lims = torch.Tensor([2, 2, 10])
        if self.store_new:
            new = False
            for i in range(len(scaffold.modules[0].shape)):
                if torch.abs(new_positions[i] - old_positions[i]) > lims[i]:
                    new = True
        else:
            new = True

        ## sensory update
        if new:
            if self.additive_shift:
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
                image.flatten().to(self.device), hard=self.hard
            )

        self.animal_ai_data["exact_position"] = p
        self.animal_ai_data["exact_angle"] += dtheta

        return (
            image,
            p,
            self.animal_ai_data["exact_angle"],
            v,
            odometry_certainty,
            sensory_certainty,
        )

    def calculate_position_err(self):
        # x_dist, y_dist, theta_dist = [
        #     self.vectorhash.scaffold.expand_distribution(i) for i in [0, 1, 2]
        # ]

        # coordinates = torch.zeros(3, device=self.vectorhash.scaffold.device)
        # coordinates[0] = (
        #     self.animal_ai_data["exact_position"][0]
        #     - self.animal_ai_data["start_position"][0]
        # )
        # coordinates[1] = (
        #     self.animal_ai_data["exact_position"][1]
        #     - self.animal_ai_data["start_position"][1]
        # )
        # coordinates[2] = (
        #     self.animal_ai_data["exact_angle"] + self.animal_ai_data["start_angle"]
        # )

        # g = self.vectorhash.scaffold.grid_state_from_cartesian_coordinates(coordinates)
        # x_true_dist, y_true_dist, theta_true_dist = [
        #     get_dim_distribution_from_g(self.vectorhash.scaffold, g, dim)
        #     for dim in [0, 1, 2]
        # ]

        # return (
        #     categorical_crossentropy(x_dist, x_true_dist),
        #     categorical_crossentropy(y_dist, y_true_dist),
        #     categorical_crossentropy(theta_dist, theta_true_dist),
        # )
        true_positions = self.vectorhash.scaffold.get_mean_positions()
        coordinates = torch.zeros(3)
        coordinates[0] = (
            self.animal_ai_data["exact_position"][0]
            - self.animal_ai_data["start_position"][0]
        )
        coordinates[1] = (
            self.animal_ai_data["exact_position"][1]
            - self.animal_ai_data["start_position"][1]
        )
        coordinates[2] = (
            self.animal_ai_data["exact_angle"] - self.animal_ai_data["start_angle"]
        )

        d = torch.abs(true_positions - coordinates)
        return torch.where(
            d < self.world_size / 2,
            d,
            self.world_size - d,
        )

    def test_path(self, path, noise=[]):
        self.history.reset()
        state, info = self.env.reset()
        img = self.postprocess_image(state)
        p, v = self.postprocess_health_pos_vel(info)

        if noise != []:
            print("------------USING NOISED ODOMETRY INPUTS-----------")
            print("Mean: ", noise[0])
            print("Std: ", noise[1])
            print("Noise type: ", noise[2])

        # self.vectorhash.reset()
        self.vectorhash.store_memory(img.flatten().to(self.device))

        ## initial history store
        estimated_image = (
            self.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
                self.vectorhash.scaffold.hippocampal_from_grid(
                    self.vectorhash.scaffold.denoise(self.vectorhash.scaffold.g)[0]
                )[0]
            )[0].reshape(84, 84)
        )
        limits = torch.Tensor([2, 2, 30])
        certainty_o = self.vectorhash.scaffold.estimate_certainty(
            limits, g=self.vectorhash.scaffold.g
        )
        certainty_s = self.vectorhash.scaffold.estimate_certainty(
            limits,
            g=self.vectorhash.scaffold.denoise(
                self.vectorhash.scaffold.grid_from_hippocampal(
                    self.vectorhash.hippocampal_sensory_layer.hippocampal_from_sensory(
                        img.flatten().to(self.device)
                    )
                ),
            )[0],
        )
        self.history.append(
            true_image=img,
            true_angle=(
                (
                    (
                        self.animal_ai_data["exact_angle"]
                        - self.animal_ai_data["start_angle"]
                    )
                    * self.vectorhash.scaffold.scale_factor[2]
                )
                % self.vectorhash.scaffold.grid_limits[2]
            ).item(),
            estimated_image=estimated_image,
            true_position=(
                (
                    (p - self.animal_ai_data["start_position"]).cpu()
                    * torch.tensor(
                        [
                            self.vectorhash.scaffold.scale_factor[0],
                            self.vectorhash.scaffold.scale_factor[1],
                        ]
                    )
                )
                % torch.tensor(
                    [
                        self.vectorhash.scaffold.grid_limits[0],
                        self.vectorhash.scaffold.grid_limits[1],
                    ]
                )
            ).cpu(),
            x_distribution=self.vectorhash.scaffold.expand_distribution(0),
            y_distribution=self.vectorhash.scaffold.expand_distribution(1),
            theta_distribution=self.vectorhash.scaffold.expand_distribution(2),
            certainty_odometry=certainty_o,
            certainty_sensory=certainty_s,
        )

        errs = [self.calculate_position_err()]
        for i in range(len(path)):
            action = path[i]
            true_img, true_p, true_ang, v, certainty_o, certainty_s = self.step(
                action, noise=noise
            )
            estimated_img = (
                self.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
                    self.vectorhash.scaffold.hippocampal_from_grid(
                        self.vectorhash.scaffold.g
                    )[0]
                )[0].reshape(84, 84)
            )
            self.history.append(
                true_image=true_img,
                estimated_image=estimated_img,
                true_position=(
                    (
                        (true_p - self.animal_ai_data["start_position"]).cpu()
                        * torch.tensor(
                            [
                                self.vectorhash.scaffold.scale_factor[0],
                                self.vectorhash.scaffold.scale_factor[1],
                            ]
                        )
                    )
                    % torch.tensor(
                        [
                            self.vectorhash.scaffold.grid_limits[0],
                            self.vectorhash.scaffold.grid_limits[1],
                        ]
                    )
                ).cpu(),
                true_angle=(
                    (
                        (true_ang - self.animal_ai_data["start_angle"])
                        * self.vectorhash.scaffold.scale_factor[2]
                    )
                    % self.vectorhash.scaffold.grid_limits[2]
                ).item(),
                x_distribution=self.vectorhash.scaffold.expand_distribution(0),
                y_distribution=self.vectorhash.scaffold.expand_distribution(1),
                theta_distribution=self.vectorhash.scaffold.expand_distribution(2),
                certainty_odometry=certainty_o,
                certainty_sensory=certainty_s,
            )
            if i % 100 == 0:
                print(f"Step {i}: {self.calculate_position_err()}")
        print("Final position error: ", self.calculate_position_err())
        return errs, path

    def close(self):
        self.env.close()


def kidnapping_test(
    agent: AnimalAIVectorhashAgent,
    path: torch.Tensor,
    noise_list,
    visible: torch.Tensor,
):
    history = VectorhashAgentKidnappedHistory()

    for action, noise, visible in zip(path, noise_list, visible):
        if visible:
            true_img, true_p, true_ang, v, certainty_o, certainty_s = agent.step(
                action, noise=noise
            )

            estimated_img = (
                agent.vectorhash.hippocampal_sensory_layer.sensory_from_hippocampal(
                    agent.vectorhash.scaffold.hippocampal_from_grid(
                        agent.vectorhash.scaffold.g
                    )[0]
                )[0].reshape(84, 84)
            )
            history.append(
                true_image=true_img,
                estimated_image=estimated_img,
                true_position=(
                    (
                        (true_p - agent.animal_ai_data["start_position"]).cpu()
                        * torch.tensor(
                            [
                                agent.vectorhash.scaffold.scale_factor[0],
                                agent.vectorhash.scaffold.scale_factor[1],
                            ]
                        )
                    )
                    % torch.tensor(
                        [
                            agent.vectorhash.scaffold.grid_limits[0],
                            agent.vectorhash.scaffold.grid_limits[1],
                        ]
                    )
                ).cpu(),
                true_angle=(
                    (
                        (true_ang - agent.animal_ai_data["start_angle"])
                        * agent.vectorhash.scaffold.scale_factor[2]
                    )
                    % agent.vectorhash.scaffold.grid_limits[2]
                ).item(),
                x_distribution=agent.vectorhash.scaffold.expand_distribution(0),
                y_distribution=agent.vectorhash.scaffold.expand_distribution(1),
                theta_distribution=agent.vectorhash.scaffold.expand_distribution(2),
                seen=True,
            )

        else:
            obs, reward, done, info = agent.env.step(action)
            image, p, v = agent.postprocess_obs(obs)

            ### update AAI data
            dtheta = 0
            if action == 1 or action == 4 or action == 4:
                dtheta = 6
            elif action == 2 or action == 7 or action == 8:
                dtheta = -6

            agent.animal_ai_data["exact_position"] = p
            agent.animal_ai_data["exact_angle"] += dtheta

            ### history
            history.append(
                true_image=image,
                estimated_image=None,
                true_position=(
                    (
                        (p - agent.animal_ai_data["start_position"]).cpu()
                        * torch.tensor(
                            [
                                agent.vectorhash.scaffold.scale_factor[0],
                                agent.vectorhash.scaffold.scale_factor[1],
                            ]
                        )
                    )
                    % torch.tensor(
                        [
                            agent.vectorhash.scaffold.grid_limits[0],
                            agent.vectorhash.scaffold.grid_limits[1],
                        ]
                    )
                ).cpu(),
                true_angle=(
                    (
                        (
                            agent.animal_ai_data["exact_angle"]
                            - agent.animal_ai_data["start_angle"]
                        )
                        * agent.vectorhash.scaffold.scale_factor[2]
                    )
                    % agent.vectorhash.scaffold.grid_limits[2]
                ).item(),
                x_distribution=None,
                y_distribution=None,
                theta_distribution=None,
                seen=False,
            )

    return history
