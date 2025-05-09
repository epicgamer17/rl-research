import torch
from skimage import color
from agent import VectorhashAgent


class AnimalAIVectorhashAgent(VectorhashAgent):
    """
    Actions:
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

    def postprocess_image(self, image):
        grayscale_img = color.rgb2gray(image)
        torch_img = torch.from_numpy(grayscale_img)
        return torch_img

    def postprocess_health_pos_vel(self, data):
        health, velocity, position = data[1], data[1:4], data[4:7]
        p_x, p_y, p_z = (
            position  # x,z dimensions are typical forward/back/left/right, y dimension is up/down
        )
        theta = 0

        p = torch.Tensor([p_x, p_z, theta]).to(self.device)

        return p

    def _get_world_size(self, env):
        return torch.Tensor(40, 40, 360)

    def _env_reset(self, env):
        obs, info = env.reset()
        img = self.postprocess_image(obs)
        p = self.postprocess_health_pos_vel(info)
        return img, p

    def _obs_postpreprocess(self, step_tuple, action):
        obs, reward, done, info = step_tuple
        image = self.postprocess_image(obs[0])
        p = self.postprocess_health_pos_vel(obs[1])
        prev_theta = self.true_data.true_position[2]
        if action == 1 or action == 4 or action == 7:
            new_theta = prev_theta + 6
        elif action == 2 or action == 5 or action == 8:
            new_theta = prev_theta - 6
        else:
            new_theta = prev_theta
        p[2] = new_theta
        return image, p
