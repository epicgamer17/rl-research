import torch

from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DomainParams


import sys

sys.path.append("../..")
from fourier_vectorhash import FourierVectorHaSHAgent

EXP_PARAMS = DomainParams()
EXP_PARAMS.set("sky_color", [0.25, 0.82, 1])
EXP_PARAMS.set("light_pos", [0, 2.5, 0])
EXP_PARAMS.set("light_color", [0.7, 0.7, 0.7])
EXP_PARAMS.set("light_ambient", [0.45, 0.45, 0.45])
EXP_PARAMS.set("obj_color_bias", [0, 0, 0])
EXP_PARAMS.set("forward_step", 0.1)
EXP_PARAMS.set("forward_drift", 0)
EXP_PARAMS.set("turn_step", 3)
EXP_PARAMS.set("bot_radius", 0.4)
EXP_PARAMS.set("cam_pitch", 0)
EXP_PARAMS.set("cam_fov_y", 60)
EXP_PARAMS.set("cam_height", 1.5)
EXP_PARAMS.set("cam_fwd_disp", 0)


class RoomExperiment(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment in which the goal is to go to a red box placed randomly in one big room.
    The `OneRoom` environment has two variants. The `OneRoomS6` environment gives you
    a room with size 6 (the `OneRoom` environment has size 10). The `OneRoomS6Fast`
    environment also is using a room with size 6, but the turning and moving motion
    is larger.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-OneRoom-v0")
    # or
    env = gymnasium.make("MiniWorld-OneRoomS6-v0")
    # or
    env = gymnasium.make("MiniWorld-OneRoomS6Fast-v0")
    ```
    """

    def __init__(
        self, start_pos, start_angle, place_red_box=True, place_blue_box=True, **kwargs
    ):
        self.start_pos = start_pos
        self.start_angle = start_angle
        self.place_red_box = place_red_box
        self.place_blue_box = place_blue_box

        MiniWorldEnv.__init__(
            self, max_episode_steps=-1, domain_rand=False, params=EXP_PARAMS, **kwargs
        )
        utils.EzPickle.__init__(
            self, max_episode_steps=-1, domain_rand=False, params=EXP_PARAMS, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=10, min_z=0, max_z=10)

        if self.place_red_box:
            self.red_box = self.place_entity(Box(color="red"), pos=[8, 0, 1.5])
        if self.place_blue_box:
            self.blue_box = self.place_entity(Box(color="blue"), pos=[1.5, 0, 8])

        self.place_agent(dir=self.start_angle, pos=self.start_pos)

    def kidnap(self, kidnap_pos, kidnap_dir):
        self.agent.pos = kidnap_pos
        self.agent.dir = kidnap_dir

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.red_box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


class RoomAgent(FourierVectorHaSHAgent):
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

    def _get_world_size(self, env):
        min_x = 0
        max_x = 10
        min_z = 0
        max_z = 10

        return torch.tensor([max_x - min_x, max_z - min_z, 360]).float()

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
