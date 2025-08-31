import torch
from torch.distributions import Normal
import numpy as np

torch.manual_seed(0)
import os

from common import (
    generate_traj_env,
    generate_combinations,
    generate_titles,
    create_agent_for_test,
    img_size_map,
)

from fourier_vectorhash import trajectory_test
from random_path import generate_rat_trajectory

results_dir = "trajectory_tests"
os.makedirs(results_dir, exist_ok=True)

# each forward step (action 2) = 0.2 units
# each right step (actions 1) = 3 degrees right
#  (0,0)                 x
#    +-------------------------- . . .  ----------------->
#    |
#    |                                   __  red box
#    |                                   || (8, 1.5)
#    |        start*[1]                  --
#    |          (3,3)    (5,3)
#    |            x->----+
#    |           /        \
#    |          /          \ (6,4)
# y  |    (4,6) \          /
#    |           \        /
#    |            +------+
#    |          (3,5)    (5,5)
#    .
#    .
#    .                                  [1]: initial direcion (->) is 0 degrees
#    |
#    |
#    |  --   blue box
#    |  ||  (1.5,8)
#    |  --
#    ↓

pos, _ = generate_rat_trajectory(
    limits=np.array([1, 1]), num_samples=300, start_location=np.array([0.5, 0.5])
)

vel = pos[1:] - pos[:-1]

if __name__ == "__main__":
    combinations, titles = generate_combinations(), generate_titles()
    for i, (combination, title) in enumerate(zip(combinations, titles)):
        print(f"(fast) test {i+1}/{len(combinations)}: {title}")
        env = generate_traj_env(with_blue_box=True, with_red_box=True)
        noise_dist = Normal(0, 0.01)
        history, noisy_vels = trajectory_test(
            agent=create_agent_for_test(env, *combination),
            velocities=torch.from_numpy(vel).float(),
            reshape_img_size=img_size_map[combination[1]],
            noise_dist=noise_dist,
        )
        with open(f"{results_dir}/{i}.pt", "wb") as f:
            torch.save(
                {"history": history, "noisy_velocities": noisy_vels, "positions": pos},
                f,
            )
