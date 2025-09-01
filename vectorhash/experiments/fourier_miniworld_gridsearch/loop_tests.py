import torch

torch.manual_seed(0)
import os

from common import (
    generate_env,
    generate_combinations,
    generate_titles,
    create_agent_for_test,
    img_size_map,
)

from fourier_vectorhash import path_test

results_dir = "loop_results_sept_1"
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


forward_20 = [2] * 20
right_60_deg = [1] * 20
loop_path = (forward_20 + right_60_deg) * 6 + forward_20

forward_20_fast = [2] * 4
right_60_deg_fast = [1] * 4
loop_path_2_fast = (forward_20_fast + right_60_deg_fast) * 8 + forward_20_fast


if __name__ == "__main__":
    # combinations, titles = generate_combinations(), generate_titles()
    # for i, (combination, title) in enumerate(zip(combinations, titles)):
    #     print(f"test {i+1}/{len(combinations)}: {title}")
    #     env = generate_env(with_red_box=True, with_blue_box=True)
    #     history = path_test(
    #         agent=create_agent_for_test(env, *combination),
    #         path=torch.tensor(loop_path_2),
    #         reshape_img_size=img_size_map[combination[1]],
    #     )
    #     with open(f"{results_dir}/{i}.pkl", "wb") as f:
    #         pickle.dump(history, f)

    combinations, titles = generate_combinations(), generate_titles()
    for i, (combination, title) in enumerate(zip(combinations, titles)):
        print(f"(fast) test {i+1}/{len(combinations)}: {title}")
        env = generate_env(with_red_box=True, with_blue_box=True, fast=False)
        agent = create_agent_for_test(env, *combination)
        noise_dist = torch.distributions.Normal(0, 0.02)
        results = path_test(
            agent=agent,
            path=torch.tensor(loop_path),
            reshape_img_size=img_size_map[combination[1]],
            noise_dist=noise_dist,
        )
        torch.save(
            {"results": results, "scaffold": agent.vectorhash.scaffold},
            f"{results_dir}/{i}.pkl",
        )
