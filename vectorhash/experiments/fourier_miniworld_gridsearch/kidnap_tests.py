import torch
torch.manual_seed(0)
import pickle
import os
import math

from common import (
    generate_env,
    generate_combinations,
    generate_titles,
    create_agent_for_test,
    img_size_map,
)

from fourier_vectorhash import kidnap_test

results_dir = "kidnap_path_results"
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
#    |          /          \ (6,4)* kidnapped position
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
pre_kidnap_path = (forward_20 + right_60_deg) * 6
post_kidnap_path = (forward_20 + right_60_deg) * 2
kidnap_pos = [6, 0, 4.0]
kidnap_dir = 120 * math.pi / 180

combinations, titles = generate_combinations(), generate_titles()
for i, (combination, title) in enumerate(zip(combinations, titles)):
    print(f"test {i+1}/{len(combinations)}: {title}")
    env = generate_env(with_red_box=True, with_blue_box=True)
    history = kidnap_test(
        agent=create_agent_for_test(env, *combination),
        pre_kidnap_path=torch.tensor(pre_kidnap_path),
        post_kidnap_path=torch.tensor(post_kidnap_path),
        kidnap_pos=kidnap_pos,
        kidnap_dir=kidnap_dir,
        reshape_img_size=img_size_map[combination[1]],
    )
    with open(f"{results_dir}/{i}.pkl", "wb") as f:
        pickle.dump(history, f)
