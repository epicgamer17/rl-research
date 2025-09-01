import torch
import os
from common import generate_titles

import sys

sys.path.append("../..")
from agent_history import FourierVectorhashAgentHistory
from common import analyze_history_errors, write_animation
from graph_utils import colored_line
from vectorhash_functions import circular_mean_weighted
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fourier_scaffold import FourierScaffold


loop_results_dir = "loop_results_sept_1"
loop_animations_dir = "loop_animations_sept_1"
loop_plots_dir = "loop_plots_sept_1"

os.makedirs(loop_animations_dir, exist_ok=True)
os.makedirs(loop_plots_dir, exist_ok=True)
titles = generate_titles()

fast = False
kidnap_t = 48 if fast else 240

for entry in os.listdir(loop_results_dir):
    result_dict = torch.load(f"{loop_results_dir}/{entry}", weights_only=False)
    data: tuple[FourierVectorhashAgentHistory, list[int], torch.Tensor] = result_dict[
        "results"
    ]
    scaffold: FourierScaffold = result_dict["scaffold"]
    history, path, noisy_world_vs = data

    # plot true path vs. estimated path
    Ps = history._Ps

    positions = torch.zeros(len(noisy_world_vs + 1), 2)
    for i in range(len(history._true_positions) - 1):
        true_pos = history._true_positions[i]
        positions[i, 0] = (
            (true_pos[0] + 3 * scaffold.scale_factor[0]) % scaffold.grid_limits[0]
        ) / scaffold.scale_factor[0]
        positions[i, 1] = (
            (true_pos[1] + 3 * scaffold.scale_factor[1]) % scaffold.grid_limits[1]
        ) / scaffold.scale_factor[1]

    true_xs = positions[:, 0]
    true_ys = positions[:, 1]

    noisy_xs = noisy_world_vs.cumsum(dim=0)[:, 0] + positions[0, 0]
    noisy_ys = noisy_world_vs.cumsum(dim=0)[:, 1] + positions[0, 1]

    est_xs = []
    est_ys = []
    scaffold = FourierScaffold(
        shapes=scaffold.shapes,
        D=400,
        features=history._scaffold_features,
        device="cuda",
        _skip_Ts_calc=True,
        _skip_K_calc=True,
        limits=torch.tensor([10, 10, torch.pi * 2]).to("cuda"),
    )
    for P in Ps:
        if P == None:
            continue
        else:
            dim_sizes = [
                int(scaffold.shapes[:, i].prod().item())
                for i in range(scaffold.shapes.shape[1])
            ]
            probs = scaffold.get_all_probabilities(P.to("cuda")).reshape(*dim_sizes)
            probs_xy = probs.sum(dim=2)
            dist_x = probs_xy.sum(dim=1)
            dist_y = probs_xy.sum(dim=0)

            circ_mean_x = circular_mean_weighted(
                torch.arange(len(dist_x), device=dist_x.device), dist_x, len(dist_x)
            )
            circ_mean_y = circular_mean_weighted(
                torch.arange(len(dist_y), device=dist_x.device), dist_y, len(dist_y)
            )
            est_xs.append(
                scaffold.grid_coords_to_world_coords(
                    circ_mean_x.item()
                    + positions[0, 0] * scaffold.scale_factor[0].item(),
                    0,
                )
            )
            est_ys.append(
                scaffold.grid_coords_to_world_coords(
                    circ_mean_y.item()
                    + positions[0, 1] * scaffold.scale_factor[1].item(),
                    1,
                )
            )

    fig, ax = plt.subplots()

    colored_line(
        true_xs,
        true_ys,
        torch.linspace(0.5, 2, len(true_xs)),
        ax,
        cmap="Greens",
    )
    colored_line(
        noisy_xs,
        noisy_ys,
        torch.linspace(0.5, 2, len(noisy_xs)),
        ax,
        cmap="Blues",
        linestyle="--",
    )
    colored_line(
        est_xs,
        est_ys,
        torch.linspace(0.5, 2, len(est_xs)),
        ax,
        cmap="Reds",
        linestyle=":",
    )

    green_patch = Patch(color="green", label="True Path")
    blue_patch = Patch(color="blue", label="Noisy Path", linestyle="--")
    red_patch = Patch(color="red", label="Estimated Path", linestyle=":")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(handles=[green_patch, blue_patch, red_patch])

    fig.savefig(f"{loop_plots_dir}/{entry}.png")

    # write_animation(history, loop_animations_dir, entry)
    fig = analyze_history_errors(history, kidnap_t=kidnap_t)

    i = int(entry.split(".")[0])
    fig.suptitle(titles[i])
    fig.savefig(f"{loop_plots_dir}/{i}.png")
