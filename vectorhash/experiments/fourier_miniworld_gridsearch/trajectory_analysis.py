import torch
import os
from common import generate_titles
from fourier_scaffold import FourierScaffold
from matplotlib import pyplot as plt

import sys

sys.path.append("../..")
from common import analyze_history_errors, write_animation

trajectory_results_dir = "trajectory_tests"
trajectory_animations_dir = "trajectory_path_animations"
trajectory_plots_dir = "trajectory_path_plots"

os.makedirs(trajectory_animations_dir, exist_ok=True)
os.makedirs(trajectory_plots_dir, exist_ok=True)
titles = generate_titles()

for entry in os.listdir(trajectory_results_dir):

    with open(f"{trajectory_results_dir}/{entry}", "rb") as f:
        history = torch.load(f, weights_only=False)
        history, positions, noisy_vels = (
            history["history"],
            history["positions"],
            history["noisy_velocities"],
        )

        # plot true path vs. estimated path
        Ps = history._Ps
        true_xs = positions[:, 0]
        true_ys = positions[:, 1]

        noisy_xs = noisy_vels.cumsum(dim=0)[:, 0] + positions[0, 0]
        noisy_ys = noisy_vels.cumsum(dim=0)[:, 1] + positions[0, 1]

        est_xs = []
        est_ys = []
        scaffold = FourierScaffold(
            shapes=history._scaffold_shapes,
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
                    history._scaffold_shapes[:, i].prod()
                    for i in range(history._scaffold_shapes.shape[1])
                ]
                probs = scaffold.get_all_probabilities(P.to("cuda")).reshape(*dim_sizes)
                probs_xy = probs.sum(dim=2)
                max_xy = probs_xy.flatten().argmax()
                x_idx = max_xy % dim_sizes[0]
                y_idx = max_xy // dim_sizes[0]
                est_xs.append(
                    scaffold.grid_coords_to_world_coords(
                        x_idx.item()
                        + positions[0, 0] * scaffold.scale_factor[0].item(),
                        0,
                    )
                )
                est_ys.append(
                    scaffold.grid_coords_to_world_coords(
                        y_idx.item()
                        + positions[0, 1] * scaffold.scale_factor[1].item(),
                        1,
                    )
                )

        fig, ax = plt.subplots()
        ax.plot(true_xs, true_ys, label="True Path")
        ax.plot(noisy_xs, noisy_ys, label="Noisy Path")
        ax.plot(est_xs, est_ys, label="Estimated Path")
        ax.legend()

        fig.savefig(f"{trajectory_plots_dir}/{entry}.png")

        fig2 = analyze_history_errors(history)

        fig2.savefig(f"{trajectory_animations_dir}/{entry}.png")

        write_animation(
            history=history,
            target_dir=trajectory_animations_dir,
            entry_name=entry,
        )
