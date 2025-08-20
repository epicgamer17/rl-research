import sys

sys.path.append("../..")

import torch

torch.manual_seed(0)
import math
from fourier_scaffold import (
    FourierScaffold,
    FourierScaffoldDebug,
    ContractionSharpening,
    GuassianFourierSmoothingMatrix,
    HadamardShiftMatrixRat,
    calculate_padding,
)
from graph_utils import plot_with_error, plot_imgs_side_by_side
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def zero(dim_sizes):
    return torch.zeros(*dim_sizes, device=device)


def uniform(dim_sizes):
    t = torch.ones_like(zero(dim_sizes))
    return t / t.sum()


def degenerate(dim_sizes):
    t = zero(dim_sizes)
    t[tuple([0] * 2)] = 1
    return t


def gaussian(dim_sizes, sigma=1):
    t = degenerate(dim_sizes)
    kernel_size = 2 * max(10, 3 * math.ceil(sigma)) + 1
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    y = torch.arange(kernel_size, device=device) - kernel_size // 2
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x_padding = calculate_padding(kernel_size, kernel.shape[0], 1)
    y_padding = calculate_padding(kernel_size, kernel.shape[1], 1)
    padded = torch.nn.functional.pad(
        t.unsqueeze(0).unsqueeze(0),
        y_padding + x_padding,
        mode="circular",
    )
    convoluted = torch.nn.functional.conv2d(
        input=padded, weight=kernel.unsqueeze(0).unsqueeze(0)
    )
    return convoluted.squeeze(0).squeeze(0)


def bimodal(dim_sizes):
    t = zero(dim_sizes)
    index = [0] * 2
    for i, size in enumerate(dim_sizes):
        index[i] = size // 2
    t[tuple([0] * 2)] = 0.5
    t[tuple(index)] = 0.5
    return t


def bimodal2(dim_sizes):
    t = zero(dim_sizes)
    index = [0] * 2
    for i, size in enumerate(dim_sizes):
        index[i] = 5
    t[tuple([1] * 2)] = 0.5
    t[tuple(index)] = 0.5
    return t


def gaussian_mixture(dim_sizes):
    g1 = gaussian(dim_sizes, sigma=1.5)
    g2 = gaussian(dim_sizes, sigma=2)
    g3 = gaussian(dim_sizes, sigma=2.5)
    t = (
        g1.roll(shifts=(7, 7), dims=(0, 1))
        + g2.roll(shifts=(30, 13), dims=(0, 1))
        + g3.roll(shifts=(17, 28), dims=(0, 1))
    )
    t = t / t.sum()
    return t


def random(dim_sizes):
    t = torch.rand(dim_sizes, device=device)
    t = t / t.sum()
    return t


def dim_sizes(shapes):
    return [int(shapes[:, dim].prod().item()) for dim in range(shapes.shape[1])]


def distributions(dim_sizes):
    ret = [
        ("degenerate", degenerate(dim_sizes)),
        ("uniform", uniform(dim_sizes)),
        ("random", random(dim_sizes)),
        ("gaussian σ=3", gaussian(dim_sizes, 3)),
        ("gaussian σ=5", gaussian(dim_sizes, 5)),
        ("bimodal", bimodal(dim_sizes)),
        ("gaussian mixture", gaussian_mixture(dim_sizes)),
    ]

    return ret


def l2_err(v1: torch.Tensor, v2: torch.Tensor):
    return torch.linalg.vector_norm(v1 - v2)


def similarity(v1: torch.Tensor, v2: torch.Tensor):
    return (v1 * v2.conj()).sum() / (v1.norm() * v2.norm())


def run_test(
    distribution: torch.Tensor,
    scaffold: FourierScaffold,
    scaffold_debug: FourierScaffoldDebug,
):
    scaffold.P = scaffold.encode_probability(distribution)
    scaffold_debug.ptensor = distribution

    original_decoded_probs = scaffold.get_all_probabilities()
    true_probs = scaffold_debug.ptensor
    original_probs = true_probs.clone()
    original_probs_l2_err = l2_err(
        true_probs.flatten(), original_decoded_probs.flatten()
    )

    scaffold.sharpen()
    scaffold_debug.sharpen()

    true_encodings = scaffold.encode_probability(scaffold_debug.ptensor).flatten()
    generated_encodings = scaffold.P.flatten()
    sharpened_encoding_similarity = similarity(true_encodings, generated_encodings)
    sharpened_encoding_l2 = l2_err(true_encodings, generated_encodings)

    sharpened_decoded_probs = scaffold.get_all_probabilities()
    true_probs = scaffold_debug.ptensor
    sharpened_probs = true_probs.clone()
    sharpened_probs_l2_err = l2_err(
        true_probs.flatten(), sharpened_decoded_probs.flatten()
    )

    return (
        original_decoded_probs.clone(),
        sharpened_decoded_probs.clone(),
        original_probs.clone(),
        sharpened_probs.clone(),
        original_probs_l2_err,
        sharpened_probs_l2_err,
        sharpened_encoding_similarity,
        sharpened_encoding_l2,
    )


def exp1():
    shapes = torch.tensor([(5, 5), (7, 7)], device=device)
    nruns = 5
    dim_sizes = [int(shapes[:, dim].prod().item()) for dim in range(shapes.shape[1])]
    Ds = [100 * i for i in range(1, 11)]
    dists = distributions(dim_sizes)
    sharpened_encoding_similarities = torch.zeros(len(dists), len(Ds), nruns)
    sharpened_encoding_l2s = torch.zeros(len(dists), len(Ds), nruns)
    original_probs_l2_errs = torch.zeros(len(dists), len(Ds), nruns)
    sharpened_probs_l2_errs = torch.zeros(len(dists), len(Ds), nruns)

    decoded_probability_heatmaps = torch.empty(
        len(dists), len(Ds), *dim_sizes, device=device
    )
    decoded_sharpened_probability_heatmaps = torch.empty(
        len(dists), len(Ds), *dim_sizes, device=device
    )
    original_sharpened_probability_heatmaps = torch.empty(
        len(dists), *dim_sizes, device=device
    )
    original_probability_heatmaps = torch.empty(len(dists), *dim_sizes, device=device)

    for run in range(nruns):
        dists = distributions(dim_sizes)
        for j, D in enumerate(Ds):
            scaffold = FourierScaffold(
                shapes,
                D=D,
                sharpening=ContractionSharpening(2),
                shift=HadamardShiftMatrixRat(shapes=shapes),
                smoothing=GuassianFourierSmoothingMatrix(
                    kernel_radii=[10, 10], kernel_sigmas=[1, 1]
                ),
                device=device,
                representation="matrix",
                _skip_K_calc=True,
                _skip_gs_calc=True,
                _skip_Ts_calc=True,
            )
            scaffold_debug = FourierScaffoldDebug(shapes, device=device)
            for i, (name, distribution) in enumerate(dists):
                print(
                    f" ----------------------- running test: {name} ({i+1}/{len(dists)}), D={D} ({j+1}/{len(Ds)}), run {run+1}/{nruns} --------------------"
                )

                (
                    original_decoded_probs,
                    sharpened_decoded_probs,
                    original_probs,
                    sharpened_probs,
                    original_probs_l2_err,
                    sharpened_probs_l2_err,
                    sharpened_encoding_similarity,
                    sharpened_encoding_l2,
                ) = run_test(distribution.clone(), scaffold, scaffold_debug)

                sharpened_encoding_similarities[i, j, run] = (
                    sharpened_encoding_similarity
                )
                sharpened_encoding_l2s[i, j, run] = sharpened_encoding_l2
                original_probs_l2_errs[i, j, run] = original_probs_l2_err
                sharpened_probs_l2_errs[i, j, run] = sharpened_probs_l2_err

                if run == 0:
                    decoded_probability_heatmaps[i, j] = (
                        original_decoded_probs.clone().cpu().reshape(35, 35)
                    )
                    decoded_sharpened_probability_heatmaps[i, j] = (
                        sharpened_decoded_probs.clone().cpu().reshape(35, 35)
                    )
                    original_probability_heatmaps[i] = (
                        original_probs.clone().cpu().reshape(35, 35)
                    )
                    original_sharpened_probability_heatmaps[i] = (
                        sharpened_probs.clone().cpu().reshape(35, 35)
                    )

    data = {
        "sharpened_encoding_similarities": sharpened_encoding_similarities,
        "sharpened_encoding_l2s": sharpened_encoding_l2s,
        "original_probs_l2_errs": original_probs_l2_errs,
        "sharpened_probs_l2_errs": sharpened_probs_l2_errs,
        "decoded_probability_heatmaps": decoded_probability_heatmaps,
        "decoded_sharpened_probability_heatmaps": decoded_sharpened_probability_heatmaps,
        "original_probability_heatmaps": original_probability_heatmaps,
        "original_sharpened_probability_heatmaps": original_sharpened_probability_heatmaps,
    }
    torch.save(data, "exp_1_data.pkl")


def exp_1_analysis():
    data = torch.load("exp_1_data.pkl")
    sharpened_encoding_similarities = data["sharpened_encoding_similarities"]
    sharpened_encoding_l2s = data["sharpened_encoding_l2s"]
    original_probs_l2_errs = data["original_probs_l2_errs"]
    sharpened_probs_l2_errs = data["sharpened_probs_l2_errs"]
    decoded_probability_heatmaps = data["decoded_probability_heatmaps"]
    decoded_sharpened_probability_heatmaps = data[
        "decoded_sharpened_probability_heatmaps"
    ]
    original_probability_heatmaps = data["original_probability_heatmaps"]
    original_sharpened_probability_heatmaps = data[
        "original_sharpened_probability_heatmaps"
    ]

    shapes = torch.tensor([(5, 5), (7, 7)], device=device)
    dim_sizes = [int(shapes[:, dim].prod().item()) for dim in range(shapes.shape[1])]
    dists = distributions(dim_sizes)
    Ds = [100 * i for i in range(1, 11)]

    ### D vs l2 err between true sharpened encodings and computed sharpened encodings
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(dists):
        plot_with_error(ax, Ds, sharpened_encoding_l2s[i].cpu(), label=name)

    ax.set_xlabel("D")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between true and computed sharpened encodings")
    ax.legend()
    fig.savefig("sharpened_encoding_l2_error_vs_D.png", bbox_inches="tight")

    ### D vs cosine similarity between true sharpened encodings and computed sharpened encodings
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(dists):
        plot_with_error(ax, Ds, sharpened_encoding_similarities[i].cpu(), label=name)

    ax.set_xlabel("D")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity between true and computed sharpened encodings")
    ax.legend()
    fig.savefig("sharpened_encoding_similarity_vs_D.png", bbox_inches="tight")

    ### D vs l2 error between original and recovered probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(dists):
        plot_with_error(ax, Ds, original_probs_l2_errs[i].cpu(), label=name)

    ax.set_xlabel("D")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between original and recovered encodings")
    ax.legend()
    fig.savefig("original_vs_recovered_l2_error_vs_D.png", bbox_inches="tight")

    ### D vs l2 error between original and recovered sharpened probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(dists):
        plot_with_error(ax, Ds, sharpened_probs_l2_errs[i].cpu(), label=name)

    ax.set_xlabel("D")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between original and recovered sharpened probabilities")
    ax.legend()
    fig.savefig(
        "original_vs_recovered_sharpened_l2_error_vs_D.png", bbox_inches="tight"
    )

    ### Probability heatmap comparisons for true distributions
    fig, ax = plt.subplots(
        nrows=len(dists),
        ncols=1 + len(Ds),
        figsize=(2 * len(Ds), 2 * len(dists)),
        layout="compressed",
    )
    for r in ax:
        for c in r:
            c.axis("off")

    for i, (name, dist) in enumerate(dists):
        plot_imgs_side_by_side(
            imgs=[original_probability_heatmaps[i].cpu()]
            + [decoded_probability_heatmaps[i, j].cpu() for j in range(len(Ds))],
            titles=[f"{name}"] + ([f"D={D}" for D in Ds] if i == 0 else [""] * len(Ds)),
            axs=ax[i, :],
            use_first_img_scale=True,
            fig=fig,
            cbar_only_on_last=True,
        )
    fig.savefig("original_vs_decoded_probability_heatmaps.png")

    ### Probability heatmap comparisons for sharpened distributions
    fig, ax = plt.subplots(
        nrows=len(dists),
        ncols=1 + len(Ds),
        figsize=(2 * len(Ds), 2 * len(dists)),
        layout="compressed",
    )
    for r in ax:
        for c in r:
            c.axis("off")

    for i, (name, _) in enumerate(dists):
        plot_imgs_side_by_side(
            imgs=[original_sharpened_probability_heatmaps[i].cpu()]
            + [
                decoded_sharpened_probability_heatmaps[i, j].cpu()
                for j in range(len(Ds))
            ],
            titles=[f"{name}"] + ([f"D={D}" for D in Ds] if i == 0 else [""] * len(Ds)),
            axs=ax[i, :],
            use_first_img_scale=True,
            fig=fig,
            cbar_only_on_last=True,
        )
    fig.savefig(
        "true_vs_decoded_sharpened_probability_heatmaps.png", bbox_inches="tight"
    )


def exp2():
    shape_configurations = [
        torch.tensor(s)
        for s in [
            [(5, 5), (8, 8)],
            [(5, 5), (12, 12)],
            [(5, 5), (16, 16)],
            [(5, 5), (21, 21)],
            [(5, 5), (26, 26)],
            [(5, 5), (31, 31j)],
        ]
    ]

    N = len(shape_configurations)
    num_dists = len(distributions(dim_sizes(shape_configurations[0])))
    D = 400
    nruns = 1

    sharpened_encoding_similarities = torch.zeros(N, num_dists, nruns)
    sharpened_encoding_l2s = torch.zeros(N, num_dists, nruns)
    original_probs_l2_errs = torch.zeros(N, num_dists, nruns)
    sharpened_probs_l2_errs = torch.zeros(N, num_dists, nruns)
    for run in range(nruns):
        for i, shapes in enumerate(shape_configurations):
            scaffold = FourierScaffold(
                shapes,
                D=D,
                sharpening=ContractionSharpening(2),
                shift=HadamardShiftMatrixRat(shapes=shapes),
                smoothing=GuassianFourierSmoothingMatrix(
                    kernel_radii=[10, 10], kernel_sigmas=[1, 1]
                ),
                device=device,
                representation="matrix",
                _skip_K_calc=True,
                _skip_gs_calc=True,
                _skip_Ts_calc=True,
            )
            scaffold_debug = FourierScaffoldDebug(shapes, device=device)

            for j, (name, distribution) in enumerate(distributions(dim_sizes(shapes))):
                print(
                    f" ----------------------- running test: {shapes.tolist()} ({i+1}/{N}), {name} ({j+1}/{num_dists}), run {run+1}/{nruns} --------------------"
                )
                (
                    _,
                    _,
                    _,
                    _,
                    original_probs_l2_err,
                    sharpened_probs_l2_err,
                    sharpened_encoding_similarity,
                    sharpened_encoding_l2,
                ) = run_test(distribution.clone(), scaffold, scaffold_debug)

                sharpened_encoding_similarities[i, j, run] = (
                    sharpened_encoding_similarity
                )
                sharpened_encoding_l2s[i, j, run] = sharpened_encoding_l2
                original_probs_l2_errs[i, j, run] = original_probs_l2_err
                sharpened_probs_l2_errs[i, j, run] = sharpened_probs_l2_err

    data = {
        "sharpened_encoding_similarities": sharpened_encoding_similarities,
        "sharpened_encoding_l2s": sharpened_encoding_l2s,
        "original_probs_l2_errs": original_probs_l2_errs,
        "sharpened_probs_l2_errs": sharpened_probs_l2_errs,
    }
    torch.save(data, "exp_2_data.pkl")


def exp2_analysis():
    data = torch.load("exp_2_data.pkl")

    sharpened_encoding_similarities = data["sharpened_encoding_similarities"]
    sharpened_encoding_l2s = data["sharpened_encoding_l2s"]
    original_probs_l2_errs = data["original_probs_l2_errs"]
    sharpened_probs_l2_errs = data["sharpened_probs_l2_errs"]

    shape_configurations = [
        torch.tensor(s)
        for s in [
            [(5, 5), (8, 8)],
            [(5, 5), (12, 12)],
            [(5, 5), (16, 16)],
            [(5, 5), (21, 21)],
            [(5, 5), (26, 26)],
            [(5, 5), (31, 31j)],
        ]
    ]
    omega_sizes = [x.prod().item() for x in shape_configurations]

    ### |\Omega| vs similarity between sharpened and computed encodings
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, _) in enumerate(distributions(dim_sizes(shape_configurations[0]))):
        plot_with_error(
            ax,
            omega_sizes,
            sharpened_encoding_similarities[:, i].cpu(),
            label=name,
        )

    ax.set_xlabel("Size of \\Omega")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity between true and computed sharpened encodings")
    ax.legend()
    fig.savefig("sharpened_encoding_similarity_vs_omega.png", bbox_inches="tight")

    ### |\Omega| vs L2 error between true and computed sharpened encodings
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(distributions(dim_sizes(shape_configurations[0]))):
        plot_with_error(
            ax,
            omega_sizes,
            sharpened_encoding_l2s[:, i].cpu(),
            label=name,
        )

    ax.set_xlabel("Size of \\Omega")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between true and computed sharpened encodings")
    ax.legend()
    fig.savefig("sharpened_encoding_l2_error_vs_omega.png", bbox_inches="tight")

    ### |\Omega| vs L2 error between recovered and true original distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(distributions(dim_sizes(shape_configurations[0]))):
        plot_with_error(
            ax,
            omega_sizes,
            original_probs_l2_errs[:, i].cpu(),
            label=name,
        )

    ax.set_xlabel("Size of \\Omega")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between true and recovered original distributions")
    ax.legend()
    fig.savefig("original_distribution_l2_error_vs_omega.png", bbox_inches="tight")

    ### |\Omega| vs L2 error between recovered and true sharpened distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, _) in enumerate(distributions(dim_sizes(shape_configurations[0]))):
        plot_with_error(
            ax,
            omega_sizes,
            sharpened_probs_l2_errs[:, i].cpu(),
            label=name,
        )

    ax.set_xlabel("Size of \\Omega")
    ax.set_ylabel("L2 error")
    ax.set_title("L2 error between true and recovered sharpened distributions")
    ax.legend()
    fig.savefig("sharpened_distribution_l2_error_vs_omega.png", bbox_inches="tight")


if __name__ == "__main__":
    exp1()
    exp_1_analysis()
    exp2()
    exp2_analysis()
