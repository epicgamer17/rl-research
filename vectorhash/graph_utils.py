import os
import pathlib
import matplotlib.axes
import matplotlib.pyplot as plt
from clean_scaffold import GridHippocampalScaffold

# from animalai_agent_history import VectorhashAgentKidnappedHistory


def graph_scaffold(g: GridHippocampalScaffold, dir=None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
        base_path = dir
    else:
        base_path = "."

    g_path = pathlib.Path(base_path, "G.png")
    h_path = pathlib.Path(base_path, "H.png")
    W_hg_path = pathlib.Path(base_path, "W_hg.png")
    W_gh_path = pathlib.Path(base_path, "W_gh.png")
    # G
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=600)
    ax.imshow(g.G.cpu().numpy(), cmap="gray", aspect="auto")
    ax.set_title("G")
    ax.set_ylabel("N_patts")
    ax.set_xlabel("N_g")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(g_path)
    # H
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=600)
    a = ax.imshow(g.H.cpu().numpy(), cmap="gray", aspect="auto")
    ax.set_ylabel("N_patts")
    ax.set_xlabel("N_h")
    ax.set_title("H")
    fig.colorbar(a)
    fig.savefig(h_path)
    # W_hg
    fig, ax = plt.subplots(1, 2, figsize=(4, 4), dpi=400)
    a = ax[0].imshow(g.W_hg.cpu().numpy(), cmap="hot")
    ax[0].set_title("W_hg")
    ax[0].set_xlabel("N_h")
    ax[0].set_ylabel("N_g")
    fig.colorbar(a)

    a = ax[1].imshow((g.W_hg.cpu().numpy() == 0), cmap="hot")
    ax[1].set_title("W_hg == 0")
    ax[1].set_xlabel("N_h")
    ax[1].set_ylabel("N_g")
    fig.savefig(W_hg_path)
    # W_gh
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=400)
    ax.set_title("W_gh")
    a = ax.imshow(g.W_gh.cpu().numpy(), cmap="hot")
    ax.set_xlabel("N_g")
    ax.set_ylabel("N_h")
    fig.colorbar(a, orientation="horizontal")
    fig.savefig(W_gh_path)


if __name__ == "__main__":
    GS = GridHippocampalScaffold(
        shapes=[(2, 2, 3), (3, 3, 5)],
        # shapes = [(3,4,5), (3,4,5), (5,7,8)],
        N_h=400,
        input_size=784,
        device="cuda",
    )
    graph_scaffold(GS)


def print_imgs_side_by_side(*imgs, out="mnist.png", captions=None, title=None):
    if captions is not None:
        assert len(captions) == len(imgs)

    fig, ax = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4), dpi=900)
    for i, img in enumerate(imgs):
        ax[i].imshow(img, cmap="gray")
        ax[i].axis("off")
        if captions is not None:
            ax[i].set_title(captions[i])

    if title is not None:
        fig.suptitle(title)

    if out is not None:
        plt.savefig(out)
        plt.close(fig)
    else:
        plt.show()


def plot_recall_info(info):
    fig, ax = plt.subplots(1, 2, dpi=200, figsize=(4, 5))

    ax[0].imshow(info["G"].cpu().numpy(), cmap="gray")
    ax[0].set_xlabel("N_g")
    ax[0].set_ylabel("N_patts")
    ax[0].title.set_text("G")

    ax[1].imshow(info["G_denoised"].cpu().numpy(), cmap="gray")
    ax[1].set_xlabel("N_g")
    ax[1].set_ylabel("N_patts")
    ax[1].title.set_text("G_denoised")

    fig, ax = plt.subplots(2, 1, dpi=400, figsize=(5, 3))

    ax[0].imshow(info["H"].cpu().numpy(), cmap="gray")
    ax[0].set_xlabel("N_h")
    ax[0].set_ylabel("N_patts")
    ax[0].title.set_text("H")

    ax[1].imshow(info["H_denoised"].cpu().numpy(), cmap="gray")
    ax[1].set_xlabel("N_h")
    ax[1].set_ylabel("N_patts")
    ax[1].title.set_text("H_denoised")

    fig, ax = plt.subplots(2, 2, dpi=400, figsize=(5, 8))

    ax[0][0].imshow(info["H"][:50, :50].cpu().numpy(), cmap="gray")
    ax[0][0].set_xlabel("N_patts")
    ax[0][0].set_ylabel("N_h")
    ax[0][0].title.set_text("H, first 50")

    ax[1][0].imshow(info["H_denoised"][:50, :50].cpu().numpy(), cmap="gray")
    ax[1][0].set_xlabel("N_patts")
    ax[1][0].set_ylabel("N_h")
    ax[1][0].title.set_text("H_denoised, first 50")

    ax[0][1].imshow(info["H"][:50, :50].cpu().numpy() == 0, cmap="gray")
    ax[0][1].set_xlabel("N_patts")
    ax[0][1].set_ylabel("N_h")
    ax[0][1].title.set_text("H, first 50, zero locations")

    ax[1][1].imshow(1 - (info["H_denoised"][:50, :50].cpu().numpy() == 0), cmap="gray")
    ax[1][1].set_xlabel("N_patts")
    ax[1][1].set_ylabel("N_h")
    ax[1][1].title.set_text("H_denoised, first 50, zero locations")


def graphing_recall(array):
    """
    graphs recall based off Nh
    two curves per graph one for mnist one for cifar
    metric y is cosine similarity
    x is % of max patterns used
    then those graphs for each combination of Nh, ratio active/not


    input :
    array is an array with n entries, one for each combination of Nh, ratio active/not
    first entry is Nh, second is ratio active/not, third is mnist scores fourth is the x values
    """
    # make input of the x values to be log scales so that we can see the differences better and the inputs are percentages
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(5, 5))
    for i in range(len(array)):

        plt.plot(array[i][3], array[i][2], label="CIFAR")
        plt.xlabel("% of max patterns used")
        plt.ylabel("cosine similarity")
        plt.title(
            "Nh = " + str(array[i][0]) + " ratio active/not = " + str(array[i][1])
        )
        plt.legend()
        ax.set_xscale("log")
        plt.show()


def print_imgs_side_by_side_on_top(imgs, out="mnist.png", captions=None, title=None):
    fig, ax = plt.subplots(len(imgs), 2, figsize=(4 * 2, 4 * len(imgs)), dpi=900)
    for i in range(len(imgs)):
        ax[i][0].imshow(imgs[i][0], cmap="gray")
        ax[i][0].axis("off")
        ax[i][1].imshow(imgs[i][1], cmap="gray")
        ax[i][1].axis("off")
        if captions is not None:
            ax[i][0].set_title(captions[0])
            ax[i][1].set_title(captions[1])

    if title is not None:
        fig.suptitle(title)

    if out is not None:
        plt.savefig(out)
        plt.close(fig)
    else:
        plt.show()


def plot_path(path, beliefs, out=None):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=600)
    # scatter plot the path as points, with lines between them
    ax[0].scatter(path[:, 0], path[:, 1], c="blue", s=1)
    ax[0].plot(path[:, 0], path[:, 1], c="blue", linewidth=0.5)
    ax[0].set_title("Path")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_aspect("equal", adjustable="box")

    # plot beliefs as a scatter plot
    ax[1].scatter(beliefs[:, 0], beliefs[:, 1], c="red", s=1)
    ax[1].set_title("Beliefs")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_aspect("equal", adjustable="box")
    if out is not None:
        plt.savefig(out)
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


import matplotlib
import numpy as np
from matplotlib.patches import StepPatch


def plot_probability_distribution_on_ax(
    distribution: np.ndarray, ax: matplotlib.axes.Axes
):
    patch = StepPatch(values=distribution, edges=np.arange(len(distribution) + 1))
    ax.add_patch(patch)
    return patch


def plot_error_over_time(
    beliefs: np.ndarray,
    true_positions: np.ndarray,
    time_steps: np.ndarray,
    out: str = None,
):
    print("beliefs", beliefs.shape)
    print("true_positions", true_positions.shape)
    errors = np.linalg.norm(beliefs - true_positions, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=600)
    ax.plot(time_steps, errors, label="Error")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title("Error over time")
    ax.legend()
    ax.grid()
    if out is not None:
        plt.savefig(out)
        plt.close(fig)
    else:
        plt.show()
    return fig, ax

def error_test(true, belief):
    loss = 0
    for i in range(len(belief)):
        diff = abs(i - true)
        loss += belief[i] * min(diff, len(belief) - diff)
    return loss


def plot_errors_on_axes(
    history,
    axis: matplotlib.axes.Axes,
    visible=None,
):
    true_pos = history._true_positions
    theta_pos = history._true_angles
    b_x_pos_dists = history._x_distributions
    b_y_pos_dists = history._y_distributions
    b_theta_pos_dists = history._theta_distributions
    x_errors = []
    y_errors = []
    theta_errors = []
    fig = plt.figure(figsize=(8, 4), dpi=600)
    for i in range(len(b_x_pos_dists)):
        if visible is not None:
            if visible[i] == False:
                continue

        x_error = error_test(true_pos[i][0], b_x_pos_dists[i])
        y_error = error_test(true_pos[i][1], b_y_pos_dists[i])
        theta_error = error_test(theta_pos[i], b_theta_pos_dists[i])
        x_errors.append(x_error)
        y_errors.append(y_error)
        theta_errors.append(theta_error)

    axis.plot(x_errors, label="x error")
    axis.plot(y_errors, label="y error")
    axis.plot(theta_errors, label="theta error")

    axis.set_xlabel("Time")
    axis.set_ylabel("Error")

    return axis
