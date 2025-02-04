import os
import pathlib
import matplotlib.pyplot as plt
from nd_scaffold import GridScaffold


def graph_scaffold(g: GridScaffold, dir=None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
        base_path = dir
    else:
        base_path = '.'
    
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
    ax.set_aspect('equal', adjustable='box')
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
    GS = GridScaffold(
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
    plt.savefig(out)
    plt.close(fig)


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

    fig, ax = plt.subplots(2, 1, dpi=400, figsize=(5,3))

    ax[0].imshow(info["H"].cpu().numpy(), cmap="gray")
    ax[0].set_xlabel("N_h")
    ax[0].set_ylabel("N_patts")
    ax[0].title.set_text("H")

    ax[1].imshow(info["H_denoised"].cpu().numpy(), cmap="gray")
    ax[1].set_xlabel("N_h")
    ax[1].set_ylabel("N_patts")
    ax[1].title.set_text("H_denoised")

    fig, ax = plt.subplots(2, 2, dpi=400, figsize=(5, 8))

    ax[0][0].imshow(info["H"][:50,:50].cpu().numpy(), cmap="gray")
    ax[0][0].set_xlabel("N_patts")
    ax[0][0].set_ylabel("N_h")
    ax[0][0].title.set_text("H, first 50")

    ax[1][0].imshow(info["H_denoised"][:50,:50].cpu().numpy(), cmap="gray")
    ax[1][0].set_xlabel("N_patts")
    ax[1][0].set_ylabel("N_h")
    ax[1][0].title.set_text("H_denoised, first 50")

    ax[0][1].imshow(info["H"][:50,:50].cpu().numpy() == 0, cmap="gray")
    ax[0][1].set_xlabel("N_patts")
    ax[0][1].set_ylabel("N_h")
    ax[0][1].title.set_text("H, first 50, zero locations")

    ax[1][1].imshow(1 - (info["H_denoised"][:50,:50].cpu().numpy() == 0), cmap="gray")
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
        plt.title("Nh = " + str(array[i][0]) + " ratio active/not = " + str(array[i][1]))
        plt.legend()
        ax.set_xscale('log')
        plt.show()