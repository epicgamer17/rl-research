import matplotlib.pyplot as plt
from nd_scaffold import GridScaffold


def graph_scaffold(g: GridScaffold):
    # G
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=600)
    ax.imshow(g.G.cpu().numpy(), cmap="gray", aspect="auto")
    ax.set_title("G")
    ax.set_ylabel("N_patts")
    ax.set_xlabel("N_g")
    fig.savefig("G.png")
    # H
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=600)
    a = ax.imshow(g.H.cpu().numpy(), cmap="gray", aspect="auto")
    ax.set_ylabel("N_patts")
    ax.set_xlabel("N_h")
    ax.set_title("H")
    fig.colorbar(a)
    fig.savefig("H.png")
    # W_hg
    fig, ax = plt.subplots(1, 2, figsize=(4, 4), dpi=200)
    a = ax[0].imshow(g.W_hg.cpu().numpy(), cmap="hot")
    ax[0].set_title("W_hg")
    ax[0].set_xlabel("N_h")
    ax[0].set_ylabel("N_g")
    fig.colorbar(a)

    a = ax[1].imshow((g.W_hg.cpu().numpy() == 0), cmap="hot")
    ax[1].set_title("W_hg == 0")
    ax[1].set_xlabel("N_h")
    ax[1].set_ylabel("N_g")
    fig.savefig("W_hg.png")
    # W_gh
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)
    ax.set_title("W_gh")
    a = ax.imshow(g.W_gh.cpu().numpy(), cmap="hot")
    ax.set_xlabel("N_g")
    ax.set_ylabel("N_h")
    fig.colorbar(a)
    fig.savefig("W_gh.png")


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

