import matplotlib.pyplot as plt
from vectorhash.nd_scaffold import GridScaffold


def graph_scaffold(g: GridScaffold):
    # G
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=900)
    ax.imshow(g.G.cpu().numpy(), cmap="gray", aspect="auto")
    fig.savefig("G.png")
    # H
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=900)
    a = ax.imshow(g.H.cpu().numpy(), cmap="gray", aspect="auto")
    fig.colorbar(a)
    fig.savefig("H.png")
    # W_hg
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.W_hg.cpu().numpy(), cmap="hot")
    fig.colorbar(a)
    fig.savefig("W_hg.png")
    # W_gh
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.W_gh.cpu().numpy(), cmap="hot")
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
