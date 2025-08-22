import sys

sys.path.append("../..")

import torch

torch.manual_seed(0)
from fourier_scaffold import FourierScaffold, GuassianFourierSmoothingMatrix
from hippocampal_sensory_layers import (
    ComplexExactPseudoInverseHippocampalSensoryLayerComplexScalars,
)
import matplotlib.pyplot as plt
from graph_utils import plot_with_error, plot_imgs_side_by_side

device = "cuda"


def corrupt(sbook: torch.Tensor, p):
    """returns a tensor that with probability p flips the sign of sbook"""
    if p == 0:
        return sbook

    mask = torch.rand_like(sbook.real) < p
    flipped = sbook.clone()
    flipped[mask] = -flipped[mask]
    return flipped


def dynamics_s_h_h_s(
    scaffold: FourierScaffold,
    layer: ComplexExactPseudoInverseHippocampalSensoryLayerComplexScalars,
    sbook: torch.Tensor,
):
    h = layer.hippocampal_from_sensory(sbook)

    def f(h):
        P = torch.outer(h, h.conj())
        P_sharp = scaffold.sharpening(P, scaffold.features)
        h_sharp = torch.einsum("ijm,ij->m", scaffold.T_s, P_sharp)
        return h_sharp

    h_sharp = torch.vmap(f, 0, 0, chunk_size=100)(h)
    s_recovered_sharp = layer.sensory_from_hippocampal(h_sharp)
    s_recovered_nosharp = layer.sensory_from_hippocampal(h)
    return h, h_sharp, s_recovered_sharp, s_recovered_nosharp


def run_test(
    scaffold: FourierScaffold,
    layer: ComplexExactPseudoInverseHippocampalSensoryLayerComplexScalars,
    sbook: torch.Tensor,
    Npatts: int,
    p_flip: float,
):
    gbook = scaffold.gbook().T[:Npatts]
    scaffold.smoothing
    hbook = gbook[:Npatts]
    layer.learn_batch(sbook, hbook)

    h, h_sharp, recovered_sbook, recovered_sbook_nosharp = dynamics_s_h_h_s(
        scaffold, layer, corrupt(sbook, p_flip)
    )
    p_err = (sbook - recovered_sbook.sign()).abs().mean() / 2
    avg_l1 = (sbook - recovered_sbook).abs().mean()
    mean_h_err_l2 = ((gbook - h).abs() ** 2).mean()
    mean_h_sharp_err_l2 = ((h_sharp - h).abs() ** 2).mean()

    P_nosharp = torch.einsum("bi,bj->bij", h, h.conj())
    P_sharp = torch.einsum("bi,bj->bij", h_sharp, h_sharp.conj())
    dists_nosharp = torch.vmap(scaffold.get_all_probabilities, chunk_size=1)(
        P_nosharp[:10]
    )  # (B, |\Omega|, |\Omega|)
    dists_sharp = torch.vmap(scaffold.get_all_probabilities, chunk_size=1)(
        P_sharp[:10]
    )  # (B, |\Omega|, |\Omega|)
    return (
        recovered_sbook,
        recovered_sbook_nosharp,
        dists_nosharp,
        dists_sharp,
        p_err,
        avg_l1,
        mean_h_err_l2,
        mean_h_sharp_err_l2,
    )


img_size = (20, 20)
N_s = 20 * 20


def exp_1(pflip=0.01):
    runs = 5
    D_list = [400, 500, 600, 700, 800]

    Npatts = 100
    shape_configs = [
        torch.tensor(s)
        for s in [[(3, 3), (4, 4), (5, 5)], [(5, 5), (12, 12)], [(60,), (60,)]]
    ]

    p_flips = torch.empty(len(shape_configs), len(D_list), runs)
    l1_errs = torch.empty(len(shape_configs), len(D_list), runs)
    h_errs = torch.empty(len(shape_configs), len(D_list), runs)
    h_sharp_errs = torch.empty(len(shape_configs), len(D_list), runs)
    recovered_sbook_0 = torch.empty(
        len(shape_configs), len(D_list), Npatts, N_s, device=device
    )
    recovered_sbook_nosharp_0 = torch.empty(
        len(shape_configs), len(D_list), Npatts, N_s, device=device
    )
    sbook = torch.sign(torch.randn(runs, Npatts, N_s, device=device))
    dists_sharp = torch.empty(len(shape_configs), len(D_list), 10, 3600, device=device)
    dists_nosharp = torch.empty(
        len(shape_configs), len(D_list), 10, 3600, device=device
    )
    dists_org = torch.empty(len(shape_configs), len(D_list), 10, 3600, device=device)
    for k in range(runs):
        for i, shapes in enumerate(shape_configs):

            for j, D in enumerate(D_list):
                scaffold = FourierScaffold(
                    shapes=shapes,
                    D=D,
                    device=device,
                    _skip_K_calc=True,
                )
                gbook = scaffold.gbook().T[:Npatts]
                layer = ComplexExactPseudoInverseHippocampalSensoryLayerComplexScalars(
                    input_size=N_s, N_h=D, N_patts=Npatts, hbook=gbook, device=device
                )
                (
                    recovered_sbook,
                    recovered_sbook_nosharp,
                    dists_sharp_10,
                    dists_nosharp_10,
                    p_flips[i, j, k],
                    l1_errs[i, j, k],
                    h_errs[i, j, k],
                    h_sharp_errs[i, j, k],
                ) = run_test(scaffold, layer, sbook[k], Npatts, pflip)
                if k == 0:
                    recovered_sbook_0[i, j] = recovered_sbook
                    recovered_sbook_nosharp_0[i, j] = recovered_sbook_nosharp
                    dists_sharp[i, j] = dists_sharp_10
                    dists_nosharp[i, j] = dists_nosharp_10

    data = {
        "D_list": D_list,
        "shape_configs": shape_configs,
        "p_flips": p_flips,
        "pflip": pflip,
        "l1_errs": l1_errs,
        "sbook": sbook,
        "recovered_sbook_run_0": recovered_sbook_0,
        "recovered_sbook_nosharp_run_0": recovered_sbook_nosharp_0,
        "h_errs": h_errs,
        "h_sharp_errs": h_sharp_errs,
        "dists_sharp": dists_sharp,
        "dists_nosharp": dists_nosharp,
        "dists": dists_org,
    }
    torch.save(data, "exp_1_results.pt")


def exp_1_analysis():
    data = torch.load("exp_1_results.pt")
    sbook = data["sbook"]
    D_list = data["D_list"]
    shape_configs = data["shape_configs"]
    p_flips = data["p_flips"]
    l1_errs = data["l1_errs"]
    recovered_sbook_0 = data["recovered_sbook_run_0"]
    recovered_sbook_nosharp_0 = data["recovered_sbook_nosharp_run_0"]
    h_errs = data["h_errs"]
    h_sharp_errs = data["h_sharp_errs"]
    dists_sharp = data["dists_sharp"]
    dists_nosharp = data["dists_nosharp"]
    dists_org = data["dists"]
    pflip = data["pflip"]

    ### avg l1 err vs D
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, shape in enumerate(shape_configs):
        plot_with_error(
            ax=ax,
            x=D_list,
            y=l1_errs[i],
            label=f"{shape.tolist()}",
        )
    ax.set_xlabel("D")
    ax.set_ylabel("Average L1 error")
    # ax.set_ylim(0, 1)
    # ax.set_title("avg_l1_err vs D for different shape configs")
    ax.legend()
    fig.savefig(f"avg_l1_err_vs_D_pflip-{pflip}.png", bbox_inches="tight")

    ### p(flip) vs D
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, shape in enumerate(shape_configs):
        plot_with_error(
            ax=ax,
            x=D_list,
            y=p_flips[i],
            label=f"{shape.tolist()}",
        )
    ax.set_xlabel("D")
    ax.set_ylabel("Probability of flipping a sign")
    # ax.set_title("p_flip error vs D for different shape configs")
    ax.legend()
    fig.savefig(f"p_flip_err_vs_D_pflip-{pflip}.png", bbox_inches="tight")

    ### h err vs D
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, shape in enumerate(shape_configs):
        plot_with_error(
            ax=ax,
            x=D_list,
            y=h_errs[i],
            label=f"{shape.tolist()}",
        )
    ax.set_xlabel("D")
    ax.set_ylabel("average l2 error ||H - H_recovered||")
    # ax.set_title("h l2 err vs D for different shape configs")
    ax.legend()
    fig.savefig(f"h_l2_err_vs_D_pflip-{pflip}.png", bbox_inches="tight")

    ### h sharp err vs D
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, shape in enumerate(shape_configs):
        plot_with_error(
            ax=ax,
            x=D_list,
            y=h_sharp_errs[i],
            label=f"{shape.tolist()}",
        )
    ax.set_xlabel("D")
    ax.set_ylabel("average l2 error ||H - H_recovered_sharp||")
    ax.set_title("h sharp l2 err vs D for different shape configs")
    ax.legend()
    fig.savefig(f"h_sharp_l2_err_vs_D_pflip-{pflip}.png", bbox_inches="tight")

    ### recovered sbook graphing for specific D and shape config
    N = 10
    for i, shape in enumerate(shape_configs):
        for j, D in enumerate(D_list):
            fig, ax = plt.subplots(
                nrows=3, ncols=N, figsize=(2 * N, 6), layout="compressed"
            )
            sbook_0 = sbook[0]
            patts = sbook_0[:N].reshape(N, *img_size).cpu()
            recovered_patts = recovered_sbook_0[i, j, :N].reshape(N, *img_size).cpu()
            recovered_patts_nosharp = (
                recovered_sbook_nosharp_0[i, j, :N].reshape(N, *img_size).cpu()
            )

            plot_imgs_side_by_side(
                axs=ax[0],
                imgs=patts,
                titles=[f"original {i}" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )
            plot_imgs_side_by_side(
                axs=ax[1],
                imgs=recovered_patts,
                titles=[f"recovered {i}" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )
            plot_imgs_side_by_side(
                axs=ax[2],
                imgs=recovered_patts_nosharp,
                titles=[f"recovered {i} (nosharp)" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )

            fig.savefig(
                f"recovered_sbook_vs_time_shapes-{shape.tolist()}_D-{D}_pflio-{pflip}.png"
            )

    ### plots dists_sharp and dists_nosharp
    for i, shape in enumerate(shape_configs):
        for j, D in enumerate(D_list):
            fig, ax = plt.subplots(
                nrows=3, ncols=N, figsize=(2 * N, 6), layout="compressed"
            )
            dists_sharp_0 = dists_sharp[i, j]
            dists_nosharp_0 = dists_nosharp[i, j]

            plot_imgs_side_by_side(
                axs=ax[0],
                imgs=dists_sharp_0.reshape(10, 60, 60).cpu(),
                titles=[f"sharp {i}" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )
            plot_imgs_side_by_side(
                axs=ax[1],
                imgs=dists_nosharp_0.reshape(10, 60, 60).cpu(),
                titles=[f"nosharp {i}" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )
            plot_imgs_side_by_side(
                axs=ax[2],
                imgs=dists_org[i, j].reshape(10, 60, 60).cpu(),
                titles=[f"org {i}" for i in range(10)],
                fig=fig,
                cbar_only_on_last=True,
            )

            fig.savefig(f"dists_vs_time_shapes-{shape.tolist()}_D-{D}_pflip-{j}.png")


def exp_2():
    runs = 1
    D = 600
    Npatts = 100

    pflip_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    shape = torch.tensor([(3, 3), (4, 4), (5, 5)], device=device)

    p_flips = torch.empty(len(pflip_list), runs)
    l1_errs = torch.empty(len(pflip_list), runs)
    h_errs = torch.empty(len(pflip_list), runs)
    h_sharp_errs = torch.empty(len(pflip_list), runs)
    recovered_sbook_0 = torch.empty(len(pflip_list), Npatts, N_s, device=device)
    recovered_sbook_nosharp_0 = torch.empty(len(pflip_list), Npatts, N_s, device=device)
    sbook = torch.sign(torch.randn(runs, Npatts, N_s, device=device))
    for k in range(runs):
        scaffold = FourierScaffold(
            shapes=shape,
            D=D,
            device=device,
            smoothing=GuassianFourierSmoothingMatrix([10, 10], [1, 1]),
            _skip_K_calc=True,
        )
        gbook = scaffold.gbook().T[:Npatts]
        for i, pflip in enumerate(pflip_list):
            layer = ComplexExactPseudoInverseHippocampalSensoryLayerComplexScalars(
                input_size=N_s, N_h=D, N_patts=Npatts, hbook=gbook, device=device
            )
            (
                recovered_sbook,
                recovered_sbook_nosharp,
                _,
                _,
                p_flips[i, k],
                l1_errs[i, k],
                h_errs[i, k],
                h_sharp_errs[i, k],
            ) = run_test(scaffold, layer, sbook[k], Npatts, pflip)
            if k == 0:
                recovered_sbook_0[i] = recovered_sbook
                recovered_sbook_nosharp_0[i] = recovered_sbook_nosharp

    data = {
        "D": D,
        "shape": shape,
        "p_flips": p_flips,
        "pflip_list": pflip_list,
        "l1_errs": l1_errs,
        "sbook": sbook,
        "recovered_sbook_run_0": recovered_sbook_0,
        "recovered_sbook_nosharp_run_0": recovered_sbook_nosharp_0,
        "h_errs": h_errs,
        "h_sharp_errs": h_sharp_errs,
    }
    torch.save(data, "exp_2_results.pt")


def exp_2_analysis():
    data = torch.load("exp_2_results.pt")
    sbook = data["sbook"]
    D = data["D"]
    shape = data["shape"]
    p_flips = data["p_flips"]
    pflip_list = data["pflip_list"]
    l1_errs = data["l1_errs"]
    recovered_sbook_0 = data["recovered_sbook_run_0"]
    recovered_sbook_nosharp_0 = data["recovered_sbook_nosharp_run_0"]
    h_errs = data["h_errs"]
    h_sharp_errs = data["h_sharp_errs"]

    ### avg l1 err vs pflip
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_error(
        ax=ax,
        x=pflip_list,
        y=l1_errs,
        label=f"{shape.tolist()}",
    )
    ax.set_xlabel("p_flip")
    ax.set_ylabel("Mean L1 error")
    # ax.set_title("avg_l1_err vs p_flip for different shape configs")
    fig.savefig(f"avg_l1_err_vs_pflip_shapes-{shape.tolist()}.png", bbox_inches="tight")

    ### p(flip) vs pflip
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_error(
        ax=ax,
        x=pflip_list,
        y=p_flips,
        label=f"{shape.tolist()}",
    )
    ax.set_xlabel("p_flip")
    ax.set_ylabel("Probability of flipping a sign")
    # ax.set_title("p_flip error vs p_flip for different shape configs")
    fig.savefig(f"p_flip_err_vs_pflip_shapes-{shape.tolist()}.png", bbox_inches="tight")

    ### h err vs pflip
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_error(
        ax=ax,
        x=pflip_list,
        y=h_errs,
        label=f"{shape.tolist()}",
    )
    ax.set_xlabel("p_flip")
    ax.set_ylabel("mse ||h - h recovered||²")
    # ax.set_title("h l2 err vs p_flip for different shape configs")
    fig.savefig(f"h_l2_err_vs_pflip_shapes-{shape.tolist()}.png", bbox_inches="tight")

    ### h sharp err vs pflip
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_with_error(
        ax=ax,
        x=pflip_list,
        y=h_sharp_errs,
        label=f"{shape.tolist()}",
    )
    ax.set_xlabel("p_flip")
    ax.set_ylabel("mse ||h - h recovered sharp||²")
    # ax.set_title("h sharp l2 err vs p_flip for different shape configs")
    fig.savefig(
        f"h_sharp_l2_err_vs_pflip_shapes-{shape.tolist()}.png", bbox_inches="tight"
    )

    ### recovered sbook graphing for specific pflip and shape config
    N = 10
    for i in range(N):
        fig, ax = plt.subplots(
            nrows=2,
            ncols=len(pflip_list) + 1,
            figsize=((1 + len(pflip_list)) * 2, 5),
            layout="compressed",
        )
        plot_imgs_side_by_side(
            axs=ax[0],
            imgs=[sbook[0, i].reshape(*img_size).cpu()]
            + [
                recovered_sbook_0[j, i].reshape(*img_size).cpu()
                for j in range(len(pflip_list))
            ],
            titles=["original"] + [f"pflip={p}" for p in pflip_list],
            fig=fig,
            cbar_only_on_last=True,
        )
        plot_imgs_side_by_side(
            axs=ax[1],
            imgs=[sbook[0, i].reshape(*img_size).cpu()]
            + [
                recovered_sbook_nosharp_0[j, i].reshape(*img_size).cpu()
                for j in range(len(pflip_list))
            ],
            titles=[""] + [f"" for p in pflip_list],
            fig=fig,
            cbar_only_on_last=True,
        )
        fig.savefig(
            f"recovered_sbook_vs_pflip_shapes-{shape.tolist()}_D-{D}_pflip-{i}.png"
        )

if __name__ == "__main__":
    # exp_2()
    exp_2_analysis()
