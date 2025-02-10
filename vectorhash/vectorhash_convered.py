import tqdm
import torch

from nd_scaffold import *


def corrupt_pcont(pflip, ptrue):
    if pflip == 0:
        return ptrue
    pinit = ptrue + pflip * torch.randn(*ptrue.shape)
    return pinit


def capacity_gcpc_vectorized(
    shapes,
    Np_lst,
    pflip,
    Niter,
    Npos,
    nruns,
    Npatts_lst,
    test_generalization="no",
    device=None,
):
    # avg error over Npatts
    # Npatts_lst = np.arange(1,Npos+1)
    # Npatts_lst = [21]
    err_gcpc = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns))
    num_correct = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns))
    l = 0
    for Np in Np_lst:
        k = 0
        for Npatts in tqdm.tqdm(Npatts_lst):
            gs = GridScaffold(
                shapes=shapes,
                N_h=Np,
                input_size=1,
                h_normal_mean=0,
                h_normal_std=1,
                device=device,
                sparse_matrix_initializer=SparseMatrixBySparsityInitializer(
                    sparsity=0.6, device=device
                ),
                calculate_g_method="fast",
                relu_theta=0.5,
                continualupdate=False,
                initialize_W_gh_with_zeroes=False,
                sanity_check=False
            )

            scores = torch.zeros((nruns, Npatts))
            for i in range(nruns):
                initializer = SparseMatrixBySparsityInitializer(
                    sparsity=0.6, device=device
                )
                gs.W_hg = initializer(gs.W_hg.shape)
                gs.H = gs.hippocampal_from_grid(gs.G)
                gs.W_gh = gs._W_gh(
                  noisy=True,
                  noisy_std=pflip,
                  Npatts=Npatts,
                )

                if test_generalization == "no":
                    test_patts = torch.arange(Npatts)
                else:
                    test_patts = torch.arange(Npos)  # torch.random.choice(Npos,100)

                # Testing
                ptrue = gs.H[test_patts]  # (len(test_patts), N_h))
                p_noisy = corrupt_pcont(pflip, ptrue)
                p_recovered = gs.hippocampal_from_grid(
                    gs.denoise(gs.grid_from_hippocampal(p_noisy))
                )
                
                score = torch.linalg.vector_norm(ptrue - p_recovered, dim=1) / Np # (Npatts, N_h) -> (Npatts)
                scores[i] = score

                # print(cleanup_vectorized.shape)

            err_gcpc[l, k] = torch.mean(scores)
            num_correct[l, k] = torch.sum((scores < 0.003).int())

            # for x in sampledpatt:
            # ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
            # pinit = corrupt_pcont(pflip, ptrue)      # make corrupted pc pattern
            # cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, thresh)   # pc-gc autoassociative cleanup
            # sum_gcpc += cleanup
            # num_corr += (cleanup<0.003).astype('int')
            # err_gcpc[l,k] = sum_gcpc/Npatts
            # num_correct[l,k] = num_corr
            k += 1
        l += 1
    return err_gcpc, num_correct
