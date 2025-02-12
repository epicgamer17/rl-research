import tqdm
import torch
from numpy.random import randn, randint
from vectorhash_imported import *
from scipy.linalg import norm
from nd_scaffold import *
from matrix_initializers import *


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


def senstrans_gs_vectorized_patts(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity, noise_level, grid_scaffold, W_hg_mean, W_hg_std):
    # avg error over Npatts
    ga = [[] for _ in range(len(Npatts_lst))]
    gd = [[] for _ in range(len(Npatts_lst))]
    gt = [[] for _ in range(len(Npatts_lst))]
    err_pc = 1*np.ones((len(Npatts_lst), nruns))
    err_sens = 1*np.ones((len(Npatts_lst), nruns))
    err_senscup = 1*np.ones((len(Npatts_lst), nruns))
    err_gc = 1*np.ones((len(Npatts_lst), nruns))
    err_sensl1 = 1*np.ones((len(Npatts_lst), nruns))
    M = len(lambdas)


    Wpg = SparseMatrixBySparsityInitializer(sparsity=0.6, device="cpu")((Np, Ng))
    thresh=0.5
    #thresh=-5

    # ein sum for Wpg of shape 1 400 50 and gbook of shape 3600 50 and P should be a 1 400 3600
    pbook = grid_scaffold.H
    Wgp = grid_scaffold.W_gh

    k=0
    b = torch.eye(Np,Ns)
    c = torch.eye(Ns,Np)
    A = torch.linalg.pinv(b)
    A2 = torch.linalg.pinv(c)

    # print("Wsp and Wps Hebbian")
    for Npatts in tqdm.tqdm(Npatts_lst):
        #print("k=",k)
        tsbook = torch.from_numpy(sbook[:, :Npatts])
        # Learning patterns 
        # make all entries floats
        tsbook = tsbook.float()
        print("tsbook shape",tsbook.shape)
        # Wps = grid_scaffold.calculate_update_Whs(sbook=tsbook, hbook=pbook[:Npatts,:])
        # Wsp = grid_scaffold.calculate_update_Wsh(hbook=pbook[:Npatts,:], sbook=tsbook)
        for i in range(Npatts):
            # print(i)
            # print(pbook[i, :].shape)
            # print(tsbook[:, i].shape)
            # print(grid_scaffold.calculate_update(input=pbook[i, :], output=tsbook[:, i]).shape)
            print(torch.linalg.cond(b))
            print(torch.linalg.svd(b))
            b = Rk1MrUpdate(A, b, tsbook[:, i].T, pbook[i, :].T, 3e-2, 0)
            A = torch.linalg.pinv(b)

            # c = Rk1MrUpdate(A2, c, pbook[i, :].T, tsbook[:, i].T, 3e-2, 0)
            # A2 = torch.linalg.pinv(c)
            grid_scaffold.W_hs += grid_scaffold.calculate_update(input=tsbook[:, i], output=pbook[i, :])
            # grid_scaffold.learned_pseudo_inversehs(input=tsbook[:, i], output=pbook[i, :])
            # grid_scaffold.learned_pseudo_inversesh(input=pbook[i, :], output=tsbook[:, i])

        

        # Wsp = train_sensory(pbook, sbook, Npatts)
        # Wps = pseudotrain_Wps(pbook, torch.from_numpy(sbook).float(), Npatts)
        # Wsp = pseudotrain_Wsp(torch.from_numpy(sbook).float(), pbook, Npatts)

        # Wps = np.einsum('ijk->ikj',Wsp)

        # grid_scaffold.W_hs = 
        grid_scaffold.W_sh = A

        # Testing
        sum_pc = 0
        sum_gc = 0 
        sum_sens = 0  
        sum_senscup = 0 
        sum_sensl1 = 0
        srep = np.repeat(sbook[None,:],nruns,axis=0)
        if noise_level == "none":
            sinit = srep
        else:
            if noise_level == "low":
                random_noise = torch.zeros_like(srep).uniform_(-1, 1)
            elif noise_level == "medium":
                random_noise = torch.zeros_like(srep).uniform_(-1.25, 1.25)
            elif noise_level == "high":
                random_noise = torch.zeros_like(srep).uniform_(-1.5, 1.5)
            sinit = srep + random_noise

        
        #For CTS
        # print(pflip)
        # sbook_std = np.std(sbook.flatten())
        # sinit = srep + np.random.normal(0,1,srep.shape)*pflip*sbook_std
        # print(srep.shape)
        # print(sinit.shape)
        
        ga[k], gd[k], gt[k], err_pc[k],err_gc[k],err_sens[k],_,err_sensl1[k] = dynamics_gs_vectorized_patts(tsbook,Niter, tsbook, pbook, gbook, Wgp, Wpg,A,grid_scaffold.W_hs,lambdas,sparsity,thresh,Npatts, gs=grid_scaffold)
        # err_pc[k],err_gc[k],err_sens[k],_,err_sensl1[k] = dynamics_gs_vectorized_patts_cts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts)

        k += 1   
    return ga, gd, gt, err_pc, err_gc, err_sens, err_senscup, err_sensl1


def dynamics_gs_vectorized_patts(sinit,Niter, sbook, pbook, gbook, Wgp, Wpg,Wsp,Wps,lambdas,sparsity,thresh,Npatts, gs):
    Ns = sbook.shape[0]
    Np = pbook.shape[1]
    Ng = gbook.shape[0]
    ga= []
    gd = []
    gt = [gbook[0,:]]
    # mean_p_norm = np.mean(np.linalg.norm(pbook[0],axis=0))
    # noise_val=1.
    # print("using p noise")#; mean_p_norm="+str(mean_p_norm))
    pin = gs.hippocampal_from_sensory(sinit.T)
    # pnoise = noise_val*mean_p_norm*np.random.normal(0,1,pin.shape)/np.sqrt(Np)
    
    # pin = pin+pnoise
    
    for i in range(Niter):
        gin = gs.grid_from_hippocampal(pin)
        # gin = gin.permute(0,2,1)
        # g = gs.denoise(gin)
        ga.append(gs.grid_from_hippocampal(pin))
        #permute back
        # g = g.permute(0,2,1)
        g = gs.denoise(gin)
        gd.append(gs.denoise(gs.grid_from_hippocampal(pin)))
        p = gs.hippocampal_from_grid(g)
    pout = p.T
    gout = g.T
    sout = gs.sensory_from_hippocampal(p).T



    # make all entries <-1 to -1 and >1 to 1
    sout = torch.clamp(sout, -1, 1)
    sout = torch.sign(sout)
    strue=sbook
    ptrue=pbook[:Npatts,:].T
    gtrue=gbook[:Npatts,:].T
    
    print("sout shape",sout.shape)
    print("strue shape",strue.shape)
    print("pout shape",pout.shape)
    print("ptrue shape",ptrue.shape)
    print("gout shape",gout.shape)
    print("gtrue shape",gtrue.shape)

    s_l1_err = np.average(abs(sout - strue).detach().numpy())/2
    print(s_l1_err)
    s_l2_err = torch.linalg.norm(torch.linalg.norm(sout-strue, dim=0), dim=0)/(Ns)
    p_l2_err = torch.linalg.norm(torch.linalg.norm(pout-ptrue, dim=0), dim=0)/(Np)
    g_l2_err = torch.linalg.norm(torch.linalg.norm(sout-strue, dim=0), dim=0)/(Ng)
    

    # print(sout.shape)
    struenormed=(strue/torch.linalg.norm(strue, dim=0))
    soutnormed=(sout*(1/torch.linalg.norm(sout,dim=0)))
    # dot product of normalized vectors, batch wise 
    dots = torch.einsum('ij,ij->i',struenormed,soutnormed)
        
    #scup = cleanup(s, sbook)
    errpc = (p_l2_err)
    errgc = (g_l2_err)
    errsens = ( s_l2_err)
    errsenscup = torch.nan*torch.zeros_like(errsens)#np.linalg.norm(scup-strue, axis=(1,2))/Ns

    errsensl1 = (s_l1_err)

    return ga, gd, gt, errpc, errgc, errsens, errsenscup, errsensl1 


