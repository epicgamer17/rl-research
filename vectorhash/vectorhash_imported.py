import numpy as np
import torch

def smooth_tuningcurve(avg_fields, Npos, mult=2, path=False, path_locations=None):
    avg_fields_sq = avg_fields.reshape((Npos,Npos))
    if path:
        afs = np.zeros_like(avg_fields_sq)
        afs[:] = np.nan
        afs[path_locations[:,0], path_locations[:,1]] = avg_fields_sq[path_locations[:,0], path_locations[:,1]]
    else:
        afs = np.copy(avg_fields_sq)
    afs = afs.T
    afs2 = upsample(afs,mult)
    hexed_afs = np.copy(afs2)

    for i in range(Npos):
        hexed_afs[mult*i:mult*(i+1)] = np.roll(hexed_afs[mult*i:mult*(i+1)],i) 
    
    return hexed_afs
    
      
def upsample(im, mult=2):
    height, width = np.shape(im)
    im_up = np.zeros((mult * height, mult * width))

    for i in range(height):
        for j in range(width):
            im_up[mult * i: mult * (i + 1), mult * j: mult * (j +1)] = im[i, j]
            
    return im_up

from scipy.ndimage import gaussian_filter
def explicit_interpolation(hexed_afs, upsample_rate=10, sigma=10):
    hexed_afs_up = upsample(hexed_afs, upsample_rate)

    # doesn't deal with nans when plotting for paths
    #hexed_afs_up_smooth = gaussian_filter(hexed_afs_up, sigma, mode='wrap')

    # deals with nans when plotting for paths but is super slow
    # from astropy.convolution import convolve_fft as asconvolve
    # from astropy.convolution import Gaussian2DKernel
    # kernel = Gaussian2DKernel(x_stddev=sigma,y_stddev=sigma)
    # hexed_afs_up_smooth = asconvolve(hexed_afs_up,kernel,boundary='wrap')


    U=hexed_afs_up.copy()
    U[np.isnan(hexed_afs_up)]=0
    UU=gaussian_filter(U,sigma=sigma)

    W=0*hexed_afs_up.copy()+1
    W[np.isnan(hexed_afs_up)]=0
    WW=gaussian_filter(W,sigma=sigma)

    hexed_afs_up_smooth=UU/WW

    return hexed_afs_up_smooth


def gen_gbook_2d(lambdas, Ng, Npos):
    """
    Return grid codebook (grid activity vector for each position)

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        Npos - int, number of spatial positions in each axis
    
    Outputs:
        gbook - np.array, size (Ng, Npos, Npos)
            gbook[:, a, b] = grid vector at position (a, b)
    """
    # Ng = np.sum(np.dot(lambdas, lambdas))
    # Npos = np.prod(lambdas)
    gbook = np.zeros((Ng, Npos, Npos))
    for x in range(Npos):
        for y in range(Npos):
            index = 0
            for period in lambdas:
                phi1, phi2 = x % period, y % period
                gpattern = np.zeros((period, period))
                gpattern[phi1, phi2] = 1
                gpattern = gpattern.flatten()
                gbook[index:index+len(gpattern), x, y] = gpattern
                index += len(gpattern)
    return gbook



def gen_gbook(lambdas, Ng, Npos):
    ginds = [0,lambdas[0],lambdas[0]+lambdas[1]]; 
    gbook=np.zeros((Ng,Npos))
    for x in range(Npos):
        phis = np.mod(x,lambdas) 
        gbook[phis+ginds,x]=1 
    return gbook


# global nearest neighbor
def nearest_neighbor(gin, gbook):
    est = np.transpose(gin)@gbook; 
    a = np.where(est[0,:]==max(est[0,:]))
    #print("Nearest neighbor: ", a)
    idx = np.random.choice(a[0])
    g = gbook[:,idx]; 
    return g


# module wise nearest neighbor
def module_wise_NN(gin, gbook, lambdas):
    size = gin.shape
    g = np.zeros(size)               #size is (Ng, 1)
    i = 0
    for j in lambdas:
        gin_mod = gin[i:i+j]           # module subset of gin
        gbook_mod = gbook[i:i+j]
        g_mod = nearest_neighbor(gin_mod, gbook_mod)
        g[i:i+j, 0] = g_mod
        i = i+j
    return g    


def capacity(sensory_model, lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, Npatts_lst, nruns, Ns, sbook, sparsity, noise_level, grid_scaffold, W_hg_mean, W_hg_std):
    ga = [[] for _ in range(len(Np_lst))]
    gd = [[] for _ in range(len(Np_lst))]
    gt = [[] for _ in range(len(Np_lst))]
    err_gc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_pc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_sens = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_senscup = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))
    err_sensl1 = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))

    l = 0
    for Np in Np_lst:
        print("l =",l)
        ga[l], gd[l], gt[l], err_pc[l], err_gc[l], err_sens[l], err_senscup[l], err_sensl1[l]  = sensory_model(lambdas, Ng, Np, pflip, Niter, Npos, 
                                                gbook, Npatts_lst, nruns, Ns, sbook, sparsity,noise_level, grid_scaffold, W_hg_mean, W_hg_std)
        l = l+1

    return err_pc, err_gc, err_sens, err_senscup, err_sensl1, ga, gd, gt


def train_gcpc(pbook, gbook, Npatts):
    return (1/Npatts)*np.einsum('ij, klj -> kil', gbook[:,:Npatts], pbook[:,:,:Npatts])  
    

def pseudotrain_Wsp(sbook, ca1book, Npatts):
    ca1inv = torch.linalg.pinv(ca1book[:Npatts, :])
    return sbook[:,:Npatts] @ ca1inv[:,:Npatts].T

def pseudotrain_Wps(ca1book, sbook, Npatts):
    sbookinv = torch.linalg.pinv(sbook[:, :Npatts])
    return  ca1book[:Npatts,:].T @ sbookinv[:Npatts,:]
    
def pseudotrain_Wsp(sbook, ca1book, Npatts):
    ca1inv = torch.linalg.pinv(ca1book[:Npatts, :])
    return np.einsum('ij, kjl -> kil', sbook[:,:Npatts], ca1inv[:,:Npatts]) 

def pseudotrain_Wps(ca1book, sbook, Npatts):
    sbookinv = np.linalg.pinv(sbook[:, :Npatts])
    return np.einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:Npatts]) 


def gridCAN_2d(gs,lambdas):
    #gs.shape == nruns,Ng,Npatts
    nruns,Ng,Npatts = gs.shape
    ls = [l**2 for l in lambdas]
    i=0
    gout = np.zeros(gs.shape)
    for j in ls:
        gmod=gs[:,i:i+j,:]
        #print(gmod.shape)
        maxes = gmod.argmax(axis=1)
        #print(maxes.shape)
        for ru in range(nruns):
            gout[ru][maxes[ru]+i,np.arange(Npatts)] = 1
        i=i+j
    return gout