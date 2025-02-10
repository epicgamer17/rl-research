import numpy as np

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