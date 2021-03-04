import sys
import time
import os
from math import sqrt, cos, sin, pi, ceil
import numpy as np
import tensorflow as tf

import pdb

def sample_all_gmm(opts, means, Sigma, batch_size=100, tensor=True):
    """
    Sample for each component of the gmm

    means/Sigma: [batch,K,zdim]/[K,zdim] for encoder/prior
    """
    if tensor:
        if len(means.get_shape().as_list())<3:
            means = tf.expand_dims(means,axis=0)
            Sigma = tf.expand_dims(covs,axis=0)
        shape = tf.shape(means)
        eps = tf.random_normal(shape, dtype=tf.float32)
        noise = means + tf.multiply(eps,tf.sqrt(1e-10+Sigma)) #[batch,K,zdim]
    else:
        if len(means.shape)<3:
            means = np.expand_dims(means,axis=0)
            Sigma = np.expand_dims(Sigma,axis=0)
        shape = means.shape[1:]
        eps = np.random.normal(0.,1.,(batch_size,)+shape).astype(np.float32)
        noise = means + np.multiply(eps,np.sqrt(1e-10+Sigma))

    return noise


def sample_gmm(opts, means, Sigma, batch_size=100, sampling_mode='one'):
    """
    Sample prior noise according to sampling_mode
    """
    noises = sample_all_gmm(opts, means, Sigma, batch_size, False)
    if sampling_mode == 'true':
        mixtures_id = np.random.randint(opts['nmixtures'],size=batch_size)
        samples = noises[np.arange(batch_size),mixture]
    elif sampling_mode == 'mixtures':
        nsamples_per_mixture = ceil(batch_size / opts['nmixtures'])
        samples = noises[:nsamples_per_mixture,:].reshape([-1,opts['zdim']])
    else:
        ValueError('Unkown {} sampling for gmm'.format(sampling_mode))

    return samples


def generate_linespace(opts, n, mode, anchors):
    """
    Genereate various latent interpolation
    """
    nanchors = np.shape(anchors)[0]
    dim_to_interpolate = min(opts['nmixtures'],opts['zdim'])
    if mode=='transformation':
        assert np.shape(anchors)[1]==0, 'Zdim needs to be 2 to plot transformation'
        ymin, xmin = np.amin(anchors,axis=0)
        ymax, xmax = np.amax(anchors,axis=0)
        x = np.linspace(1.1*xmin,1.1*xmax,n)
        y = np.linspace(1.1*ymin,1.1*ymax,n)
        linespce = np.stack(np.meshgrid(y,x)).T
    elif mode=='points_interpolation':
        assert np.shape(anchors)[0]%2==0, 'Need an even number of anchors points'
        axs = [[np.linspace(anchors[2*k,d],anchors[2*k+1,d],n) for d in range(dim_to_interpolate)] for k in range(int(nanchors/2))]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack([np.asarray(axs[i][j]) for j in range(dim_to_interpolate)],axis=0).T
            coord = np.zeros((crd.shape[0],opts['zdim']))
            coord[:,:crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    elif mode=='priors_interpolation':
        axs = [[np.linspace(anchors[0,d],anchors[k,d],n) for d in range(dim_to_interpolate)] for k in range(1,nanchors)]
        linespce = []
        for i in range(len(axs)):
            crd = np.stack([np.asarray(axs[i][j]) for j in range(dim_to_interpolate)],axis=0).T
            coord = np.zeros((crd.shape[0],opts['zdim']))
            coord[:,:crd.shape[1]] = crd
            linespce.append(coord)
        linespace = np.asarray(linespce)
    else:
        assert False, 'Unknown mode %s for vizualisation' % opts['mode']
    return linespace

def generate_latent_grid(opts, n, pz_means, pz_sigma):
    """
    Genereate linear latent grid
    """
    zdim = pz_means.shape[-1]
    assert zdim==2, "latent dimension must be equal to 2"
    idx_max = np.argmax(pz_means,axis=0) #[zdim,]
    idx_min = np.argmin(pz_means,axis=0) #[zdim,]
    xs = []
    for d in range(zdim):
        xmin = pz_means[idx_min[d],d] - pz_sigma[idx_min[d],d]
        xmax = pz_means[idx_max[d],d] + pz_sigma[idx_max[d],d]
        xs.append(np.linspace(xmin, xmax, n, endpoint=True))
    xv, yv = np.meshgrid(xs[0],xs[1])
    grid = np.stack((xv,yv),axis=-1)
    return grid
