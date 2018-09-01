import sys
import time
import os
from math import sqrt, cos, sin, pi, ceil
import numpy as np
import tensorflow as tf

import pdb

def sample_mixtures(opts, means, covs, num=100,typ='numpy'):
    """
    Sample noise from gaussian distribution with parameters
    means and covs
    """
    if typ =='tensorflow':
        mean = tf.expand_dims(means,axis=2)
        cov = tf.expand_dims(covs,axis=2)
        eps = tf.random_normal([num,opts['nmixtures'],opts['nsamples'],
                                                        opts['zdim']],
                                                        dtype=tf.float32)
        noise = mean + tf.multiply(eps,tf.sqrt(1e-8+cov))
        # eps = tf.random_normal([num,opts['nmixtures'],opts['zdim']],
        #                                                 dtype=tf.float32)
        #noise = means + tf.multiply(eps,tf.sqrt(1e-8+covs))
    elif typ =='numpy':
        mean = np.expand_dims(means,axis=1)
        eps = np.random.normal(0.,1.,(num, opts['nmixtures'],opts['nsamples'],
                                                        opts['zdim'])).astype(np.float32)
        noise = mean + np.multiply(eps,np.sqrt(1e-8+covs))
        #eps = np.random.normal(0.,1.,(num, opts['nmixtures'],opts['zdim'])).astype(np.float32)
        # noise = means + np.multiply(eps,np.sqrt(1e-8+covs))
    return noise


def sample_pz(opts, means, covs, num=100, sampling_mode='one_mixture'):
    """
    Sample prior noise according to sampling_mode
    """
    noise = None
    noises = sample_mixtures(opts,means,covs,num)
    if sampling_mode == 'one_mixture':
        mixture = np.random.randint(opts['nmixtures'],size=num)
        noise = noises[np.arange(num),mixture,0]
        #noise = noises[np.arange(num),mixture]
    elif sampling_mode == 'per_mixture':
        samples_per_mixture = ceil(num / opts['nmixtures'])
        noise = noises[:samples_per_mixture,:,0]
        #noise = noises[:samples_per_mixture]
    elif sampling_mode == 'all_mixtures':
        noise = noises
    else:
        assert False, 'Unknown sampling mode.'
    return opts['pz_scale'] * noise


def generate_linespace(opts, n, mode, anchors):
    """
    Genereate latent linear grid space
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
        assert np.shape(anchors)[0]%2==0, 'Need an ode number of anchors points'
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
