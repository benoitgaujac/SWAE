import sys
import time
import os
from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import pdb


def init_gaussian_prior(opts):
    """
    Initialize the prior parameters (mu_0,sigma_0)
    for all our mixtures
    """
    if opts['full_cov_matrix'] and opts['zdim']==2:
        # pz_means, pz_sigma = set_2d_priors(opts['nmixtures'], opts['full_cov_matrix'],
        #                             x_var=1./3., y_var=sin(pi / nmixtures)/3.)
        pz_means, pz_sigma = set_2d_priors(opts['nmixtures'], opts['full_cov_matrix'],
                                    opts['x_var'], opts['y_var'])
        pz_means = opts['pz_scale']*pz_means
        pz_sigma *= opts['pz_scale']**2
    else:
        if opts['zdim']==2:
            means, pz_sigma = set_2d_priors(opts['nmixtures'])
            pz_means = opts['pz_scale']*means
            pz_sigma *= (opts['pz_scale']*sin(pi / opts['nmixtures'])/3)**2
        else:
            if opts['zdim']+1>=opts['nmixtures']:
                means = np.zeros([opts['nmixtures'], opts['zdim']],dtype='float32')
                for k in range(opts['nmixtures']):
                    if k<opts['zdim']:
                        means[k,k] = 1
                    else:
                        means[k] = np.ones((opts['zdim'],),dtype='float32')
                        means[k] *= (1. - sqrt(1.+opts['zdim'])) / opts['zdim']
                pz_means = opts['pz_scale']*means
                pz_sigma = opts['pz_sigma_scale']*np.ones((opts['nmixtures'], opts['zdim']),dtype='float32')
                pz_sigma *= 2 * (opts['pz_scale'] / 6.)**2
            else:
                assert False, 'Too many mixtures for the latents dim.'
    return pz_means, pz_sigma


def set_2d_priors(nmixtures, is_full_cov=False, x_var=1., y_var=1.):
    """
    Initialize prior parameters for zdim=2 and nmixtures=10
    Return:
    means: [nmixtures,zdim]
    sigma: [nmixtures,zdim,zdim] if is_full_cov [nmixtures,zdim] else
    """
    # means
    means = []
    mean = np.array([1., 0.], dtype='float32')
    base_angle = 2*pi / nmixtures
    for i in range(nmixtures):
        angle = i * base_angle
        means.append(np.array([cos(angle), sin(angle)], dtype='float32'))
    means = np.vstack(means)
    # cov matrix
    if is_full_cov:
        sigmas = []
        sigma = np.array([[x_var**2,0], [0,y_var**2]], dtype='float32')
        # sigma = np.array([[1./3.**2,0], [0,(sin(pi / nmixtures)/3.)**2]], dtype='float32')
        for i in range(nmixtures):
            rot = np.array([[cos(i * base_angle), -sin(i * base_angle)],
                    [sin(i * base_angle), cos(i * base_angle)]], dtype='float32')
            sigmas.append(np.matmul(np.matmul(rot,sigma),np.transpose(rot)))# + 1e-10*np.eye(2))
        sigmas = np.stack(sigmas)
    else:
        assert x_var==y_var, '{} and {} must be equal for scalar prior Sig.'.format(x_var,y_var)
        sigmas = x_var**2 * np.ones((nmixtures, 2),dtype='float32')

    return means, sigmas



def get_nearest(opts,means_list,mean):
    if len(means_list)==0:
        return opts['prior_threshold']
    else:
        nearest = np.sum(np.square(means_list[0]-mean))
        for e in means_list[1:]:
            dist = np.sum(np.square(e-mean))
            if dist<nearest:
                nearest = dist
    return nearest


def init_cat_prior(opts):
    """
    Initialize parameters of discrete distribution
    """
    # mean_params = tf.constant(1/opts['nmixtures'], shape=[opts['nmixtures']],
    #                                                         dtype=tf.float32,
    #                                                         name='pi0')
    mean_params = (np.ones(opts['nmixtures']) / opts['nmixtures']).astype(np.float32)
    return mean_params
