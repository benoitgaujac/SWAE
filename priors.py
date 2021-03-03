import sys
import time
import os
from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import pdb

cat_initializer = tf.random_normal_initializer(mean=0.0, stddev=.1, dtype=tf.float64)


def init_gaussian_prior(opts):
    """
    Initialize the prior parameters (mu_0,sigma_0)
    for all our mixtures
    """
    if opts['zdim']==2:
        means = set_2d_priors(opts['nmixtures'])
    else:
        if opts['zdim']+1>=opts['nmixtures']:
            means = np.zeros([opts['nmixtures'], opts['zdim']],dtype='float32')
            for k in range(opts['nmixtures']):
                if k<opts['zdim']:
                    means[k,k] = 1
                else:
                    means[k] = - 1. / (1. + sqrt(opts['nmixtures']+1)) \
                                    * np.ones((opts['zdim'],),dtype='float32')
        else:
            means_list = []
            for k in range(opts['nmixtures']):
                nearest = 0.
                count = 0
                while nearest<opts['prior_threshold'] and count<10:
                    mean = np.random.uniform(low=-opts['prior_threshold'],
                                             high=opts['prior_threshold'],
                                             size=(opts['zdim']))
                    nearest = get_nearest(opts,means_list,mean)
                    count += 1
                means_list.append(mean)
            means = np.array(means_list)
            eps = np.random.normal(loc=0.0, scale=.01, size=np.shape(means))
            means = means + eps
            #assert False, 'Too many mixtures for the latents dim.'
    pz_means = opts['pz_scale']*means
    pz_sigma = opts['sigma_prior']*np.ones((opts['zdim']),dtype='float32')
    return pz_means, np.tile(np.expand_dims(pz_sigma,axis=0),(opts['nmixtures'],1))


def set_2d_priors(nmixtures):
    """
    Initialize prior parameters for zdim=2 and nmixtures=10
    """
    means, Sigmas = [], []
    mean = np.array([1., 0.], dtype='float32')
    # Sigma = np.diaglat([1., 0.2]).astype(np.float32)
    base_angle = 2*pi / nmixtures
    for i in range(nmixtures):
        angle = i * base_angle
        means.append(np.array([cos(angle), sin(angle)], dtype='float32'))
        # Sigmas.append(np.matmul(rot, sigma))
    means = np.vstack(means)
    # Sigmas = np.vstack(Sigmas)
    return means


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


def init_learnable_cat_prior(opts):
    """
    Initialize parameters of discrete distribution
    """
    with tf.variable_scope('prior'):
        logits = tf.get_variable("pi0", [opts['nmixtures']], initializer=cat_initializer)
    mean_params = tf.nn.softmax(logits)
    return mean_params

def init_cat_prior(opts):
    """
    Initialize parameters of discrete distribution
    """
    # mean_params = tf.constant(1/opts['nmixtures'], shape=[opts['nmixtures']],
    #                                                         dtype=tf.float32,
    #                                                         name='pi0')
    mean_params = (np.ones(opts['nmixtures']) / opts['nmixtures']).astype(np.float32)
    return mean_params
