import sys
import time
import os
from math import sqrt, pi, cos, sin, pow
import numpy as np
import tensorflow as tf

import utils
from datahandler import datashapes

import pdb


### latent regularization ###
def MMD(opts, pi, sample_qz, pi0, sample_pz):
    """
    Compute MMD between prior and aggregated posterior
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    sample_qz/sample_pz: latent samples [batch,K,zdim]
    """

    K, zdim = sample_qz.get_shape().as_list()[1:]
    n = utils.get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2,tf.int32)
    # reshape pi0 to be broadcastable along batch dim
    pi0 = tf.expand_dims(pi0,axis=0) #[1,K]
    # get pairwise distances
    distances_pz = square_dist(sample_pz, sample_pz) #[batch,K,K,batch]
    distances_qz = square_dist(sample_qz, sample_qz) #[batch,K,K,batch]
    distances = square_dist(sample_qz, sample_pz) #[batch,K,K,batch]


    if opts['mmd_kernel'] == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel [K,]
        sigma2_k = tf.nn.top_k(tf.reshape(distances,[-1]),
                                    half_size).values[:,half_size - 1]
        sigma2_k += tf.nn.top_k(tf.reshape(distances_qz,[-1]),
                                    half_size).values[:,half_size - 1]
        # q term
        res_q = tf.exp( - distances_qz / 2. / sigma2_k)
        pi_broadcast_1 = tf.expand_dims(tf.expand_dims(pi,axis=2),axis=2) #[batch,K,1,1]
        pi_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(pi),axis=0),axis=0) #[1,1,K,batch]
        res_q *= pi_broadcast_1 * pi_broadcast_2
        # p term
        res_p = tf.exp( - distances_pz / 2. / sigma2_k)
        pi0_broadcast_1 = tf.expand_dims(tf.expand_dims(pi0,axis=2),axis=2) #[batch,K,1,1]
        pi0_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(pi0),axis=0),axis=0) #[1,1,K,batch]
        res_p *= pi0_broadcast_1 * pi0_broadcast_2
        #correction term
        res1 = tf.reduce_sum(res_q+res_p) - tf.linalg.trace(tf.reduce_sum(res_q+res_p,axis=[1,2]))
        res1 /= nf * nf - nf
        # cross term
        res_qp = tf.exp( - distances / 2. / sigma2_k)
        res_qp *= pi_broadcast_1 * pi0_broadcast_2
        res2 = tf.reduce_sum(res_qp) / (nf * nf)
        # mmd
        res = res1 - 2. * res2
    elif opts['mmd_kernel'] == 'IMQ':
        # k(x, y) = C / (C + ||x - y||^2)
        Cbase = 2 * zdim * opts['pz_scale']**2
        res = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10., 20., 50., 100.]:
            C = Cbase * scale
            # q term
            res_q = C / (C + distances_qz)
            pi_broadcast_1 = tf.expand_dims(tf.expand_dims(pi,axis=2),axis=2) #[batch,K,1,1]
            pi_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(pi),axis=0),axis=0) #[1,1,K,batch]
            res_q *= pi_broadcast_1 * pi_broadcast_2
            # p term
            res_p = C / (C + distances_pz)
            pi0_broadcast_1 = tf.expand_dims(tf.expand_dims(pi0,axis=2),axis=2) #[batch,K,1,1]
            pi0_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(pi0),axis=0),axis=0) #[1,1,K,batch]
            res_p *= pi0_broadcast_1 * pi0_broadcast_2
            #correction term
            res1 = tf.reduce_sum(res_q+res_p) - tf.linalg.trace(tf.reduce_sum(res_q+res_p,axis=[1,2]))
            res1 /= nf * nf - nf
            # cross term
            res_qp = C / (C + distances)
            res_qp *= pi_broadcast_1 * pi0_broadcast_2
            res2 = tf.reduce_sum(res_qp) / (nf * nf)
            # mmd
            res += res1 - 2. * res2
    else:
        raise ValueError('%s Unknown kernel' % opts['mmd_kernel'])

    return res

def square_dist(p, q):
    """
    Wrapper to compute square distances within each mixtures:
    ||x_n,k - y_m,k||^2 for n,m batch and k mixtures

    p/q: samples   [batch,K,zdim]
    Return:
    dist    [batch,K,K,batch]
    """

    p = tf.expand_dims(tf.expand_dims(p,axis=2),axis=2) #[batch,K,1,1,zdim]
    q = tf.transpose(q,perm=[1,0,2]) #[K,batch,zdim]
    q = tf.expand_dims(tf.expand_dims(q,axis=0),axis=0) #[1,1,K,batch,zdim]
    dist = tf.reduce_sum(tf.square(p-q),axis=-1)

    return dist


def KL(opts, qz_pi, qz_mean, qz_sigma, pz_pi, pz_mean, pz_sigma):
    """
    Compute kl(qz,pz) with qz,pz MOG

    qz_pi:              [batch,K]
    pz_pi:              [K,]
    qz_mean/qz_sigma:   [batch,K,zdim]
    pz_mean/pz_sigma:   [K,zdim]
    """

    g_comp_kl = gauss_kl(qz_mean, qz_sigma, pz_mean, pz_sigma) #[batch,K]
    g_kl = tf.reduce_sum(qz_pi*g_kl, axis=-1)
    c_kl = cat_kl(qz_pi, pz_pi)
    c_kl = tf.reduce_sum(c_kl, axis=-1)

    return tf.reduce_mean(c_kl+g_kl), tf.reduce_mean(g_kl), tf.reduce_mean(c_kl)

def gauss_kl(qz_mean, qz_sigma, pz_mean, pz_sigma):
    """
    Compute kl between gaussian components qz[:,k], pz[:,k]
    """
    pz_mean = tf.expand_dims(pz_mean, axis=0)
    pz_sigma = tf.expand_dims(pz_sigma, axis=0)
    cov_ratio = qz_sigma / pz_sigma
    mean_sqr_dist = tf.square(qz_mean - pz_mean)
    kl = mean_sqr_dist + mean_sqr_dist/pz_sigma - tf.math.log(cov_ratio) - 1.

    return 0.5 * tf.reduce_sum(kl, axis=-1)

def cat_kl(pi0, pi1):
    """
    Compute kl between cat. distribution with probs pi0 and pi1
    """
    pi1 = tf.expand_dims(pi1, axis=0)
    log_ratio = tf.log(pi0 / pi1)
    kl = pi0 * log_ratio

    return tf.reduce_sum(kl, axis=-1)


### reconstructions loss ###
def ground_cost(opts, inputs, pi, reconstruction):
    """
    Compute the WAE's reconstruction losses
    inputs: image data  [batch,imgdim]
    rec: image data     [batch,K,imgdim]
    pi: mixture weights [batch,K]
    """

    inputs = tf.expand_dims(inputs,axis=1)
    if opts['cost'] == 'l2':
        # c(x,y) = ||x - y||_2
        cost = tf.reduce_sum(tf.square(inputs-reconstruction), axis=[-3,-2,-1])
        cost = tf.sqrt(1e-10 + cost)
    elif opts['cost'] == 'l2sq':
        # c(x,y) = ||x - y||_2^2
        cost = tf.reduce_sum(tf.square(inputs-reconstruction), axis=[-3,-2,-1])
    elif opts['cost'] == 'l1':
        # c(x,y) = ||x - y||_1
        cost = tf.reduce_sum(tf.abs(inputs-reconstruction), axis=[-3,-2,-1])
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    loss = tf.reduce_sum(cost*pi,axis=-1)

    return tf.reduce_mean(loss)

def cross_entropy_loss(opts, inputs, mean, sigma, pi):
    """
    Compute the VAE's reconstruction loss.
    Implementation for Bernoulli or Gaussian p(x|y)
    inputs: image data          [batch,imgdim]
    mean/sigma: decoder params  [batch,K,imgdim]
    pi: mixture weights         [batch,K]
    """

    if opts['decoder']=='bernoulli':
        xent = cat_cross_entropy(inputs, mean)
    elif opts['decoder']=='gauss':
        xent = gauss_cross_entropy(inputs, mean, sigma)
    else:
        ValueError('Incompatibel {} decoder for VAE'.format(opts['decoder']))
    loss = - tf.reduce_sum(pi*xent, axis=-1)

    return tf.reduce_mean(loss)

def cat_cross_entropy(inputs, reconstruction):
    """
    Compute the xent for bernouill decoder
    inputs:[batch,im_dim]
    reconstruction: [batch,K,prod(imgdim)]
    """
    inputs = tf.expand_dims(tf.flatten(inputs), axis=1)

    return tf.nn.sigmoid_cross_entropy_with_logits(inputs, reconstruction)

def gauss_cross_entropy(inputs, mean, sigma):
    """
    Compute the xent for gaussian decoder
    inputs:[batch,im_dim]
    mean/sigma: [batch,K,prod(imgdim)]
    """
    inputs = tf.expand_dims(tf.flatten(inputs), axis=1)
    loss = tf.log(pi*sigma) + tf.square(inputs-mean) / sigma

    return -0.5 * tf.reduce_sum(loss, axis=-1)


### pre-training loss###
def moments_loss(prior_samples, model_samples):
    # Matching the first 2 moments (mean and covariance)
    # Means
    #qz_means = tf.reduce_mean(model_samples, axis=0, keepdims=True)
    qz_means = tf.reduce_mean(model_samples, axis=[0,2], keepdims=True)
    #pz_mean = tf.reduce_mean(prior_samples, axis=0, keepdims=True)
    pz_mean = tf.reduce_mean(prior_samples, axis=[0,2], keepdims=True)
    mean_loss = tf.reduce_sum(tf.square(qz_means - pz_mean),axis=-1)
    mean_loss = tf.reduce_mean(mean_loss)
    # Covariances
    qz_covs = tf.reduce_mean(tf.square(model_samples-qz_means),axis=[0,2])
    pz_cov = tf.reduce_mean(tf.square(prior_samples-pz_mean),axis=[0,2])
    # qz_covs = tf.reduce_mean(tf.square(model_samples-qz_means),axis=0)
    # pz_cov = tf.reduce_mean(tf.square(prior_samples-pz_mean),axis=0)
    cov_loss = tf.reduce_sum(tf.square(qz_covs - pz_cov),axis=-1)
    cov_loss = tf.reduce_mean(cov_loss)
    # Loss
    pre_loss = mean_loss + cov_loss
    return pre_loss
