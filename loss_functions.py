import sys
import time
import os
from math import sqrt, cos, sin, pow
import numpy as np
import tensorflow as tf

import utils
from datahandler import datashapes

import pdb

def matching_penalty(opts, pi0, pi, encoded_mean, encoded_sigma,
                                            pz_mean, pz_sigma,
                                            samples_pz, samples_qz):
    """
    Compute the matching penalty part of the objective function
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    if opts['method']=='swae':
        kl_g, kl_d, match_loss, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list = wae_matching_penalty(opts, pi0, pi,
                                                        samples_pz, samples_qz)
    elif opts['method']=='vae':
        kl_g, kl_d, match_loss = vae_matching_penalty(opts, pi0, pi,
                                                        encoded_mean, encoded_sigma,
                                                        pz_mean, pz_sigma)
    else:
        assert False, 'Unknown algo %s' % opts['method']
    return kl_g, kl_d, match_loss#, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list


def wae_matching_penalty(opts, pi0, pi, samples_pz, samples_qz):
    """
    Compute the WAE's matching penalty
    (add here other penalty if any)
    """
    cont_penalty, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list = mmd_penalty(opts, pi0, pi, samples_pz, samples_qz)

    return None, None, cont_penalty, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list


def mmd_penalty(opts, pi0, pi, sample_pz, sample_qz):
    """
    Compute the MMD penalty for WAE
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    # Compute MMD
    MMD, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list = mmd(opts, pi0, pi, sample_pz, sample_qz)
    if opts['sqrt_MMD']:
        MMD_penalty = tf.exp(tf.log(MMD+1e-8)/2.)
    else:
        MMD_penalty = MMD
    return MMD_penalty, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list


def mmd(opts, pi0, pi, sample_pz, sample_qz):
    """
    Compute MMD between prior and aggregated posterior
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']

    # Dataset, batch and samples size
    if opts['train_dataset_size']!=-1:
        N = tf.cast(opts['train_dataset_size'], tf.float32)
    else:
        if opts['dataset']=='mnist':
            N = tf.cast(60000, tf.float32)
        else:
            assert False, 'data_set_size unknown. To implement'
    n = utils.get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nb = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2,tf.int32)
    ns = tf.cast(opts['nsamples'], tf.float32)

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=-1, keepdims=False)
    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=-1, keepdims=False)
    distances_pz = square_dist(opts, sample_pz, norms_pz, sample_pz, norms_pz)
    distances_qz = square_dist(opts, sample_qz, norms_qz, sample_qz, norms_qz)
    distances = square_dist(opts, sample_qz, norms_qz, sample_pz, norms_pz)

    if kernel == 'RBF':
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        # First 2 terms of the MMD
        res1_qz = tf.exp( - distances_qz / 2. / sigma2_k)
        shpe = [-1,opts['nmixtures']]
        res1_qz = tf.multiply(res1_qz, tf.reshape(pi,shpe+[1,1]))
        res1_qz = tf.multiply(res1_qz, tf.reshape(pi,[1,1]+shpe))
        res1_pz = tf.exp( - distances_pz / 2. / sigma2_k)
        res1_pz = tf.multiply(res1_pz,tf.reshape(pi0,[1,opts['nmixtures'],1,1]))
        res1_pz = tf.multiply(res1_pz,tf.reshape(pi0,[1,1,1,opts['nmixtures']]))
        res1 = res1_qz + res1_pz
        # Correcting for diagonal terms
        res1_diag = tf.trace(tf.reduce_sum(res1,axis=[1,-1]))
        res1 = (tf.reduce_sum(res1) - res1_diag) / (nb * nb - nb)
        # Cross term of the MMD
        res2 = tf.exp( - distances / 2. / sigma2_k)
        res2 = tf.multiply(res2, tf.reshape(pi,shpe+[1,1]))
        res2 = tf.multiply(res2,tf.reshape(pi0,[1,1,1,opts['nmixtures']]))
        res2 = tf.reduce_sum(res2) / (nb * nb)
        res = res1 - 2. * res2
    elif kernel == 'IMQ':
        # k(x, y) = C / (C + ||x - y||^2)
        shpe = [-1,opts['nmixtures']]
        Cbase = 2 * opts['zdim'] * sigma2_p
        res = 0.
        res_list = []
        base_scale = [1.,2.,5.]
        #scales = [base_scale[i]*pow(10.,j) for j in range(-2,3) for i in range(len(base_scale))]
        scales = [base_scale[i]*pow(10.,j) for j in range(2,-3,-1) for i in range(len(base_scale)-1,-1,-1)]
        #for scale in [.1, .2, .5, 1., 2., 5., 10., 20., 50., 100.]:
        for scale in scales:
            C = Cbase * scale
            # First 2 terms of the MMD
            # pz term
            K_pz = tf.reduce_mean(C / (C + distances_pz),axis=[2,-1])
            K_pz = tf.multiply(K_pz,tf.reshape(pi0,[1,opts['nmixtures'],1,1]))
            K_pz = tf.multiply(K_pz,tf.reshape(pi0,[1,1,1,opts['nmixtures']]))
            res1_pz = tf.reduce_sum(K_pz)
            res1_pz /= (nb * nb)
            res2_pz = tf.trace(tf.reduce_sum(K_pz,axis=[0,2]))
            res2_pz /= ((nb * nb - nb) * nb)
            res3_pz = tf.trace(tf.trace(tf.transpose(K_pz,perm=[0,2,1,3])))
            res3_pz /= (nb * nb - nb)
            res_pz = res1_pz + res2_pz - res3_pz
            # K_pz_trace_K = tf.trace(tf.transpose(K_pz,perm=[0,2,1,3]))
            # res2_pz = tf.reduce_sum(K_pz_trace_K)
            # res2_pz /= ((nb * nb - nb) * nb)
            # res3_pz = tf.trace(K_pz_trace_K)
            # res3_pz /= (nb * nb - nb)
            # res_pz = res1_pz + res2_pz - res3_pz
            # qz term
            K_qz = tf.reduce_mean(C / (C + distances_qz),axis=[2,-1])
            K_qz = tf.multiply(K_qz, tf.reshape(pi,shpe+[1,1]))
            K_qz = tf.multiply(K_qz, tf.reshape(pi,[1,1]+shpe))
            K_qz = tf.reduce_sum(K_qz,axis=[1,3])
            res1_qz = tf.reduce_sum(K_qz)
            #res1_qz /= (ns * ns)
            #res1_qz /= (nb * nb - nb)
            #res1_qz *= (N - 1.) / N
            res2_qz = tf.trace(K_qz)
            #res2_qz /= (ns * ns)
            #res2_qz /= (nb * nb - nb)
            res_qz = res1_qz - res2_qz
            res_qz /= (nb * nb - nb)
            # K_qz_trace_batch = tf.trace(tf.transpose(K_qz,perm=[1,3,0,2]))
            # res2_qz = tf.reduce_sum(K_qz_trace_batch)
            # res2_qz /= (ns * ns)
            # res2_qz /= (nb * nb - nb)
            #res2_qz *= (N - 1.) / N
            # res2_qz *= (ns - N) / ((nb * nb - nb) * N)
            # res3_qz = tf.trace(K_qz_trace_batch)
            # res3_qz /= ((ns * ns - ns) * ns)
            # K_qz_diag = tf.trace(tf.transpose(C / (C + distances_qz),perm=[0,1,3,4,2,5]))
            # K_qz_diag = tf.multiply(K_qz_diag, tf.reshape(pi,shpe+[1,1]))
            # K_qz_diag = tf.multiply(K_qz_diag, tf.reshape(pi,[1,1]+shpe))
            # res4_qz = tf.trace(tf.trace(tf.transpose(K_qz_diag,perm=[1,3,0,2])))
            # res4_qz /= (ns * ns - ns)
            # res4_qz /= nb
            # res4_qz /= N
            # res_qz = res1_qz + res2_qz + res3_qz - res4_qz
            # Cross term of the MMD
            K_qzpz = tf.reduce_mean(C / (C + distances),axis=[2,-1])
            res_qzpz = tf.multiply(K_qzpz, tf.reshape(pi,shpe+[1,1]))
            res_qzpz = tf.multiply(res_qzpz,tf.reshape(pi0,[1,1,1,opts['nmixtures']]))
            res_qzpz = tf.reduce_sum(res_qzpz) / (nb * nb)
            res_list.append(res_pz + res_qz - 2. * res_qzpz)
            res += res_pz + res_qz - 2. * res_qzpz
    else:
        raise ValueError('%s Unknown kernel' % kernel)
    #return res
    res_list = tf.stack(res_list)
    return res, distances_pz, distances_qz, distances, K_pz, K_qz, K_qzpz, res_list


def square_dist(opts, sample_x, norms_x, sample_y, norms_y):
    """
    Wrapper to compute square distance
    """
    norms_x_shape = tf.shape(norms_x)
    nx_reshape = tf.reshape(norms_x,tf.concat([norms_x_shape,
                                              tf.constant([1,1,1])],
                                              axis=-1))
    norms_y_shape = tf.shape(norms_y)
    ny_reshape = tf.reshape(norms_y,tf.concat([tf.constant([1,1,1]),
                                              norms_y_shape],
                                              axis=-1))
    dotprod = tf.tensordot(sample_x, sample_y, [[-1],[-1]])
    distances = nx_reshape + ny_reshape - 2. * dotprod
    #return distances
    return tf.nn.relu(distances)


def vae_matching_penalty(opts, pi0, pi, encoded_mean, encoded_sigma,
                                                pz_mean, pz_sigma):
    """
    Compute the VAE's matching penalty
    """
    # Continuous KL
    kl_g = encoded_sigma / pz_sigma \
            + tf.square(pz_mean - encoded_mean) / pz_sigma - 1. \
            + tf.log(pz_sigma) - tf.log(encoded_sigma)
    kl_g = 0.5 * tf.reduce_sum(kl_g,axis=-1)
    kl_g = tf.multiply(kl_g,pi)
    kl_g = tf.reduce_sum(kl_g,axis=-1)
    kl_g = tf.reduce_mean(kl_g)
    # Discrete KL
    eps = 1e-10
    kl_d = tf.log(eps+pi) - tf.log(pi0)
    kl_d = tf.multiply(kl_d,pi)
    kl_d = tf.reduce_sum(kl_d,axis=-1)
    kl_d = tf.reduce_mean(kl_d)
    return kl_g, kl_d, kl_g + kl_d


def reconstruction_loss(opts, pi, x1, x2):
    """
    Compute the reconstruction part of the objective function
    """
    if opts['method']=='swae':
        loss = wae_recons_loss(opts, pi, x1, x2)
    elif opts['method']=='vae':
        loss = vae_bernoulli_recons_loss(opts, pi, x1, x2)
    return loss


def wae_recons_loss(opts, pi, x1, x2):
    """
    Compute the WAE's reconstruction losses
    pi: weights
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,K,S,im_dim]
    """
    # Data shape
    shpe = datashapes[opts['dataset']]
    data = tf.reshape(x1,[-1,1,1]+shpe)
    if opts['cost'] == 'l2':
        # c(x,y) = ||x - y||_2
        cost = tf.reduce_sum(tf.square(data - x2), axis=[-3,-2,-1])
        cost = tf.sqrt(1e-10 + cost)
        cost = tf.reduce_mean(cost,axis=-1)
    elif opts['cost'] == 'l2sq':
        # c(x,y) = ||x - y||_2^2
        cost = tf.reduce_sum(tf.square(data - x2), axis=[-3,-2,-1])
        cost = tf.reduce_mean(cost,axis=-1)
    elif opts['cost'] == 'l1':
        # c(x,y) = ||x - y||_1
        cost = tf.reduce_sum(tf.abs(data - x2), axis=[-3,-2,-1])
        cost = tf.reduce_mean(cost,axis=-1)
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    # Compute loss
    loss = tf.reduce_sum(tf.multiply(cost, pi),axis=-1)
    loss = 1. * tf.reduce_mean(loss) #coef: .2 for L2 and L1, .05 for L2sqr,
    return loss


def contrast_norm(pics):
    # pics is a [N, H, W, C] tensor
    mean, var = tf.nn.moments(pics, axes=[-3, -2, -1], keep_dims=True)
    return pics / tf.sqrt(var + 1e-08)


def vae_bernoulli_recons_loss(opts, pi, x1, x2):
    """
    Compute the VAE's reconstruction losses
    with bernoulli observation model
    """
    data_shape = datashapes[opts['dataset']]
    real = tf.reshape(x1,shape=[-1,1,1] + data_shape)
    logit = x2
    eps = 1e-10
    l = real*tf.log(eps+logit) + (1-real)*tf.log(eps+1-logit)
    loss = tf.reduce_sum(l,axis=[-3,-2,-1])
    loss = tf.reduce_mean(loss,axis=-1)
    loss = tf.reduce_sum(tf.multiply(loss,pi),axis=-1)
    loss = tf.reduce_mean(loss)
    return -loss


# def vae_betabinomial_recons_loss(opts, pi, x1, x2):
#     """
#     Compute the VAE's reconstruction losses
#     with beta-binomial observation model
#     """
#     real = tf.expand_dims(x1,axis=1)
#     alpha, beta = tf.split(mean_params,2,axis=-1)
#
#     eps = 1e-10
#     l = real*tf.log(eps+logit) + (1-real)*tf.log(eps+1-logit)
#     loss = tf.reduce_sum(l,axis=[-3,-2,-1])
#     loss = tf.reduce_sum(tf.multiply(loss,pi))
#     loss = tf.reduce_mean(loss)
#     return -loss


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
