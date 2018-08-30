import sys
import time
import os
from math import sqrt, cos, sin
import numpy as np
import tensorflow as tf

import utils
from datahandler import datashapes

import pdb

def matching_penalty(opts, pi0, pi, encoded_mean, encoded_logsigma,
                                            pz_mean, pz_sigma,
                                            samples_pz, samples_qz):
    """
    Compute the matching penalty part of the objective function
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    if opts['method']=='swae':
        kl_g, kl_d, match_loss = wae_matching_penalty(opts, pi0, pi,
                                                        samples_pz, samples_qz)
    elif opts['method']=='vae':
        kl_g, kl_d, match_loss = vae_matching_penalty(opts, pi0, pi,
                                                        encoded_mean, encoded_logsigma,
                                                        pz_mean, pz_sigma)
    else:
        assert False, 'Unknown algo %s' % opts['method']
    return kl_g, kl_d, match_loss


def vae_matching_penalty(opts, pi0, pi, encoded_mean, encoded_logsigma,
                                                pz_mean, pz_sigma):
    """
    Compute the VAE's matching penalty
    """
    # Continuous KL
    kl_g = tf.exp(encoded_logsigma) / pz_sigma\
            + tf.square(pz_mean - encoded_mean) / pz_sigma\
            - 1. + tf.log(pz_sigma) - encoded_logsigma
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


def wae_matching_penalty(opts, pi0, pi, samples_pz, samples_qz):
    """
    Compute the WAE's matching penalty
    (add here other penalty if any)
    """
    cont_penalty = mmd_penalty(opts, pi0, pi, samples_pz, samples_qz)

    return None, None, cont_penalty


def mmd_penalty(opts, pi0, pi, sample_pz, sample_qz):
    """
    Compute the MMD penalty for WAE
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    # Compute MMD
    MMD = mmd(opts, pi0, pi, sample_pz, sample_qz)
    if opts['sqrt_MMD']:
        MMD_penalty = tf.exp(tf.log(MMD+1e-8)/2.)
    else:
        MMD_penalty = MMD
    return MMD_penalty


def mmd(opts, pi0, pi, sample_pz, sample_qz):
    """
    Compute MMD between prior and aggregated posterior
    pi0: prior weights [K]
    pi: variational weights [batch,K]
    """
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = utils.get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2,tf.int32)
    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=-1, keepdims=True)
    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=-1, keepdims=True)
    distances_pz = square_dist(opts, sample_pz, norms_pz, sample_pz, norms_pz)
    distances_qz = square_dist(opts, sample_qz, norms_qz, sample_qz, norms_qz)
    distances = square_dist(opts, sample_qz, norms_qz, sample_pz, norms_pz)

    if kernel == 'RBF':
        assert False, 'To implement'
        # Median heuristic for the sigma^2 of Gaussian kernel
        sigma2_k = tf.nn.top_k(
            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        sigma2_k += tf.nn.top_k(
            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

        if opts['verbose']:
            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')

        # First 2 terms of the MMD
        self.res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        self.res1 = tf.multiply(tf.transpose(self.res1),tf.transpose(self.enc_mixweight))
        self.res1 = tf.multiply(tf.transpose(self.res1),tf.transpose(self.enc_mixweight))
        self.res1 += tf.exp( - distances_pz / 2. / sigma2_k) / (opts['nmixtures']*opts['nmixtures'])
        # Correcting for diagonal terms
        self.res1_diag = tf.diag_part(tf.reduce_sum(self.res1,axis=[1,2]))
        self.res1 = (tf.reduce_sum(self.res1)\
                - tf.reduce_sum(self.res1_diag)) / (nf * nf - nf)
        # Cross term of the MMD
        self.res2 = tf.exp( - distances / 2. / sigma2_k)
        self.res2 =  tf.multiply(tf.transpose(self.res2),tf.transpose(self.enc_mixweight))
        self.res2 = tf.transpose(self.res2) / opts['nmixtures']
        self.res2 = tf.reduce_sum(self.res2) * 2. / (nf * nf)
        stat = self.res1 - self.res2
    elif kernel == 'IMQ':
        # k(x, y) = C / (C + ||x - y||^2)
        Cbase = 2 * opts['zdim'] * sigma2_p
        res = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            # First 2 terms of the MMD
            res1_qz = C / (C + distances_qz)
            res1_qz = tf.transpose( tf.multiply(tf.transpose(res1_qz),
                                    tf.transpose(pi)))
            res1_qz = tf.multiply(res1_qz, tf.transpose(pi))
            res1_pz = (C / (C + distances_pz))
            res1_pz = tf.transpose( tf.multiply(tf.transpose(res1_pz),
                                    tf.expand_dims(pi0,axis=-1)))
            res1_pz = tf.multiply(res1_pz, tf.expand_dims(pi0,axis=-1))
            res1 = res1_qz + res1_pz
            # Correcting for diagonal terms
            res1_diag = tf.trace(tf.reduce_sum(res1,axis=[1,2]))
            res1 = (tf.reduce_sum(res1) - res1_diag) / (nf * nf - nf)
            # Cross term of the MMD
            res2 = C / (C + distances)
            res2 = tf.transpose( tf.multiply(tf.transpose(res2),
                                    tf.transpose(pi)))
            res2 = tf.multiply(res2, tf.expand_dims(pi0,axis=-1))
            # res2 =  tf.multiply(tf.transpose(res2),tf.transpose(self.enc_mixweight))
            # res2 = tf.transpose(res2) / opts['nmixtures']
            res2 = tf.reduce_sum(res2) / (nf * nf)
            res += res1 - 2. * res2
    else:
        raise ValueError('%s Unknown kernel' % kernel)
    return res


def square_dist(opts, sample_x, norms_x, sample_y, norms_y):
    """
    Wrapper to compute square distance
    """
    dotprod = tf.tensordot(sample_x, tf.transpose(sample_y), [[-1],[0]])
    reshape =  [-1,opts['nmixtures'],1,1]
    distances = tf.reshape(norms_x, reshape) + tf.transpose(norms_y) - 2. * dotprod
    return distances


def reconstruction_loss(opts, pi, x1, x2):
    """
    Compute the reconstruction part of the objective function
    """
    if opts['method']=='swae':
        loss = wae_recons_loss(opts, pi, x1, x2)
    elif opts['method']=='vae':
        loss = vae_recons_loss(opts, pi, x1, x2)
    return loss


def wae_recons_loss(opts, pi, x1, x2):
    """
    Compute the WAE's reconstruction losses
    pi: weights
    x1: image data             [batch,im_dim]
    x2: image reconstruction   [batch,K,im_dim]
    """
    # Data shape
    shpe = datashapes[opts['dataset']]
    # Continuous cost
    cont_real = tf.expand_dims(x1,axis=1)
    cont_recon = x2
    cont_cost = continous_cost(opts, cont_real, cont_recon)
    # Compute loss
    loss = tf.multiply(cont_cost, pi)
    loss = tf.reduce_sum(loss,axis=-1)
    loss = .05 * tf.reduce_mean(loss) #coef: .2 for L2 and L1, .05 for L2sqr,
    return loss


def continous_cost(opts, x1, x2):
    if opts['cost'] == 'l2':
        # c(x,y) = ||x - y||_2
        c = tf.reduce_sum(tf.square(x1 - x2), axis=[2,3,4])
        c = tf.sqrt(1e-10 + c)
    elif opts['cost'] == 'l2sq':
        # c(x,y) = ||x - y||_2^2
        c = tf.reduce_sum(tf.square(x1 - x2), axis=[2,3,4])
    elif opts['cost'] == 'l1':
        # c(x,y) = ||x - y||_1
        c = tf.reduce_sum(tf.abs(x1 - x2), axis=[2,3,4])
    else:
        assert False, 'Unknown cost function %s' % opts['cost']
    return c


def vae_recons_loss(opts, pi, x1, x2):
    """
    Compute the VAE's reconstruction losses
    """
    real = tf.expand_dims(x1,axis=1)
    logit = x2
    eps = 1e-10
    l = real*tf.log(eps+logit) + (1-real)*tf.log(eps+1-logit)
    loss = tf.reduce_sum(l,axis=[2,3,4])
    loss = tf.reduce_sum(tf.multiply(loss,pi))
    loss = tf.reduce_mean(loss)
    return -loss


def moments_loss(prior_samples, model_samples):
    # Matching the first 2 moments (mean and covariance)
    # Means
    qz_means = tf.reduce_mean(model_samples, axis=0, keepdims=True)
    pz_mean = tf.reduce_mean(prior_samples, axis=0, keepdims=True)
    mean_loss = tf.reduce_sum(tf.square(qz_means - pz_mean),axis=-1)
    mean_loss = tf.reduce_mean(mean_loss)
    # Covariances
    qz_covs = tf.reduce_mean(tf.square(model_samples-qz_means),axis=0)
    pz_cov = tf.reduce_mean(tf.square(prior_samples-pz_mean),axis=0)
    cov_loss = tf.reduce_sum(tf.square(qz_covs - pz_cov),axis=-1)
    cov_loss = tf.reduce_mean(cov_loss)
    # Loss
    pre_loss = mean_loss + cov_loss
    return pre_loss
