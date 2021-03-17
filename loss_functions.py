import sys
import time
import os

import numpy as np
import tensorflow as tf
from math import pi
from scipy.stats import moment

import utils
from datahandler import datashapes

import pdb


### latent regularization ###
def MMD(opts, resp_qz, sample_qz, resp_pz, sample_pz):
    """
    Compute MMD between prior and aggregated posterior
    resp_pz: prior mixture resp. [K]
    resp_qz: variational mixture resp. [batch,K]
    sample_qz/sample_pz: latent samples [batch,K,zdim]
    """

    K, zdim = sample_qz.get_shape().as_list()[1:]
    nf = tf.cast(utils.get_batch_size(sample_qz), tf.float32)
    half_size = tf.cast((nf * nf - nf) / 2, tf.int32)
    # reshape resp_pz to be broadcastable along batch dim
    resp_pz = tf.expand_dims(resp_pz,axis=0) #[1,K]
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
        resp_qz_broadcast_1 = tf.expand_dims(tf.expand_dims(resp_qz,axis=2),axis=2) #[batch,K,1,1]
        resp_qz_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(resp_qz),axis=0),axis=0) #[1,1,K,batch]
        res_q *= resp_qz_broadcast_1 * resp_qz_broadcast_2
        # p term
        res_p = tf.exp( - distances_pz / 2. / sigma2_k)
        resp_pz_broadcast_1 = tf.expand_dims(tf.expand_dims(resp_pz,axis=2),axis=2) #[batch,K,1,1]
        resp_pz_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(resp_pz),axis=0),axis=0) #[1,1,K,batch]
        res_p *= resp_pz_broadcast_1 * resp_pz_broadcast_2
        #correction term
        res1 = tf.reduce_sum(res_q+res_p) - tf.linalg.trace(tf.reduce_sum(res_q+res_p,axis=[1,2]))
        res1 /= nf * nf - nf
        # cross term
        res_qp = tf.exp( - distances / 2. / sigma2_k)
        res_qp *= resp_qz_broadcast_1 * resp_pz_broadcast_2
        res2 = tf.reduce_sum(res_qp) / (nf * nf)
        # mmd
        res = res1 - 2. * res2
    elif opts['mmd_kernel'] == 'IMQ':
        # k(x, y) = C / (C + ||x - y||^2)
        Cbase = 2 * zdim * ((opts['x_var']+opts['x_var']) / 2.)**2
        res = 0.
        # for scale in [.1, .2, .5, 1., 2., 5., 10., 20., 50., 100.]:
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            # q term
            res_q = C / (C + distances_qz)
            resp_qz_broadcast_1 = tf.expand_dims(tf.expand_dims(resp_qz,axis=2),axis=2) #[batch,K,1,1]
            resp_qz_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(resp_qz),axis=0),axis=0) #[1,1,K,batch]
            res_q *= resp_qz_broadcast_1 * resp_qz_broadcast_2
            # p term
            res_p = C / (C + distances_pz)
            resp_pz_broadcast_1 = tf.expand_dims(tf.expand_dims(resp_pz,axis=2),axis=2) #[batch,K,1,1]
            resp_pz_broadcast_2 = tf.expand_dims(tf.expand_dims(tf.transpose(resp_pz),axis=0),axis=0) #[1,1,K,batch]
            res_p *= resp_pz_broadcast_1 * resp_pz_broadcast_2
            #correction term
            res1 = tf.reduce_sum(res_q+res_p) - tf.linalg.trace(tf.reduce_sum(res_q+res_p,axis=[1,2]))
            res1 /= nf * nf - nf
            # cross term
            res_qp = C / (C + distances)
            res_qp *= resp_qz_broadcast_1 * resp_pz_broadcast_2
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


def KL(opts, resp_qz, mean_qz, sigma_qz, resp_pz, mean_pz, sigma_pz):
    """
    Compute kl(qz,pz) with qz,pz MOG

    resp_qz:              [batch,K]
    resp_pz:              [K,]
    mean_qz/sigma_qz:   [batch,K,zdim]
    mean_pz/sigma_pz:   [K,zdim]
    """

    g_kl = gauss_kl(mean_qz, sigma_qz, mean_pz, sigma_pz) #[batch,K]
    g_kl = tf.reduce_sum(resp_qz*g_kl, axis=-1)
    c_kl = cat_kl(resp_qz+1e-10, resp_pz) #[batch,K]
    c_kl = tf.reduce_sum(c_kl, axis=-1)

    return tf.reduce_mean(c_kl+g_kl), tf.reduce_mean(g_kl), tf.reduce_mean(c_kl)

def gauss_kl(mean_qz, sigma_qz, mean_pz, sigma_pz):
    """
    Compute kl between gaussian components qz[:,k], pz[:,k]

    mean_qz/sigma_qz: [batch,K,zdim]
    resp_qz: [batch,K]
    mean_pz: [K,zdim]
    sigma_pz: [K,zdim] or [K,zdim,zdim] for full cov
    resp_pz: [K,]
    """
    mean_pz = tf.expand_dims(mean_pz, axis=0)
    sigma_pz = tf.expand_dims(sigma_pz, axis=0)
    if len(sigma_pz.shape)==3:
        # diagonal cov
        cov_ratio = sigma_qz / sigma_pz
        mean_sqr_dist = tf.square(mean_qz - mean_pz)
        kl = cov_ratio + mean_sqr_dist/sigma_pz - tf.math.log(cov_ratio) - 1.
        return 0.5 * tf.reduce_sum(kl, axis=-1)
    else:
        # full cov
        sigma_pz_shape = sigma_pz.shape
        K = sigma_pz_shape[1]
        zdimf = tf.cast(sigma_pz_shape[-1], tf.float32)
        zdim = int(sigma_pz_shape[-1])
        eye = 1e-10* tf.compat.v1.linalg.eye(num_rows=zdim, num_columns=zdim,
                                    batch_shape=sigma_pz_shape[:2])
        sigma_pz_invert = tf.linalg.inv(sigma_pz + eye)
        mean_diff = tf.expand_dims(mean_qz - mean_pz,axis=-1)
        mean_square_dist = tf.linalg.matmul(sigma_pz_invert,mean_diff)
        mean_square_dist = tf.linalg.matmul(
                                    tf.transpose(mean_diff,perm=(0,1,3,2)),
                                    mean_square_dist)
        mean_square_dist = tf.reshape(mean_square_dist,[-1,K])
        log_det = tf.linalg.logdet(sigma_pz) - tf.reduce_sum(tf.math.log(sigma_qz), axis=-1)
        kl = tf.linalg.trace(tf.linalg.matmul(sigma_pz_invert,tf.linalg.diag(sigma_qz))) + \
                mean_square_dist + log_det - zdimf
        return 0.5 * kl

def cat_kl(prob0, prob1):
    """
    Compute kl between cat. distribution with probs prob0 and prob1
    """
    prob1 = tf.expand_dims(prob1, axis=0)
    log_ratio = tf.math.log(prob0 / prob1)

    return prob0 * log_ratio


### reconstructions loss ###
def ground_cost(opts, inputs, resp, reconstruction):
    """
    Compute the WAE's reconstruction losses
    inputs: image data  [batch,imgdim]
    rec: image data     [batch,K,imgdim]
    resp: mixture resp. [batch,K]
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
    loss = tf.reduce_sum(cost*resp,axis=-1)

    return tf.reduce_mean(loss)

def cross_entropy_loss(opts, inputs, mean, sigma, resp):
    """
    Compute the VAE's reconstruction loss.
    Implementation for Bernoulli or Gaussian p(x|y)
    inputs: image data          [batch,imgdim]
    mean/sigma: decoder params  [batch,K,imgdim]
    resp: mixture resp.         [batch,K]
    """
    if opts['decoder']=='bernoulli':
        xent = cat_cross_entropy(inputs, mean)
    elif opts['decoder']=='gauss':
        xent = gauss_cross_entropy(inputs, mean, sigma)
    else:
        ValueError('Incompatibel {} decoder for VAE'.format(opts['decoder']))
    loss = tf.reduce_sum(resp*xent, axis=-1)

    return tf.reduce_mean(loss)

def cat_cross_entropy(inputs, reconstruction):
    """
    Compute the xent for bernouill decoder
    inputs:[batch,im_dim]
    reconstruction: [batch,K,prod(imgdim)]
    """
    K = reconstruction.get_shape().as_list()[1]
    inputs = tf.tile(tf.expand_dims(tf.compat.v1.layers.flatten(inputs),axis=1),
                                    [1,K,1])
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs,
                                    logits=reconstruction)
    return tf.reduce_sum(xent,axis=-1)

def gauss_cross_entropy(inputs, mean, sigma):
    """
    Compute the xent for gaussian decoder
    inputs:[batch,im_dim]
    mean/sigma: [batch,K,prod(imgdim)]
    """
    inputs = tf.expand_dims(tf.compat.v1.layers.flatten(inputs), axis=1)
    loss = tf.math.log(pi*sigma) + tf.square(inputs-mean) / sigma

    return 0.5 * tf.reduce_sum(loss, axis=-1)


### pre-training loss###
def moments_loss_empirical_pz(sample_qz, sample_pz):
    """
    Matching the first 2 empirical moments (mean and covariance)
    of prior and aggregated post.

    sample_qz/pz: [batch,K,zdim]
    Return:
    sum_k ||muq[k]-mup[k]||^2 + ||covq[k]-covp[k]||^2 where
    mu[k] = 1/N sum_n sample[n,k]
    cov[k] = 1

    """
    mean_qz = tf.reduce_mean(sample_qz, axis=0, keepdims=True) #[1,K,zdim]
    mean_pz = tf.reduce_mean(sample_pz, axis=0, keepdims=True) #[1,K,zdim]
    mean_loss = tf.reduce_sum(tf.square(mean_qz - mean_pz),axis=[0,-1]) #[K,]
    hat_qz = tf.expand_dims(sample_qz-mean_qz, axis=-1) #[batch,K,zdim,1]
    cov_qz = tf.reduce_mean(tf.linalg.matmul(hat_qz,hat_qz,transpose_b=True),axis=0) #[K,zdim,zdim]
    hat_pz = tf.expand_dims(sample_pz-mean_pz, axis=-1) #[batch,K,zdim,1]
    cov_pz = tf.reduce_mean(tf.linalg.matmul(hat_pz,hat_pz,transpose_b=True),axis=0) #[K,zdim,zdim]
    cov_loss = tf.reduce_sum(tf.square(cov_qz-cov_pz),axis=[-2,-1]) #[K,]
    loss = mean_loss + cov_loss

    return tf.reduce_sum(loss)

def moments_loss(sample_qz, pz_mean, pz_sigma):
    """
    Matching the first 2 moments (mean and covariance)
    of prior and aggregated post, using true prior parameters.

    sample_qz: [batch,K,zdim]
    pz_mean/pz_sigma: [K,zdim]
    Return:
    sum_k ||muq[k]-mup[k]||^2 + ||covq[k]-covp[k]||^2 where
    mu[k] = 1/N sum_n sample[n,k]
    cov[k] = 1

    """
    mean_qz = tf.reduce_mean(sample_qz, axis=0, keepdims=True) #[1,K,zdim]
    mean_pz = tf.expand_dims(pz_mean, axis=0) #[1,K,zdim]
    mean_loss = tf.reduce_sum(tf.square(mean_qz - mean_pz),axis=[0,-1]) #[K,]
    hat_qz = tf.expand_dims(sample_qz-mean_qz, axis=-1) #[batch,K,zdim,1]
    cov_qz = tf.reduce_mean(tf.linalg.matmul(hat_qz,hat_qz,transpose_b=True),axis=0) #[K,zdim,zdim]
    if len(pz_sigma.shape)==2:
        cov_pz = tf.expand_dims(tf.linalg.diag(pz_sigma),axis=0)#[K,zdim,zdim]
    else:
        cov_pz = pz_sigma
    cov_loss = tf.reduce_sum(tf.square(cov_qz-cov_pz),axis=[-2,-1]) #[K,]
    loss = mean_loss + cov_loss

    return tf.reduce_sum(loss)
