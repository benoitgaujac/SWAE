# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

""" Wasserstein Auto-Encoder models

"""

import sys
import time
import os
from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf
import logging
import ops
import utils
from model_nn import encoder, decoder
from kernel_nn import k_encoder, k_decoder
from datahandler import datashapes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as scistats
import umap

import pdb

class WAE(object):

    def __init__(self, opts):

        logging.error('Building the Tensorflow Graph')

        self.sess = tf.Session()
        self.opts = opts

        # --- Some of the parameters for future use
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # --- Placeholders
        self.add_model_placeholders()
        self.add_training_placeholders()
        sample_size = tf.shape(self.sample_points,out_type=tf.int64)[0]

        self.init_prior()

        # --- Transformation ops
        # Encode the content of sample_points placeholder
        if opts['e_means']=='fixed':
            _, _, enc_logmixweight = encoder(opts, inputs=self.sample_points,
                                                            is_training=self.is_training)
            self.enc_mixweight = tf.nn.softmax(enc_logmixweight,axis=-1)
            eps = tf.zeros([tf.cast(sample_size,dtype=tf.int32),opts['nmixtures'],opts['zdim']],dtype=tf.float32)
            self.enc_mean = self.pz_means + eps
            self.enc_logsigmas = opts['init_e_std']*tf.ones([tf.cast(sample_size,dtype=tf.int32),opts['nmixtures'],opts['zdim']],dtype=tf.float32)
        elif opts['e_means']=='mean':
            enc_mean, _, enc_logmixweight = encoder(opts, inputs=self.sample_points,
                                                            is_training=self.is_training)
            self.debug_mix = enc_logmixweight
            self.enc_mixweight = tf.nn.softmax(enc_logmixweight,axis=-1)
            self.enc_mean = enc_mean
            self.enc_logsigmas = opts['init_e_std']*tf.ones([tf.cast(sample_size,dtype=tf.int32),opts['nmixtures'],opts['zdim']],dtype=tf.float32)
        elif opts['e_means']=='learnable':
            enc_mean, enc_logsigmas, enc_logmixweight = encoder(opts, inputs=self.sample_points,
                                                            is_training=self.is_training)
            self.debug_mix = enc_logmixweight
            self.enc_mixweight = tf.nn.softmax(enc_logmixweight,axis=-1)
            self.enc_mean = enc_mean
            enc_logsigmas = tf.clip_by_value(enc_logsigmas, -50, 50)
            self.enc_logsigmas = enc_logsigmas
        # Encoding all mixtures
        self.mixtures_encoded = self.sample_mixtures(self.enc_mean,
                                            tf.exp(self.enc_logsigmas),
                                            opts['e_noise'],sample_size,'tensor')
        # select mixture components according to the encoded mixture weights
        idx = tf.reshape(tf.multinomial(enc_logmixweight, 1),[-1])
        rng = tf.range(sample_size)
        zero = tf.zeros([tf.cast(sample_size,dtype=tf.int32)],dtype=tf.int64)
        mix_idx = tf.stack([rng,idx],axis=-1)
        self.encoded_means = tf.gather_nd(self.enc_mean,mix_idx)
        mix_idx = tf.stack([rng,idx,zero],axis=-1)
        self.encoded = tf.gather_nd(self.mixtures_encoded,mix_idx)
        # Decode the all points encoded above (i.e. reconstruct)
        noise = tf.reshape(self.mixtures_encoded,[-1,opts['zdim']])
        self.reconstructed, self.reconstructed_logits = decoder(opts, noise=noise,
                                                    is_training=self.is_training)
        self.reconstructed = tf.reshape(self.reconstructed,
                                        [-1,opts['nmixtures'],opts['nsamples']]+self.data_shape)
        self.reconstructed_logits = tf.reshape(self.reconstructed_logits,
                                        [-1,opts['nmixtures'],opts['nsamples']]+self.data_shape)
        # Decode the point sampled from multinomial
        self.one_recons = tf.gather_nd(self.reconstructed,mix_idx)
        self.one_recons_logits = tf.gather_nd(self.reconstructed_logits,mix_idx)
        # Decode the content of sample_noise
        self.decoded, self.decoded_logits = decoder(opts, reuse=True, noise=self.sample_noise,
                                                                is_training=self.is_training)
        # --- Objectives, losses, penalties, pretraining
        # Compute reconstruction cost
        self.loss_reconstruct = self.reconstruction_loss()
        # Compute matching penalty cost
        self.penalty = self.matching_penalty(self.sample_mix_noise,self.mixtures_encoded)
        # Compute wae obj
        self.objective = self.loss_reconstruct \
                        + self.lmbd * self.penalty
        # Compute entropy of mixture weights if needed
        if opts['entropy']:
            mean_mixweight = tf.reduce_sum(self.enc_mixweight,axis=0)
            h = tf.multiply(mean_mixweight,tf.log(mean_mixweight))
            self.H = - tf.reduce_sum(h)
            self.objective = self.objective - self.H_lambda * self.H
        else:
            self.H = None
        # Add pretraining
        if opts['e_pretrain']:
            self.loss_pretrain = self.pretrain_loss()
        else:
            self.loss_pretrain = None

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        data = tf.placeholder(
            tf.float32, [None] + shape, name='real_points_ph')
        noise = tf.placeholder(
            tf.float32, [None] + [opts['zdim']], name='noise_ph')
        mix_noise = tf.placeholder(
            tf.float32, [None] + [opts['nmixtures'],opts['nsamples'],opts['zdim']], name='mix_noise_ph')

        self.sample_points = data
        self.sample_noise = noise
        self.sample_mix_noise = mix_noise

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        lmbda = tf.placeholder(tf.float32, name='lambda')

        self.lr_decay = decay
        self.is_training = is_training
        self.lmbd = lmbda
        self.H_lambda = opts['h_lambda']

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_logsigmas)
        if opts['e_noise'] == 'implicit':
            tf.add_to_collection('encoder_A', self.encoder_A)
        tf.add_to_collection('encoder', self.encoded)
        tf.add_to_collection('decoder', self.decoded)
        tf.add_to_collection('lambda', self.lmbd)
        self.saver = saver

    def init_prior(self):
        opts = self.opts
        distr = opts['pz']
        if distr == 'uniform':
            self.pz_means = [-1.0,1.0].astype(np.float32)
            self.pz_covs = None
        elif distr == 'mixture':
            if opts['zdim']==2:
                if opts['dataset']=='mnist' and opts['nmixtures']==10:
                    means = set_2d_priors(opts['nmixtures'])
                else:
                    means = np.random.uniform(-1.,1.,(opts['nmixtures'], opts['zdim'])).astype(np.float32)
            else:
                if opts['zdim']+1>=opts['nmixtures']:
                    means = np.zeros([opts['nmixtures'], opts['zdim']],dtype='float32')
                    for k in range(opts['nmixtures']):
                        if k<opts['zdim']:
                            means[k,k] = 1
                        else:
                            means[-1] = - 1. / (1. + sqrt(opts['nmixtures']+1)) \
                                            * np.ones((opts['zdim'],),dtype='float32')
                else:
                    assert False, 'Too many mixtures for the latents dim.'
            self.pz_means = opts['pz_scale']*means
            self.pz_covs = opts['sigma_prior']*np.ones((opts['zdim']),dtype='float32')
        else:
            assert False, 'Unknown latent model.'

    def sample_mixtures(self,means,cov,distr,num=100,tpe='numpy'):
        opts = self.opts
        if tpe=='tensor':
            if distr == 'mixture':
                means = tf.expand_dims(means,axis=2)
                cov = tf.expand_dims(cov,axis=2)
                eps = tf.random_normal([num,opts['nmixtures'],opts['nsamples'],opts['zdim']],dtype=tf.float32)
                noises = means + tf.multiply(eps,tf.sqrt(1e-8+cov))
            else:
                assert False, 'Unknown latent model.'
        elif tpe=='numpy':
            if distr == 'mixture':
                means = np.expand_dims(means,axis=1)
                eps = np.random.normal(0.,1.,(num, opts['nmixtures'],opts['nsamples'],opts['zdim'])).astype(np.float32)
                noises = means + np.multiply(eps,np.sqrt(1e-8+cov))
            else:
                assert False, 'Unknown latent model.'
        return noises

    def sample_pz(self, num=100, sampling='one_mixture'):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if distr == 'uniform':
            noise = np.random.uniform(
                self.pz_means[0], self.pz_means[1], [num, opts['zdim']]).astype(np.float32)
        elif distr == 'mixture':
            noises = self.sample_mixtures(self.pz_means,self.pz_covs,distr,num)
            if sampling == 'one_mixture':
                mixture = np.random.randint(opts['nmixtures'],size=num)
                noise = noises[np.arange(num),mixture,0]
            elif sampling == 'per_mixture':
                samples_per_mixture = int(num / opts['nmixtures'])
                class_i = np.repeat(np.arange(opts['nmixtures']),samples_per_mixture,axis=0)
                mixture = np.zeros([num,],dtype='int32')
                mixture[(num % opts['nmixtures']):] = class_i
                noise = noises[np.arange(num),mixture,0]
            elif sampling == 'all_mixtures':
                noise = noises
        else:
            assert False, 'Unknown latent model.'
        return opts['pz_scale'] * noise

    def generate_linespace(self, n, mode, anchors):
        opts = self.opts
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

    def matching_penalty(self,samples_pz, samples_qz):
        opts = self.opts
        if opts['method']=='swae':
            loss_match = self.wae_matching_penalty(samples_pz, samples_qz)
        elif opts['method']=='vae':
            loss_match = self.vae_matching_penalty(samples_qz)
        else:
            assert False, 'Unknown algo %s' % opts['method']

        return loss_match

    def wae_matching_penalty(self,samples_pz, samples_qz):
        opts = self.opts
        if opts['penalty'] == 'mmd':
            loss_match = self.mmd_penalty(samples_pz, samples_qz)
        elif opts['penalty'] == 'kl':
            loss_match = self.kl_penalty(samples_pz)
        elif opts['penalty'] == 'OT':
            loss_match = self.OT_penalty(samples_pz)
        else:
            assert False, 'Unknown penalty %s' % opts['penalty']
        return loss_match

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        sample_shape = sample_pz.get_shape().as_list()[1:]
        # Compute kernel embedings if MMD_gan
        if opts['MMD_gan']:
            ### MMD injective regu
            # Pz samples
            input_pz = tf.reshape(sample_pz,[-1,opts['zdim']])
            f_e_pz = k_encoder(opts, inputs=input_pz,
                                    is_training=self.is_training)
            f_d_pz = k_decoder(opts, noise=f_e_pz, output_dim=opts['zdim'],
                                    is_training=self.is_training)
            recons_pz = tf.reshape(f_d_pz,[-1]+sample_shape)
            l2sq_pz = tf.reduce_sum(tf.square(sample_pz - recons_pz),axis=-1)
            l2sq_pz = tf.reduce_mean(l2sq_pz,axis=-1)
            MMD_regu_pz = tf.reduce_mean(l2sq_pz / opts['nmixtures'] , axis=0)
            MMD_regu_pz = tf.reduce_sum(MMD_regu_pz)
            # Qz samples
            input_qz = tf.reshape(sample_qz,[-1,opts['zdim']])
            f_e_qz = k_encoder(opts, inputs=input_qz,
                            reuse=True,is_training=self.is_training)
            f_d_qz = k_decoder(opts, noise=f_e_qz, output_dim=opts['zdim'],
                            reuse=True,is_training=self.is_training)
            recons_qz = tf.reshape(f_d_qz,[-1]+sample_shape)
            l2sq_qz = tf.reduce_sum(tf.square(sample_qz - recons_qz),axis=-1)
            l2sq_qz = tf.reduce_mean(l2sq_qz,axis=-1)
            weighted_l2sq_qz = tf.multiply(l2sq_qz, self.enc_mixweight)
            MMD_regu_qz = tf.reduce_mean(weighted_l2sq_qz,axis=0)
            MMD_regu_qz = tf.reduce_sum(MMD_regu_qz)

            MMD_regu = MMD_regu_pz + MMD_regu_qz
            ### MMD reducing feasible set
            sample_pz = tf.reshape(f_e_pz,[-1]+sample_shape[:-1]+[opts['k_outdim']])
            E_f_e_pz = tf.reduce_mean(sample_pz, axis=2)
            E_f_e_pz = tf.reduce_mean(E_f_e_pz/opts['nmixtures'], axis=0)
            E_f_e_pz = tf.reduce_sum(E_f_e_pz,axis=0)
            sample_qz = tf.reshape(f_e_qz,[-1]+sample_shape[:-1]+[opts['k_outdim']])
            E_f_e_qz = tf.reduce_mean(sample_qz, axis=2)
            E_f_e_qz = tf.multiply(E_f_e_qz, tf.expand_dims(self.enc_mixweight,axis=-1))
            E_f_e_qz = tf.reduce_mean(E_f_e_qz, axis=0)
            E_f_e_qz = tf.reduce_sum(E_f_e_qz,axis=0)
            one_sided_err = tf.reduce_mean(E_f_e_pz - E_f_e_qz)
            one_sided_err = - tf.nn.relu(-one_sided_err)
        # Compute MMD
        MMD = self.mmd(sample_pz,sample_qz)
        # MMD penalty and mmd_objective for MMD_GAN
        if opts['MMD_gan']:
            MMD_penalty = tf.sqrt(MMD+1e-8) + opts['rg_lambda'] * one_sided_err
            self.mmd_objective = tf.sqrt(MMD+1e-8) \
                                        + opts['rg_lambda'] * one_sided_err \
                                        - opts['ae_lambda'] * MMD_regu
        else:
            if opts['sqrt_MMD']:
                #MMD_penalty = tf.sqrt(MMD+1e-8)
                MMD_penalty = tf.exp(tf.log(MMD+1e-8)/2.)
            else:
                MMD_penalty = MMD
            self.mmd_objective = None

        # for plotting purposes
        self.kl_g = None
        self.kl_d = None

        return MMD_penalty

    def mmd(self, sample_pz, sample_qz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=-1, keepdims=True)
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=-1, keepdims=True)
        distances_pz = self.square_dist(sample_pz, norms_pz, sample_pz, norms_pz)
        distances_qz = self.square_dist(sample_qz, norms_qz, sample_qz, norms_qz)
        distances = self.square_dist(sample_qz, norms_qz, sample_pz, norms_pz)

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
            stat = 0.
            self.res1, self.res2 = 0.0, 0.0
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                # First 2 terms of the MMD
                res1_qz = C / (C + distances_qz)
                res1_qz = tf.reduce_mean(res1_qz,axis=[2,3])
                reshape_enc_mixweight = [-1]+self.enc_mixweight.get_shape().as_list()[1:]+[1,1]
                reshaped_enc_mixweight = tf.reshape(self.enc_mixweight,reshape_enc_mixweight)
                res1_qz = tf.multiply(res1_qz,reshaped_enc_mixweight)
                res1_qz = tf.multiply(res1_qz,tf.transpose(self.enc_mixweight))
                # res1 = tf.multiply(tf.transpose(res1),tf.transpose(self.enc_mixweight))
                # res1 = tf.multiply(tf.transpose(res1),tf.transpose(self.enc_mixweight))
                res1_pz = (C / (C + distances_pz))
                res1_pz = tf.reduce_mean(res1_pz,axis=[2,3]) / (opts['nmixtures']*opts['nmixtures'])
                res1 = res1_qz + res1_pz
                # Correcting for diagonal terms
                res1_diag = tf.trace(tf.reduce_sum(res1,axis=[1,2]))
                res1 = (tf.reduce_sum(res1) - res1_diag) / (nf * nf - nf)
                self.res1 += res1
                # Cross term of the MMD
                res2 = C / (C + distances)
                res2 = tf.reduce_mean(res2,axis=[2,3])
                res2 = tf.multiply(res2,reshaped_enc_mixweight) / opts['nmixtures']
                # res2 =  tf.multiply(tf.transpose(res2),tf.transpose(self.enc_mixweight))
                # res2 = tf.transpose(res2) / opts['nmixtures']
                res2 = tf.reduce_sum(res2) / (nf * nf)
                self.res2 += res2
                stat += res1 - 2. * res2
        else:
            raise ValueError('%s Unknown kernel' % kernel)

        return stat

    def square_dist(self,sample_x, norms_x, sample_y, norms_y):
        dotprod = tf.tensordot(sample_x, tf.transpose(sample_y), [[-1],[0]])
        reshape_norms_x = [-1]+norms_x.get_shape().as_list()[1:]+[1,1]
        distances = tf.reshape(norms_x, reshape_norms_x) + tf.transpose(norms_y) - 2. * dotprod
        # norm_nk = tf.tensordot(norms_x,tf.ones(tf.shape(tf.transpose(norms_x))),[[-1],[0]])
        # norm_lm = tf.tensordot(tf.ones(tf.shape(norms_y)),tf.transpose(norms_y),[[-1],[0]])
        # distances = norm_nk + norm_lm - 2. * dotprod
        return distances

    def kl_penalty(self, sample_pz):
        assert False, 'To implement'
        opts = self.opts
        # Pz term
        logdet_pz = tf.log(tf.reduce_prod(self.pz_covs))# + opts['zdim'] * tf.log(2*pi)
        square_pz = tf.divide(tf.square(sample_pz - self.pz_means),self.pz_covs)
        musigmu_pz = tf.reduce_sum(square_pz,axis=-1)
        log_pz = - (logdet_pz + musigmu_pz) / 2 - tf.log(tf.cast(opts['nmixtures'],dtype=tf.float32))
        log_pz = tf.reduce_mean(log_pz,axis=0)
        kl_pz = tf.reduce_mean(log_pz)
        # Qz term
        logdet_qz = tf.log(tf.reduce_prod(tf.exp(self.enc_logsigmas),axis=-1))# + opts['zdim'] * tf.log(2*pi)
        square_qz = tf.divide(tf.square(sample_pz - self.enc_mean),tf.exp(self.enc_logsigmas))
        musigmu_qz = tf.reduce_sum(square_qz,axis=-1)
        log_qz = - (logdet_qz + musigmu_qz) / 2 + tf.log(self.enc_mixweight)
        log_qz = tf.reduce_mean(log_qz,axis=0)
        kl_qz = tf.reduce_mean(log_qz)

        return tf.reduce_mean(log_pz-log_qz)

    def OT_penalty(self, sample_pz, sample_qz):
        assert False, 'To implement'

    def vae_matching_penalty(self,samples_qz):
        opts = self.opts
        # Continuous KL (actually -KL)
        kl_g = 1 + self.enc_logsigmas \
                    - tf.square(self.enc_mean) \
                    - tf.exp(self.enc_logsigmas)
        kl_g = 0.5 * tf.reduce_sum(kl_g,axis=-1)
        kl_g = tf.multiply(kl_g,self.enc_mixweight)
        kl_g = tf.reduce_sum(kl_g,axis=-1)
        kl_g = tf.reduce_mean(kl_g)
        self.kl_g = -kl_g
        # Discrete KL (actually -KL)
        kl_d = - tf.log(tf.cast(opts['nmixtures'],dtype=tf.float32)) \
                    - tf.log(self.enc_mixweight)
        kl_d = tf.multiply(kl_d,self.enc_mixweight)
        kl_d = tf.reduce_sum(kl_d,axis=-1)
        kl_d = tf.reduce_mean(kl_d)
        self.kl_d = -kl_d

        loss_match = kl_g + kl_d
        return - loss_match

    def reconstruction_loss(self):
        opts = self.opts
        if opts['method']=='swae':
            loss = self.wae_recons_loss()
        elif opts['method']=='vae':
            loss = self.vae_recons_loss()

        return loss

    def wae_recons_loss(self):
        opts = self.opts
        real = tf.expand_dims(tf.expand_dims(self.sample_points,axis=1),axis=1)
        reconstr = self.reconstructed
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[3,4,5])
            loss = tf.sqrt(1e-10 + loss)
            loss = tf.reduce_mean(loss,axis=-1)
            loss = tf.multiply(loss, self.enc_mixweight)
            loss = tf.reduce_mean(loss,axis=0)
            loss = .2 * tf.reduce_sum(loss)
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[3,4,5])
            loss = tf.reduce_mean(loss,axis=-1)
            loss = tf.multiply(loss, self.enc_mixweight)
            loss = tf.reduce_mean(loss,axis=0)
            loss = .05 * tf.reduce_sum(loss)
        elif opts['cost'] == 'l2sq_wrong':
            # c(x,y) = ||x - y||_2^2
            real = self.sample_points
            reconstr = self.one_recons
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = .05 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[3,4,5])
            loss = tf.reduce_mean(loss,axis=-1)
            loss = tf.multiply(loss, self.enc_mixweight)
            loss = tf.reduce_mean(loss,axis=0)
            loss = .2 * tf.reduce_sum(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def vae_recons_loss(self):
        opts = self.opts
        real = tf.expand_dims(tf.expand_dims(self.sample_points,axis=1),axis=1)
        logit = self.reconstructed
        eps = 1e-10
        l = real*tf.log(eps+logit) + (1-real)*tf.log(eps+1-logit)
        loss = tf.reduce_sum(l,axis=[3,4,5])
        loss = tf.reduce_mean(loss,axis=-1)
        loss = tf.reduce_mean(tf.multiply(loss,self.enc_mixweight))
        return -loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        opts = self.opts
        # SWAE optimizer
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars
        if opts['clip_grad']:
            # Clipping gradient
            grad, var = zip(*opt.compute_gradients(loss=self.objective,
                                                    var_list=ae_vars))
            clip_grad, _ = tf.clip_by_global_norm(grad, opts['clip_norm'])
            self.swae_opt = opt.apply_gradients(zip(clip_grad, var))
        else:
            # No clipping
            self.swae_opt = opt.minimize(loss=self.objective,
                                        var_list=ae_vars)
        # MMD optimizer
        if opts['penalty']=='mmd' and opts['MMD_gan']:
            k_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='kernel_encoder')
            k_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='kernel_generator')
            gan_vars = k_encoder_vars + k_decoder_vars
            mmd_lr = opts['mmd_lr']
            mmd_opt = self.optimizer(mmd_lr, self.lr_decay)
            grads_and_vars = mmd_opt.compute_gradients(loss=-self.mmd_objective,
                                    var_list=gan_vars)
            clip_grads_and_vars = [(tf.clip_by_value(gv[0],-0.01,0.01), gv[1]) for gv in grads_and_vars]
            self.MMD_opt = mmd_opt.apply_gradients(clip_grads_and_vars)
        # Pretraining optimizer
        if opts['e_pretrain']:
            pre_opt = self.optimizer(lr)
            self.pretrain_opt = pre_opt.minimize(loss=self.loss_pretrain,
                                    var_list=encoder_vars)
        else:
            self.pretrain_opt = None

    def pretrain_loss(self):
        opts = self.opts
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        # qz_means = tf.reduce_mean(self.encoded_means, axis=0)
        # qz_covs = tf.reduce_mean(tf.exp(self.enc_logsigmas), axis=0)
        qz_means = tf.reduce_mean(self.mixtures_encoded, axis=[0,2], keepdims=True)
        pz_means = tf.reduce_mean(self.sample_mix_noise, axis=[0,2], keepdims=True)
        # Mean loss
        mean_loss = tf.reduce_mean(tf.square(qz_means - pz_means))
        # Covariances
        qz_covs = tf.reduce_sum(tf.square(self.mixtures_encoded-qz_means),axis=[0,2])
        pz_covs = tf.reduce_sum(tf.square(self.sample_mix_noise-pz_means),axis=[0,2])
        cov_loss = tf.reduce_mean(tf.square(qz_covs - pz_covs))

        return mean_loss + cov_loss

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 1500
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_mix_noise = self.sample_pz(batch_size,sampling='all_mixtures')
            [_, loss_pretrain] = self.sess.run(
                [self.pretrain_opt,
                 self.loss_pretrain],
                feed_dict={self.sample_points: batch_images,
                            self.sample_mix_noise: batch_mix_noise,
                            self.is_training: True})

    def train(self, data):
        opts = self.opts
        if opts['method']=='swae':
            logging.error('Training WAE, Matching penalty: %s' % (opts['penalty']))
        elif opts['method']=='vae':
            logging.error('Training VAE')

        utils.create_dir(opts['method'])
        work_dir = os.path.join(opts['method'],opts['work_dir'])

        losses, losses_rec, losses_match, kl_gau, kl_dis  = [], [], [], [], []
        mmd_losses= []
        batches_num = int(data.num_points / opts['batch_size'])
        train_size = data.num_points
        pdb.set_trace()
        self.num_pics = opts['plot_num_pics']
        self.fixed_noise = self.sample_pz(opts['plot_num_pics'],sampling = 'per_mixture')
        self.sess.run(self.init)

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')


        self.start_time = time.time()
        counter = 0
        decay = 1.
        if opts['method']=='swae':
            wae_lambda = opts['lambda']
        else:
            wae_lambda = 1
            opts['lambda_schedule'] = 'constant'
        wait = 0
        wait_lambda = 0

        for epoch in range(opts['epoch_num']):

            # Update learning rate if necessary
            if opts['lr_schedule'] == 'manual':
                if epoch == 30:
                    decay = decay / 2.
                if epoch == 50:
                    decay = decay / 5.
                if epoch == 100:
                    decay = decay / 10.
            elif opts['lr_schedule'] != 'plateau':
                assert type(opts['lr_schedule']) == float
                decay = 1.0 * 10**(-epoch / float(opts['lr_schedule']))

            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(work_dir,
                                                            'checkpoints',
                                                            'trained-wae'),
                                global_step=counter)

            # Iterate over batches
            for it in range(batches_num):
                if opts['penalty'] == 'mmd' and opts['MMD_gan'] and it % opts['mmd_every'] == 0:
                    # Maximize MMD
                    for Dit in range(opts['mmd_iter']):
                        # Sample batches of data points and Pz noise
                        data_ids = np.random.choice(train_size,
                                            opts['batch_size'],
                                            replace=False)
                        batch_images = data.data[data_ids].astype(np.float32)
                        batch_mix_noise = self.sample_pz(opts['batch_size'],sampling='all_mixtures')
                        # Update encoder and decoder
                        [_, loss] = self.sess.run([self.MMD_opt,self.mmd_objective],
                                feed_dict={self.sample_points: batch_images,
                                           self.sample_mix_noise: batch_mix_noise,
                                           self.lr_decay: decay,
                                           self.lmbd: wae_lambda,
                                           self.is_training: True})
                        mmd_losses.append(loss)

                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size,
                                    opts['batch_size'],
                                    replace=False)
                batch_images = data.data[data_ids].astype(np.float32)
                batch_mix_noise = self.sample_pz(opts['batch_size'],sampling='all_mixtures')
                # Update encoder and decoder
                [_, loss, loss_rec, loss_match, kl_g, kl_d] = self.sess.run(
                        [self.swae_opt,
                         self.objective,
                         self.loss_reconstruct,
                         self.penalty,
                         self.kl_g,
                         self.kl_d],
                        feed_dict={self.sample_points: batch_images,
                                   self.sample_mix_noise: batch_mix_noise,
                                   self.lr_decay: decay,
                                   self.lmbd: wae_lambda,
                                   self.is_training: True})


                # Update learning rate if necessary
                if opts['lr_schedule'] == 'plateau':
                    # First 30 epochs do nothing
                    if epoch >= 30:
                        # If no significant progress was made in last 10 epochs
                        # then decrease the learning rate.
                        if loss < min(losses[-20 * batches_num:]):
                            wait = 0
                        else:
                            wait += 1
                        if wait > 10 * batches_num:
                            decay = max(decay  / 1.4, 1e-6)
                            logging.error('Reduction in lr: %f' % decay)
                            wait = 0

                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                if kl_g is not None:
                    kl_gau.append(kl_g)
                if kl_d is not None:
                    kl_dis.append(kl_d)
                if opts['verbose']:
                    logging.error('Matching penalty after %d steps: %f' % (
                        counter, losses_match[-1]))

                # Update regularizer if necessary
                if opts['lambda_schedule'] == 'adaptive':
                    if wait_lambda >= 1999 and len(losses_rec) > 0:
                        last_rec = losses_rec[-1]
                        last_match = losses_match[-1]
                        wae_lambda = 0.5 * wae_lambda + \
                                     0.5 * last_rec / abs(last_match)
                        logging.error('Lambda updated to %f' % wae_lambda)
                        wait_lambda = 0
                    else:
                        wait_lambda += 1

                counter += 1

                # Print debug info
                if opts['method']=='vae':
                    cond1 = counter < 61 and counter % 2==0
                else:
                    cond1 = counter==1
                cond2 = counter % opts['print_every'] == 0
                if cond1 or cond2:
                    now = time.time()
                    # Auto-encoding test images
                    [loss_rec_test, enc_mean_all, encoded, enc_mean, rec_test, prob] = self.sess.run(
                                [self.loss_reconstruct,
                                 self.enc_mean,
                                 self.encoded,
                                 self.encoded_means,
                                 self.one_recons,
                                 self.enc_mixweight],
                                feed_dict={self.sample_points: data.test_data[:self.num_pics],
                                                                    self.is_training: False})

                    # Auto-encoding training images
                    [rec_train, mix_train] = self.sess.run(
                                [self.one_recons,
                                 self.enc_mixweight],
                                feed_dict={self.sample_points: data.data[:self.num_pics],
                                                                self.is_training: False})

                    # Random samples generated by the model
                    sample_gen = self.sess.run(
                                self.decoded,
                                feed_dict={self.sample_noise: self.fixed_noise,
                                           self.is_training: False})

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                                epoch + 1, opts['epoch_num'],
                                it + 1, batches_num)
                    logging.error(debug_str)
                    debug_str = 'LOSS=%.3f, MATCH=%.3f, ' \
                                'RECONS=%.3f, RECONS_TEST=%.3f' % (
                                losses[-1], losses_match[-1],
                                losses_rec[-1], loss_rec_test)
                    logging.error(debug_str)

                    # Making plots
                    save_plots(opts, data.data[:self.num_pics], data.test_data[:self.num_pics],
                                    data.test_labels[:self.num_pics],
                                    rec_train[:self.num_pics], rec_test[:self.num_pics],
                                    prob,
                                    enc_mean_all, encoded, enc_mean,
                                    self.fixed_noise,
                                    sample_gen,
                                    losses, losses_rec, losses_match,
                                    kl_gau, kl_dis,
                                    work_dir,
                                    'res_e%04d_mb%05d.png' % (epoch, it))

        # Save the final model
        if epoch > 0:
            self.saver.save(self.sess,
                             os.path.join(work_dir,
                                          'checkpoints',
                                          'trained-wae-final'),
                             global_step=counter)

    def test(self, data, MODEL_DIR, WEIGHTS_FILE):
        opts = self.opts
        if opts['method']=='swae':
            logging.error('SWAE with %s matching penalty' % (opts['penalty']))
        elif opts['method']=='vae':
            logging.error('VAE')

        # Load trained weights
        MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # Split test data
        num_test = 400
        heldout_data = data.test_data[:num_test]
        heldout_labels = data.test_labels[:num_test]
        test_data = data.test_data[num_test:]
        test_labels = data.test_labels[num_test:]
        debug_str = 'Full test data size: %d ' % (np.shape(data.test_data)[0])
        logging.error(debug_str)
        debug_str = 'Held out data size: %d' % (np.shape(heldout_data)[0])
        logging.error(debug_str)
        debug_str = 'Effective test data size: %d' % (np.shape(test_data)[0])
        logging.error(debug_str)

        # Getting probs on held out set
        logging.error('Determining clusters ID..')
        probs = self.sess.run(
                    self.enc_mixweight,
                    feed_dict={self.sample_points: heldout_data,
                                    self.is_training: False})
        # Determine clusters given probs
        labelled_clusters = get_labels(heldout_labels,probs)
        print(labelled_clusters)
        # Getting predictions on effective test set
        logging.error('Computing accuracy..')
        probs = self.sess.run(
                    self.enc_mixweight,
                    feed_dict={self.sample_points: test_data,
                                    self.is_training: False})
        # compute accuracy
        acc = accuracy(test_labels,probs,labelled_clusters)
        # Getting predictions on test set
        probs_full = self.sess.run(
                    self.enc_mixweight,
                    feed_dict={self.sample_points: data.test_data,
                                    self.is_training: False})
        # compute accuracy
        acc_full = accuracy(data.test_labels,probs_full,labelled_clusters)
        # Printing various loss values
        debug_str = 'acc=%.3f, full acc=%.3f' % (
                    acc, acc_full)
        logging.error(debug_str)
        # saving
        name = 'acc'
        np.save(os.path.join(MODEL_PATH,name),np.array(acc,acc_full))

    def vizu(self, data, MODEL_DIR, WEIGHTS_FILE):
        opts = self.opts
        if opts['method']=='swae':
            logging.error('SWAE with %s matching penalty' % (opts['penalty']))
        elif opts['method']=='vae':
            logging.error('VAE')

        num_pics = 400
        step_inter = 20
        num_anchors = 10
        imshape = datashapes[opts['dataset']]

        # Load trained weights
        MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)

        # Auto-encoding training images
        logging.error('Encoding and decoding train images..')
        rec_train = self.sess.run(
                    self.one_recons,
                    feed_dict={self.sample_points: data.data[:num_pics],
                                                    self.is_training: False})

        # Auto-encoding test images
        logging.error('Encoding and decoding test images..')
        [encoded, enc_mean, rec_test, prob] = self.sess.run(
                    [self.encoded,
                     self.encoded_means,
                     self.one_recons,
                     self.enc_mixweight],
                    feed_dict={self.sample_points: data.test_data[:num_pics],
                                                        self.is_training: False})

        # Encode anchors points and interpolate
        logging.error('Encoding anchors points and interpolating..')
        anchors_ids = np.random.choice(1000,2*num_anchors,replace=False)
        anchors = data.test_data[anchors_ids]
        enc_anchors = self.sess.run(
                    self.encoded,
                    feed_dict={self.sample_points: anchors,
                                                        self.is_training: False})
        encod_interpolation = self.generate_linespace(step_inter,'points_interpolation',anchors=enc_anchors)
        noise = encod_interpolation.reshape(-1,opts['zdim'])
        decoded = self.sess.run(
                    self.decoded,
                    feed_dict={self.sample_noise: noise,
                               self.is_training: False})
        decod_inteprolation = decoded.reshape([-1,step_inter]+imshape)
        start_anchors = anchors[::2]
        end_anchors = anchors[1::2]
        decod_inteprolation = np.concatenate((start_anchors[:,np.newaxis],np.concatenate((decod_inteprolation,end_anchors[:,np.newaxis]), axis=1)),axis=1)

        # Random samples generated by the model
        prior_noise = self.sample_pz(num_pics,sampling = 'per_mixture')
        logging.error('Decoding random samples..')
        sample_gen = self.sess.run(
                    self.decoded,
                    feed_dict={self.sample_noise: prior_noise,
                               self.is_training: False})

        # Encode prior means and interpolate
        logging.error('Generating latent linespace and decoding..')
        if opts['zdim']==2:
            prior_interpolation = self.generate_linespace(step_inter,'transformation',anchors=self.pz_means)
        else:
            prior_interpolation = self.generate_linespace(step_inter,'priors_interpolation',anchors=self.pz_means)
        noise = prior_interpolation.reshape(-1,opts['zdim'])
        decoded = self.sess.run(
                    self.decoded,
                    feed_dict={self.sample_noise: noise,
                               self.is_training: False})
        prior_decod_interpolation = decoded.reshape([-1,step_inter]+imshape)

        # Making plots
        logging.error('Saving images..')
        save_plots_vizu(opts, data.data[:num_pics],
                        rec_train,
                        data.test_data[:num_pics],data.test_labels[:num_pics],
                        encoded, enc_mean, rec_test, prob,
                        data.test_data[:num_anchors],
                        decod_inteprolation,
                        prior_noise, sample_gen,
                        prior_decod_interpolation,
                        MODEL_PATH)

def save_plots(opts, sample_train,sample_test,
                    label_test,
                    rec_train, rec_test,
                    prob,
                    enc_mean_all, encoded, enc_mean,
                    sample_prior,
                    sample_gen,
                    losses, losses_rec, losses_match,
                    kl_gau, kl_dis,
                    work_dir,
                    filename):
    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img6 | img5

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   Means mixture weights
        img5    -   real pics
        img6    -   loss curves

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = sample_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        sample_train = sample_train / 2. + 0.5
        sample_test = sample_test / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        rec_test = rec_test / 2. + 0.5
        sample_gen = sample_gen / 2. + 0.5

    images = []

    ### Reconstruction plots
    for pair in [(sample_train, rec_train),
                 (sample_test, rec_test)]:

        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        assert len(sample) == num_pics
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2

        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])

        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    ### Sample plots
    for sample in [sample_gen, sample_train]:
        assert len(sample) == num_pics
        pics = []
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - sample[idx, :, :, :])
            else:
                pics.append(sample[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    img1, img2, img3, img5 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 4 * height_pic / float(dpi)
    fig_width = 6 * width_pic / float(dpi)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 3)

    # Filling in separate parts of the plot

    # First samples and reconstructions
    for img, (gi, gj, title) in zip([img1, img2, img3],
                             [(0, 0, 'Train reconstruction'),
                              (0, 1, 'Test reconstruction'),
                              (0, 2, 'Generated samples')]):
        plt.subplot(gs[gi, gj])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)

        ax = plt.subplot(gs[gi, gj])
        plt.text(0.47, 1., title,
                 ha="center", va="bottom", size=20, transform=ax.transAxes)

        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    ### Then the mean mixtures plots
    mean_probs = []
    for i in range(10):
        probs = [prob[k] for k in range(num_pics) if label_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    entropies = calculate_row_entropy(mean_probs)
    relab_mask = relabelling_mask_from_entropy(mean_probs, entropies)
    mean_probs = mean_probs[relab_mask]
    ax = plt.subplot(gs[1, 0])
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.text(0.47, 1., 'Test means probs',
           ha="center", va="bottom", size=20, transform=ax.transAxes)
    plt.yticks(np.arange(10),relab_mask)
    plt.xticks(np.arange(10))

    ###UMAP visualization of the embedings
    ax = plt.subplot(gs[1, 1])
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,sample_prior),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded,sample_prior),axis=0))
                                #metric='correlation').fit_transform(np.concatenate((encoded,enc_mean,sample_prior),axis=0))

    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
                c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='tab10'))
                #c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    plt.colorbar()
    plt.scatter(embedding[num_pics:, 0], embedding[num_pics:, 1],
                            color='navy', s=10, marker='*',label='Pz')
    # plt.scatter(embedding[num_pics:(2*num_pics-1), 0], embedding[num_pics:(2*num_pics-1), 1],
    #             color='deepskyblue', s=10, marker='x',label='mean Qz test')
    # plt.scatter(embedding[2*num_pics:, 0], embedding[2*num_pics:, 1],
    #                         color='navy', s=10, marker='*',label='Pz')

    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.1
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify

    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper left')
    plt.text(0.47, 1., 'UMAP latents', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The loss curves
    ax = plt.subplot(gs[1, 2])
    total_num = len(losses_rec)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(losses_rec) + 1, x_step)

    y = np.log(losses_rec[::x_step])
    plt.plot(x, y, linewidth=2, color='red', label='log(rec loss)')

    y = np.log(losses[::x_step])
    plt.plot(x, y, linewidth=3, color='black', label='log(wae loss)')

    if len(kl_gau)>0:
        y = np.log(np.abs(losses_match[::x_step]))
        plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

        y = np.log(kl_gau[::x_step])
        plt.plot(x, y, linewidth=2, color='blue', linestyle=':', label='log(cont KL)')
    else:
        y = np.log(opts['lambda']*np.abs(losses_match[::x_step]))
        plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

    if len(kl_dis)>0:
        y = np.log(kl_dis[::x_step])
        plt.plot(x, y, linewidth=2, color='blue', linestyle='--', label='log(disc KL)')

    plt.grid(axis='y')
    plt.legend(loc='lower left')
    plt.text(0.47, 1., 'Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### Saving plots and data
    # Plot
    utils.create_dir(work_dir)
    fig.savefig(utils.o_gfile((work_dir, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()

    # data
    save_dir = 'data_for_plots'
    save_path = os.path.join(work_dir,save_dir)
    utils.create_dir(save_path)
    # Losses
    loss_path = os.path.join(save_path,'loss')
    utils.create_dir(loss_path)
    name = filename[:-4]
    if len(kl_gau)>0:
        np.savez(os.path.join(loss_path,name),
                    loss=np.array(losses[::x_step]),
                    loss_rec=np.array(losses_rec[::x_step]),
                    loss_match=np.array(losses_match[::x_step]),
                    kl_cont=np.array(kl_gau[::x_step]),
                    kl_disc=np.array(kl_dis[::x_step]))
    else:
        np.savez(os.path.join(loss_path,name),
                    loss=np.array(losses[::x_step]),
                    loss_rec=np.array(losses_rec[::x_step]),
                    loss_match=np.array(opts['lambda']*losses_match[::x_step]))
    # Probs
    probs_path = os.path.join(save_path,'probs')
    utils.create_dir(probs_path)
    np.save(os.path.join(probs_path,name),prob)

    # Means
    means_path = os.path.join(save_path,'means')
    utils.create_dir(means_path)
    np.save(os.path.join(means_path,name),enc_mean_all)

    # reconstruct
    recon_path = os.path.join(save_path,'recon')
    utils.create_dir(recon_path)
    np.savez(os.path.join(recon_path,name),
                test_data=sample_test,
                rec_test=rec_test)

def save_plots_vizu(opts, data_train,
                rec_train,
                data_test, label_test,
                encoded, enc_mean, rec_test, prob,
                anchors,
                decod_inteprolation,
                sample_prior,
                sample_gen,
                prior_decod_interpolation,
                work_dir):
    """ Generates and saves the following plots:
        img1    -   train reconstruction
        img2    -   test reconstruction
        img3    -   samples
        img4    -   test interpolation
        img5    -   prior interpolation
        img6    -   discrete latents
        img7    -   UMAP
    """

    if not tf.gfile.IsDirectory(work_dir):
        raise Exception("working directory doesnt exist")
    save_dir = os.path.join(work_dir,'figures')
    utils.create_dir(save_dir)

    greyscale = np.shape(prior_decod_interpolation)[-1] == 1

    if opts['input_normalize_sym']:
        data_train = data_train / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        anchors = anchors / 2. + 0.5
        decod_inteprolation = decod_inteprolation / 2. + 0.5
        sample_gen = sample_gen / 2. + 0.5
        prior_decod_interpolation = prior_decod_interpolation / 2. + 0.5

    images = []


    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test)]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        num_pics = np.shape(sample)[0]
        num_cols = 20
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)


    ### Points Interpolation plots
    white_pix = 4
    num_pics = np.shape(decod_inteprolation)[0]
    num_cols = np.shape(decod_inteprolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - decod_inteprolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            white = np.zeros((white_pix,)+np.shape(pic)[2:])
            pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = decod_inteprolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            white = np.zeros((white_pix,)+np.shape(pic)[1:])
            pic = np.concatenate(white,pic)
            pics.append(pic)
    image = np.concatenate(pics, axis=0)
    images.append(image)

    ###Prior Interpolation plots
    white_pix = 4
    num_pics = np.shape(prior_decod_interpolation)[0]
    num_cols = np.shape(prior_decod_interpolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - prior_decod_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            if opts['zdim']!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[2:])
                pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = prior_decod_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            if opts['zdim']!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[1:])
                pic = np.concatenate(white,pic)
            pics.append(pic)
    # Figuring out a layout
    image = np.concatenate(pics, axis=0)
    images.append(image)

    img1, img2, img3, img4 = images

    ###Settings for pyplot fig
    dpi = 100
    for img, title, filename in zip([img1, img2, img3, img4],
                         ['Train reconstruction',
                         'Test reconstruction',
                         'Points interpolation',
                         'Priors interpolation'],
                         ['train_recon',
                         'test_recon',
                         'point_inter',
                         'prior_inter']):
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        # fig_height = height_pic / float(dpi)
        # fig_width = width_pic / float(dpi)
        fig_height = height_pic / 10
        fig_width = width_pic / 10
        fig = plt.figure(figsize=(fig_width, fig_height))
        #fig = plt.figure()
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        #plt.title(title)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        # fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
        #             format='png')
        plt.close()


    # Set size for following plots
    height_pic= img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)


    ###The mean mixtures plots
    mean_probs = []
    num_pics = np.shape(prob)[0]
    for i in range(10):
        probs = [prob[k] for k in range(num_pics) if label_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    entropies = calculate_row_entropy(mean_probs)
    cluster_to_digit = relabelling_mask_from_entropy(mean_probs, entropies)
    digit_to_cluster = np.argsort(cluster_to_digit)
    mean_probs = mean_probs[::-1,digit_to_cluster]
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.title('Average probs')
    plt.yticks(np.arange(10),np.arange(10)[::-1])
    plt.xticks(np.arange(10))
    # Saving
    filename = 'probs.png'
    fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()


    ###Sample plots
    pics = []
    num_cols = 10
    num_pics = np.shape(sample_gen)[0]
    size_pics = np.shape(sample_gen)[1]
    num_to_keep = 10
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - sample_gen[idx, :, :, :])
        else:
            pics.append(sample_gen[idx, :, :, :])
    # Figuring out a layout
    pics = np.array(pics)
    cluster_pics = np.array(np.split(pics, num_cols))[digit_to_cluster]
    img = np.concatenate(cluster_pics.tolist(), axis=2)
    img = np.concatenate(img, axis=0)
    img = img[:num_to_keep*size_pics]
    fig = plt.figure(figsize=(img.shape[1]/10, img.shape[0]/10))
    #fig = plt.figure()
    if greyscale:
        image = img[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(image, cmap='Greys',
                        interpolation='none', vmin=0., vmax=1.)
    else:
        plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
    #plt.title('Generated samples')
    # Removing axes, ticks, labels
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    # Saving
    filename = 'gen_sample.png'
    plt.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
    # fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
    #             format='png')
    plt.close()


    ###UMAP visualization of the embedings
    num_pics = 200
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,sample_prior),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],sample_prior),axis=0))
                                #metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],enc_mean[:num_pics],sample_prior),axis=0))
    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
               c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    plt.colorbar()
    plt.scatter(embedding[num_pics:, 0], embedding[num_pics:, 1],
                            color='navy', s=3, alpha=0.5, marker='*',label='Pz')
    # plt.scatter(embedding[num_pics:(2*num_pics-1), 0], embedding[num_pics:(2*num_pics-1), 1],
    #            color='aqua', s=3, alpha=0.5, marker='x',label='mean Qz test')
    # plt.scatter(embedding[2*num_pics:, 0], embedding[2*num_pics:, 1],
    #                         color='navy', s=3, alpha=0.5, marker='*',label='Pz')
    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.1
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify
    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off',
                    right='off',
                    left='off',
                    labelleft='off')
    plt.legend(loc='upper left')
    plt.title('UMAP latents')
    # Saving
    filename = 'umap.png'
    fig.savefig(utils.o_gfile((save_dir, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()


    ###Saving plots and data
    data_dir = 'data_for_plots'
    save_path = os.path.join(work_dir,data_dir)
    utils.create_dir(save_path)
    filename = 'final_plots.npy'
    np.savez(os.path.join(save_path,filename),
                smples=sample_gen,
                smples_pr=sample_prior,
                rec_tr=rec_train,
                rec_te=rec_test,
                enc=encoded,
                enc_mean=enc_mean,
                points=decod_inteprolation,
                priors=prior_decod_interpolation,
                prob=prob)

def accuracy(labels, probs, clusters_id):
    preds = np.argmax(probs,axis=-1)
    relabelled_preds = np.choose(preds,clusters_id)
    correct_prediction = (relabelled_preds==labels)
    return np.mean(correct_prediction)

def get_labels(labels,probs):
    mean_probs = []
    num_pics = np.shape(probs)[0]
    for i in range(10):
        prob = [probs[k] for k in range(num_pics) if labels[k]==i]
        prob = np.mean(np.stack(prob,axis=0),axis=0)
        mean_probs.append(prob)
    mean_probs = np.stack(mean_probs,axis=0)
    cluster_to_digit = relabelling_mask_from_probs(mean_probs)
    return cluster_to_digit

def relabelling_mask_from_probs(mean_probs):
    probs_copy = mean_probs
    nmixtures = np.shape(mean_probs)[-1]
    k_vals = []
    min_prob = np.zeros(nmixtures)
    mask = np.arange(10)
    while np.amax(probs_copy) > 0.:
        max_probs = np.amax(probs_copy,axis=-1)
        digit_idx = np.argmax(max_probs)
        k_val_sort = np.argsort(probs_copy[digit_idx])
        i = -1
        k_val = k_val_sort[i]
        while k_val in k_vals:
            i -= 1
            k_val = k_val_sort[i]
        k_vals.append(k_val)
        mask[k_val] = digit_idx
        probs_copy[digit_idx] = min_prob
    return mask

def relabelling_mask_from_entropy(mean_probs, entropies):
    k_vals = []
    max_entropy_state = np.ones(len(entropies))/len(entropies)
    max_entropy = scistats.entropy(max_entropy_state)
    mask = np.arange(10)
    while np.amin(entropies) < max_entropy:
        digit_idx = np.argmin(entropies)
        k_val_sort = np.argsort(mean_probs[digit_idx])
        i = -1
        k_val = k_val_sort[i]
        while k_val in k_vals:
            i -= 1
            k_val = k_val_sort[i]
        k_vals.append(k_val)
        mask[k_val] = digit_idx
        entropies[digit_idx] = max_entropy
    return mask

def calculate_row_entropy(mean_probs):
    entropies = []
    for i in range(np.shape(mean_probs)[0]):
        entropies.append(scistats.entropy(mean_probs[i]))
    entropies = np.asarray(entropies)
    return entropies

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)

def set_2d_priors(nmixtures):
    assert nmixtures==10, 'Too many mixtures to initialize prior'
    means = np.zeros([10, 2]).astype(np.float32)
    angles = []
    for i in range(3):
        angle = np.array([sin(i*pi/3.), cos(i*pi/3.)])
        angles.append(angle)
    for k in range(1,4):
        means[k] = k / 3. * angles[0]
    for k in range(1,4):
        means[k+3] = k / 3. * angles[1]
    for k in range(1,3):
        means[k+2*3] = k / 3. * angles[2] + np.array([.0, 1.])
    means[9] = [sqrt(3)/6., .5]

    means -= means[9]

    return means
