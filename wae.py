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
        if opts['e_noise'] in ('deterministic', 'implicit', 'add_noise'):
            self.enc_mean, self.enc_logsigmas, self.enc_mixweight = None, None, None
            res = encoder(opts, inputs=self.sample_points,
                            is_training=self.is_training)
            if opts['e_noise'] == 'implicit':
                self.encoded, self.encoder_A = res
            else:
                self.encoded = res
        elif opts['e_noise'] in ('gaussian', 'mixture'):
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
                enc_logmixweight = tf.clip_by_value(enc_logmixweight, -50, 50)
                self.enc_mixweight = tf.nn.softmax(enc_logmixweight,axis=-1)
                self.enc_mean = enc_mean
                enc_logsigmas = tf.clip_by_value(enc_logsigmas, -10, 10)
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
        # self.one_recons, self.one_recons_logits = decoder(opts, reuse=True, noise=self.encoded,
        #                                                         is_training=False)
        # Decode the content of sample_noise
        self.decoded, self.decoded_logits = decoder(opts, reuse=True, noise=self.sample_noise,
                                                                is_training=self.is_training)
        # --- Objectives, losses, penalties, pretraining
        # Compute reconstruction cost
        self.loss_reconstruct = self.reconstruction_loss()
        # Compute matching penalty cost
        self.penalty = self.matching_penalty(self.sample_mix_noise[:,:,-1],self.mixtures_encoded[:,:,-1])
        # Compute wae obj
        self.wae_objective = self.loss_reconstruct \
                            + self.lmbd * self.penalty
        # Compute entropy of mixture weights if needed
        if opts['entropy']:
            mean_mixweight = tf.reduce_sum(self.enc_mixweight,axis=0)
            h = tf.multiply(mean_mixweight,tf.log(mean_mixweight))
            self.H = - tf.reduce_sum(h)
            self.wae_objective = self.wae_objective - self.H_lambda * self.H
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
        wae_lambda = tf.placeholder(tf.float32, name='lambda_ph')

        self.lr_decay = decay
        self.is_training = is_training
        self.lmbd = wae_lambda
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
        self.saver = saver

    def init_prior(self):
        opts = self.opts
        distr = opts['pz']
        if distr == 'uniform':
            self.pz_means = [-1.0,1.0].astype(np.float32)
            self.pz_covs = None
        elif distr == 'mixture':
            if opts['zdim']==2:
                means = np.zeros([opts['nmixtures'], opts['zdim']]).astype(np.float32)
                for k in range(0,opts['nmixtures']):
                    means[k] = sqrt(2.0)*np.array([cos(k * 2*pi/opts['nmixtures']),sin(k * 2*pi/opts['nmixtures'])]).astype(np.float32)
            else:
                assert 2*opts['zdim']>=opts['nmixtures'], 'Too many mixtures in the latents.'
                means = np.zeros([opts['nmixtures'], opts['zdim']]).astype(np.float32)
                for k in range(opts['nmixtures']):
                    if k % 2 == 0:
                        means[k,int(k/2)] = sqrt(2.0)*max(opts['sigma_prior'],1.)
                    else:
                        means[k,int(k/2)] = -sqrt(2.0)*max(opts['sigma_prior'],1.)
            self.pz_means = opts['pz_scale']*means
            self.pz_covs = opts['sigma_prior']*np.ones((opts['zdim'])).astype(np.float32)
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

    def matching_penalty(self,samples_pz, samples_qz):
        opts = self.opts
        if opts['penalty'] == 'mmd':
            loss_match = self.mmd_penalty(samples_pz, samples_qz)
        elif opts['penalty'] == 'kl':
            loss_match = self.kl_penalty(samples_pz)
        else:
            assert False, 'Unknown penalty %s' % opts['penalty']
        return loss_match

    def mmd_penalty(self, sample_pz, sample_qz):
        opts = self.opts
        # Compute kernel embedings if MMD_gan
        if opts['MMD_gan']:
            assert False, 'To implement'
            ### MMD injective regu
            # Pz samples
            input_pz = tf.reshape(sample_pz,[-1,opts['zdim']])
            f_e_pz = k_encoder(opts, inputs=input_pz,
                                    is_training=self.is_training)
            f_d_pz = k_decoder(opts, noise=f_e_pz, output_dim=opts['zdim'],
                                    is_training=self.is_training)
            recons_pz = tf.reshape(f_d_pz,[-1,opts['nmixtures'],opts['zdim']])
            l2sq_pz = tf.reduce_sum(tf.square(sample_pz - recons_pz),axis=-1)
            MMD_regu_pz = tf.reduce_mean(l2sq_pz, axis=0)
            MMD_regu_pz = tf.reduce_sum(MMD_regu_pz) / opts['nmixtures']
            # Qz samples
            input_qz = tf.reshape(sample_qz,[-1,opts['zdim']])
            f_e_qz = k_encoder(opts, inputs=input_qz,
                            reuse=True,is_training=self.is_training)
            f_d_qz = k_decoder(opts, noise=f_e_qz, output_dim=opts['zdim'],
                            reuse=True,is_training=self.is_training)
            recons_qz = tf.reshape(f_d_qz,[-1,opts['nmixtures'],opts['zdim']])
            l2sq_qz = tf.reduce_sum(tf.square(sample_qz - recons_qz),axis=-1)
            weighted_l2sq_qz = tf.multiply(l2sq_pz, self.enc_mixweight)
            MMD_regu_qz = tf.reduce_mean(weighted_l2sq_qz,axis=0)
            MMD_regu_qz = tf.reduce_sum(MMD_regu_qz)

            MMD_regu = MMD_regu_pz + MMD_regu_qz
            ### MMD reducing feasible set
            sample_pz = tf.reshape(f_e_pz,[-1,opts['nmixtures'],opts['k_outdim']])
            sample_qz = tf.reshape(f_e_qz,[-1,opts['nmixtures'],opts['k_outdim']])
            E_f_e_pz = tf.reduce_mean(sample_pz, axis=0)
            E_f_e_pz = tf.reduce_sum(E_f_e_pz,axis=0) / opts['nmixtures']
            E_f_e_qz = tf.multiply(sample_qz, tf.expand_dims(self.enc_mixweight,axis=-1))
            E_f_e_qz = tf.reduce_mean(E_f_e_qz, axis=0)
            E_f_e_qz = tf.reduce_sum(E_f_e_qz,axis=0)
            one_sided_err = tf.reduce_mean(E_f_e_pz - E_f_e_qz)
            one_sided_err = - tf.nn.relu(-one_sided_err)
        # Compute MMD
        MMD = self.mmd(sample_pz,sample_qz)
        # MMD penalty and mmd_objective for MMD_GAN
        if opts['MMD_gan']:
            MMD_penalty = tf.sqrt(MMD) + opts['rg_lambda'] * one_sided_err
            self.mmd_objective = tf.sqrt(MMD) \
                                        + opts['rg_lambda'] * one_sided_err \
                                        - opts['ae_lambda'] * MMD_regu
        else:
            MMD_penalty = tf.sqrt(MMD)
            self.mmd_objective = None

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
            for scale in [.1, .2, .5, 1., 2., 5.]:
                C = Cbase * scale
                # First 2 terms of the MMD
                res1 = C / (C + distances_qz)
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(self.enc_mixweight))
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(self.enc_mixweight))
                res1 += (C / (C + distances_pz)) / (opts['nmixtures']*opts['nmixtures'])
                # Correcting for diagonal terms
                res1_diag = tf.trace(tf.reduce_sum(res1,axis=[1,2]))
                res1 = (tf.reduce_sum(res1) - res1_diag) / (nf * nf - nf)
                self.res1 += res1
                # Cross term of the MMD
                res2 = C / (C + distances)
                res2 =  tf.multiply(tf.transpose(res2),tf.transpose(self.enc_mixweight))
                res2 = tf.transpose(res2) / opts['nmixtures']
                res2 = tf.reduce_sum(res2) / (nf * nf)
                self.res2 += res2
                stat += res1 - 2. * res2
        else:
            raise ValueError('%s Unknown kernel' % kernel)

        return stat

    def square_dist(self,sample_x, norms_x, sample_y, norms_y):
        # assert sample_x.get_shape().as_list() == sample_y.get_shape().as_list(), \
        #     'Prior samples need to have same shape as posterior samples'
        # assert len(sample_x.get_shape().as_list()) == 4, \
        #     'Prior samples need to have shape [batch,nmixtures,nsamples,zdim] for mixture model'
        dotprod = tf.tensordot(sample_x, tf.transpose(sample_y), [[-1],[0]])
        norm_nk = tf.tensordot(norms_x,tf.ones(tf.shape(tf.transpose(norms_x))),[[-1],[0]])
        norm_lm = tf.tensordot(tf.ones(tf.shape(norms_y)),tf.transpose(norms_y),[[-1],[0]])
        distances = norm_nk + norm_lm - 2. * dotprod

        return distances

    def kl_penalty(self, sample_pz):
        opts = self.opts
        # Pz term
        logdet = tf.log(tf.reduce_prod(self.pz_covs))# + opts['zdim'] * tf.log(2*pi)
        square = tf.divide(tf.square(sample_pz - self.pz_means),self.pz_covs)
        musigmu = tf.reduce_sum(square,axis=-1)
        log_pz = - (logdet + musigmu) / 2 - tf.log(tf.cast(opts['nmixtures'],dtype=tf.float32))
        kl_pz = tf.reduce_mean(log_pz)
        # Qz term
        logdet = tf.log(tf.reduce_prod(tf.exp(self.enc_logsigmas),axis=-1))# + opts['zdim'] * tf.log(2*pi)
        square = tf.divide(tf.square(sample_pz - self.enc_mean),tf.exp(self.enc_logsigmas))
        musigmu = tf.reduce_sum(square,axis=-1)
        log_qz = - (logdet + musigmu) / 2 + tf.log(self.enc_mixweight)
        kl_qz = tf.reduce_mean(log_qz)

        return kl_pz - kl_qz

    def pretrain_loss(self):
        opts = self.opts
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        # Means
        mean_pz = self.pz_means
        mean_qz = tf.reduce_mean(self.mixtures_encoded, axis=[0,2])
        mean_loss = tf.reduce_sum(tf.square(mean_pz - mean_qz))
        # Covariances
        centered_pz = self.sample_mix_noise - tf.expand_dims(mean_pz,axis=-2)
        square = tf.reduce_sum(tf.square(centered_pz),axis=-1)
        cov_pz = tf.reduce_mean(square,axis=[0,2])
        #cov_pz /= opts['e_pretrain_sample_size'] - 1.
        centered_qz = self.mixtures_encoded - tf.expand_dims(mean_qz,axis=-2)
        square = tf.reduce_sum(tf.square(centered_qz),axis=-1)
        cov_qz = tf.reduce_mean(square,axis=[0,2])
        #cov_qz /= opts['e_pretrain_sample_size'] - 1.
        cov_loss = tf.reduce_sum(tf.square(cov_pz - cov_qz))

        return mean_loss + cov_loss

    def reconstruction_loss(self):
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
        grads_and_vars = opt.compute_gradients(loss=self.wae_objective,
                                    var_list=ae_vars)
        self.grad = [gv[0] for gv in grads_and_vars]
        self.swae_opt = opt.minimize(loss=self.wae_objective,
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

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 500
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
        logging.error('Training SWAE, Matching penalty: %s' % opts['penalty'])
        losses, losses_rec, losses_match, losses_means  = [], [], [], []
        mmd_losses= []
        batches_num = int(data.num_points / opts['batch_size'])
        train_size = data.num_points
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
        wae_lambda = opts['lambda']
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
                self.saver.save(self.sess, os.path.join(opts['work_dir'],
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
                [_, loss, loss_rec, loss_match] = self.sess.run(
                        [self.swae_opt,
                         self.wae_objective,
                         self.loss_reconstruct,
                         self.penalty],
                        feed_dict={self.sample_points: batch_images,
                                   self.sample_mix_noise: batch_mix_noise,
                                   self.lr_decay: decay,
                                   self.lmbd: wae_lambda,
                                   self.is_training: True})

                # [_, grad, loss, loss_rec, loss_match, means, sigmas, mix] = self.sess.run(
                #         [self.swae_opt,
                #          self.grad,
                #          self.wae_objective,
                #          self.loss_reconstruct,
                #          self.penalty,
                #          self.enc_mean,
                #          self.enc_logsigmas,
                #          self.debug_mix],
                #         feed_dict={self.sample_points: batch_images,
                #                    self.sample_mix_noise: batch_mix_noise,
                #                    self.lr_decay: decay,
                #                    self.lmbd: wae_lambda,
                #                    self.is_training: True})
                # mix_print = np.amax(mix,axis=0)
                # print(mix_print)
                # print("")
                # mea_print = np.amax(means,axis=(0,1))
                # print(mea_print)
                # print("")
                # sig_print = np.amax(sigmas,axis=(0,1))
                # print(sig_print)
                # print("")
                # logging.error('res1: %f' % res1)
                # logging.error('res2: %f' % res2)
                # print("")
                # max_l = [np.amax(t) for t in grad]
                # print(max_l)


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
                if opts['verbose']:
                    logging.error('Matching penalty after %d steps: %f' % (
                        counter, losses_match[-1]))

                # Update regularizer if necessary
                if opts['lambda_schedule'] == 'adaptive':
                    if wait_lambda >= 999 and len(losses_rec) > 0:
                        last_rec = losses_rec[-1]
                        last_match = losses_match[-1]
                        wae_lambda = 0.5 * wae_lambda + \
                                     0.5 * last_rec / abs(last_match)
                        if opts['verbose']:
                            logging.error('Lambda updated to %f' % wae_lambda)
                        wait_lambda = 0
                    else:
                        wait_lambda += 1

                counter += 1

                # Print debug info
                if counter % opts['print_every'] == 0 or counter==1:
                    now = time.time()

                    # Auto-encoding test images
                    [loss_rec_test, enc_test, enc_means_test, rec_test, mix_test] = self.sess.run(
                                [self.loss_reconstruct,
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
                    debug_str = 'LOSS=%.5f, MATCH=%.5f, ' \
                                'RECONS=%.5f, RECONS_TEST=%.5f' % (
                                losses[-1], losses_match[-1],
                                losses_rec[-1], loss_rec_test)
                    logging.error(debug_str)

                    # Making plots
                    save_plots(opts, data.data[:self.num_pics], data.test_data[:self.num_pics],
                                    data.test_labels,
                                    rec_train[:self.num_pics], rec_test[:self.num_pics],
                                    mix_test,
                                    enc_test, enc_means_test,
                                    self.fixed_noise,
                                    sample_gen,
                                    losses, losses_rec, losses_match,
                                    'res_e%04d_mb%05d.png' % (epoch, it))

        # # Save the final model
        # if epoch > 0:
        #     self.saver.save(self.sess,
        #                      os.path.join(opts['work_dir'],
        #                                   'checkpoints',
        #                                   'trained-wae-final'),
        #                      global_step=counter)

def save_plots(opts, sample_train,sample_test,
                    label_test,
                    recon_train, recon_test,
                    mix_test,
                    enc_test, enc_means_test,
                    sample_prior,
                    sample_gen,
                    losses, losses_rec, losses_match,
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
        recon_train = recon_train / 2. + 0.5
        recon_test = recon_test / 2. + 0.5
        sample_gen = sample_gen / 2. + 0.5

    images = []

    ### Reconstruction plots
    for pair in [(sample_train, recon_train),
                 (sample_test, recon_test)]:

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
    test_probs_labels = []
    #n = np.shape(mix_test)[0]
    for i in range(10):
        te_prob = [mix_test[k] for k in range(num_pics) if label_test[k]==i]
        te_prob = np.mean(np.stack(te_prob,axis=0),axis=0)
        test_probs_labels.append(te_prob)
    test_probs_labels = np.stack(test_probs_labels,axis=0)
    ax = plt.subplot(gs[1, 0])
    plt.imshow(test_probs_labels,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    #plt.colorbar()
    plt.text(0.47, 1., 'Test means probs',
           ha="center", va="bottom", size=20, transform=ax.transAxes)

    ###UMAP visualization of the embedings
    ax = plt.subplot(gs[1, 1])
    if opts['zdim']==2:
        embedding = np.concatenate((enc_test,enc_means_test,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((enc_test,enc_means_test,sample_prior),axis=0))

    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
                c=label_test[:num_pics], s=40, label='Qz test',cmap='Accent')
    plt.colorbar()
    plt.scatter(embedding[num_pics:(2*num_pics-1), 0], embedding[num_pics:(2*num_pics-1), 1],
                color='deepskyblue', s=10, marker='x',label='mean Qz test')
    plt.scatter(embedding[2*num_pics:, 0], embedding[2*num_pics:, 1],
                            color='navy', s=10, marker='*',label='Pz')

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


    ###The loss curves
    ax = plt.subplot(gs[1, 2])
    total_num = len(losses_rec)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(losses_rec) + 1, x_step)

    y = np.log(losses_rec[::x_step])
    plt.plot(x, y, linewidth=2, color='red', label='log(rec loss)')

    y = np.log(np.abs(losses_match[::x_step]))
    plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

    y = np.log(losses[::x_step])
    plt.plot(x, y, linewidth=2, color='black', label='log(wae loss)')

    plt.grid(axis='y')
    plt.legend(loc='lower left')
    plt.text(0.47, 1., 'Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    # Saving
    utils.create_dir(opts['work_dir'])
    fig.savefig(utils.o_gfile((opts['work_dir'], filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()
