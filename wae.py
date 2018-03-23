# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

""" Wasserstein Auto-Encoder models

"""

import sys
import time
import os
from math import sqrt
import numpy as np
import tensorflow as tf
import logging
import ops
import utils
from models import encoder, decoder
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

        # -- Some of the parameters for future use

        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # -- Placeholders

        self.add_model_placeholders()
        self.add_training_placeholders()
        sample_size = tf.shape(self.sample_points,out_type=tf.int64)[0]
        self.init_prior()

        # -- Transformation ops

        # Encode the content of sample_points placeholder
        if opts['e_noise'] in ('deterministic', 'implicit', 'add_noise'):
            self.enc_mean, self.enc_sigmas, self.enc_mixprob = None, None, None
            res = encoder(opts, inputs=self.sample_points,
                            is_training=self.is_training)
            if opts['e_noise'] == 'implicit':
                self.encoded, self.encoder_A = res
            else:
                self.encoded = res
        elif opts['e_noise'] in ('gaussian', 'mixture'):
            # Encoder outputs means and variances of Gaussians, and mixing probs
            if opts['stop_grad']:
                _, _, enc_mixprob = encoder(opts, inputs=self.sample_points,
                                                                is_training=self.is_training)
                self.enc_mixprob = enc_mixprob
                #eps = tf.random_normal([sample_size,opts['nmixtures'],opts['zdim']],mean=0.0,stddev=0.0099999,dtype=tf.float32)
                eps = tf.zeros([tf.cast(sample_size,dtype=tf.int32),opts['nmixtures'],opts['zdim']],dtype=tf.float32)
                self.enc_mean = self.pz_means + eps
                self.enc_sigmas = opts['init_std']*tf.ones([tf.cast(sample_size,dtype=tf.int32),opts['nmixtures'],opts['zdim']],dtype=tf.float32)
            else:
                enc_mean, enc_sigmas, enc_mixprob = encoder(opts, inputs=self.sample_points,
                                                                is_training=self.is_training)
                enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
                self.enc_mixprob = enc_mixprob
                self.enc_mean = enc_mean
                self.enc_sigmas = enc_sigmas

            if opts['verbose']:
                self.add_sigmas_debug()

            # Encoding
            # sampling from all mixtures
            self.mixtures_encoded = self.sample_mixtures(self.enc_mean,
                                                tf.exp(self.enc_sigmas),
                                                opts['e_noise'],sample_size,'tensor')
            # Select corresponding mixtures
            if opts['e_noise'] == 'mixture':
                mixture_idx = tf.reshape(tf.multinomial(self.enc_mixprob, 1),[-1])
                self.mixture = tf.stack([tf.range(sample_size),mixture_idx],axis=-1)
                self.encoded = tf.gather_nd(self.mixtures_encoded,self.mixture)
            else:
                self.encoded = self.mixtures_encoded
        # Decode the points encoded above (i.e. reconstruct)
        self.reconstructed, self.reconstructed_logits = \
                        decoder(opts, noise=self.encoded,
                                is_training=self.is_training)

        # Decode the content of sample_noise
        self.decoded, self.decoded_logits = decoder(opts, reuse=True, noise=self.sample_noise,
                                                                is_training=self.is_training)
        # -- Objectives, losses, penalties, vizu
        self.penalty = self.matching_penalty()
        self.loss_reconstruct = self.reconstruction_loss()
        self.wae_objective = self.loss_reconstruct + self.wae_lambda * self.penalty
        self.blurriness = self.compute_blurriness()

        #self.add_least_gaussian2d_ops()

        # -- Optimizers, savers, etc
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
            tf.float32, [None] + [opts['nmixtures'],opts['zdim']], name='mix_noise_ph')

        self.sample_points = data
        self.sample_noise = noise
        self.sample_mix_noise = mix_noise

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        wae_lambda = tf.placeholder(tf.float32, name='lambda_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.lr_decay = decay
        self.wae_lambda = wae_lambda
        self.is_training = is_training

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        if self.enc_mean is not None:
            tf.add_to_collection('encoder_mean', self.enc_mean)
            tf.add_to_collection('encoder_var', self.enc_sigmas)
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
        elif distr in ('normal', 'sphere'):
            self.pz_means = np.zeros(opts['zdim']).astype(np.float32)
            self.pz_covs = opts['sigma_prior']*np.identity(opts['zdim']).astype(np.float32)
        elif distr == 'mixture':
            assert opts['zdim']>=opts['nmixtures'], 'Too many mixtures in the latents.'
            means = np.zeros([opts['nmixtures'], opts['zdim']]).astype(np.float32)
            for k in range(opts['nmixtures']):
                means[k,k] = sqrt(2.0)*max(opts['sigma_prior'],sqrt(2.0))
            self.pz_means = means
            self.pz_covs = opts['sigma_prior']*np.ones((opts['zdim'])).astype(np.float32)
        else:
            assert False, 'Unknown latent model.'

    def sample_mixtures(self,means,cov,distr,num=100,tpe='numpy'):
        if tpe=='tensor':
            if distr in ('normal', 'sphere'):
                eps = tf.random_normal([num, self.opts['zdim']])
                noises = means + tf.multiply(eps,tf.sqrt(1e-8+cov))
            elif distr == 'mixture':
                eps = tf.random_normal([num, self.opts['nmixtures'],self.opts['zdim']],dtype=tf.float32)
                noises = means + tf.multiply(eps,tf.sqrt(1e-8+cov))
            else:
                assert False, 'Unknown latent model.'
        elif tpe=='numpy':
            if distr in ('normal', 'sphere'):
                eps = np.random.normal(0.,1.,(num, self.opts['zdim']))
                noises = means + np.multiply(eps,np.sqrt(1e-8+cov))
            elif distr == 'mixture':
                eps = np.random.normal(0.,1.,(num, self.opts['nmixtures'],self.opts['zdim'])).astype(np.float32)
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
        elif distr in ('normal', 'sphere'):
            noise = self.sample_mixtures(self.pz_means,self.pz_covs,distr,num)
            if distr == 'sphere':
                noise = noise / np.sqrt(np.sum(noise * noise, axis=1))[:, np.newaxis]
        elif distr == 'mixture':
            noises = self.sample_mixtures(self.pz_means,self.pz_covs,distr,num)
            if sampling == 'one_mixture':
                mixture = np.random.randint(opts['nmixtures'],size=num)
                noise = noises[np.arange(num),mixture]
                #noise = tf.gather_nd(noises,tf.stack([tf.range(num,dtype=tf.int32),mixture],axis=-1))
            elif sampling == 'per_mixture':
                samples_per_mixture = int(num / opts['nmixtures'])
                class_i = np.repeat(np.arange(opts['nmixtures']),samples_per_mixture,axis=0)
                mixture = np.zeros([num,],dtype='int32')
                mixture[(num % opts['nmixtures']):] = class_i
                noise = noises[np.arange(num),mixture]
                #noise = tf.gather_nd(noises,tf.stack([tf.range(num,dtype=tf.int32),mixture],axis=-1))
            elif sampling == 'all_mixtures':
                noise = noises
        else:
            assert False, 'Unknown latent model.'
        return opts['pz_scale'] * noise

    def matching_penalty(self):
        opts = self.opts
        sample_qz = self.mixtures_encoded
        assert sample_qz.get_shape().as_list()[1:]==[opts['nmixtures'],opts['zdim']], \
                                                            'Wrong shape for encodings'
        if opts['pz'] == 'mixture':
            sample_pz = self.sample_mix_noise
            assert sample_pz.get_shape().as_list()[1:]==[opts['nmixtures'],opts['zdim']], \
                                                    'Wrong shape for samples from prior'
        else:
            sample_pz = self.sample_noise

        if opts['z_test'] == 'mmd':
            loss_match = self.mmd_penalty(sample_qz, sample_pz)
        else:
            assert False, 'Unknown penalty %s' % opts['z_test']
        return loss_match

    def mmd_penalty(self, sample_qz, sample_pz):
        opts = self.opts
        sigma2_p = opts['pz_scale'] ** 2
        kernel = opts['mmd_kernel']
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2,tf.int32)

        """
        TODO : add case where qz is different from pz: pz=mixture, qz=gaussian
        """
        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=-1, keepdims=True)
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        distances_pz = self.square_dist(sample_pz, norms_pz, sample_pz, norms_pz, opts['pz'])
        distances_qz = self.square_dist(sample_qz, norms_qz, sample_qz, norms_qz, opts['e_noise'])
        distances = self.square_dist(sample_qz, norms_qz, sample_pz, norms_pz, opts['e_noise'])
        probs = tf.exp(self.enc_mixprob)/tf.reduce_sum(tf.exp(self.enc_mixprob),axis=-1,keepdims=True)

        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')

            # First 2 terms of the MMD
            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            if opts['pz'] == 'mixture':
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(probs))
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(probs))
                res1 += tf.exp( - distances_pz / 2. / sigma2_k) / (opts['nmixtures']*opts['nmixtures'])
                # Correcting for diagonal terms
                # Correcting for diagonal terms
                # res1_ddiag = tf.diag_part(tf.transpose(res1,perm=(0,1,3,2)))
                # res1_diag = tf.diag_part(tf.reduce_sum(res1,axis=[0,3]))
                # res1 = tf.reduce_sum(res1) / (nf * nf - 1) \
                #         + tf.reduce_sum(res1_diag) / (nf * (nf * nf - nf)) \
                #         - tf.reduce_sum(res1_ddiag) / (nf * nf - nf)
                res1_diag = tf.diag_part(tf.reduce_sum(res1,axis=[1,2]))
                res1 = (tf.reduce_sum(res1)\
                        - tf.reduce_sum(res1_diag)) / (nf * nf - nf)
            else:
                res1 += tf.exp( - distances_pz / 2. / sigma2_k)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            # Cross term of the MMD
            res2 = tf.exp( - distances / 2. / sigma2_k)
            if opts['pz'] == 'mixture':
                res2 =  tf.multiply(tf.transpose(res2),tf.transpose(probs))
                res2 = tf.transpose(res2) / opts['nmixtures']
            else:
                res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            Cbase = 2 * opts['zdim'] * sigma2_p
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                # First 2 terms of the MMD
                res1 = C / (C + distances_qz)
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(probs))
                res1 = tf.multiply(tf.transpose(res1),tf.transpose(probs))
                res1 += (C / (C + distances_pz)) / (opts['nmixtures']*opts['nmixtures'])
                # Correcting for diagonal terms
                # res1_ddiag = tf.diag_part(tf.transpose(res1,perm=(0,1,3,2)))
                # res1_diag = tf.diag_part(tf.reduce_sum(res1,axis=[0,3]))
                # res1 = tf.reduce_sum(res1) / (nf * nf - 1) \
                #         + tf.reduce_sum(res1_diag) / (nf * (nf * nf - nf)) \
                #         - tf.reduce_sum(res1_ddiag) / (nf * nf - nf)
                res1_diag = tf.diag_part(tf.reduce_sum(res1,axis=[1,2]))
                res1 = (tf.reduce_sum(res1)\
                        - tf.reduce_sum(res1_diag)) / (nf * nf - nf)
                # Cross term of the MMD
                res2 = C / (C + distances)
                res2 =  tf.multiply(tf.transpose(res2),tf.transpose(probs))
                res2 = tf.transpose(res2) / opts['nmixtures']
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        else:
            raise ValueError('%s Unknown kernel' % kernel)

        return stat

    def square_dist(self,sample_x, norms_x, sample_y, norms_y, distr):
        assert sample_x.get_shape().as_list() == sample_x.get_shape().as_list(), \
            'Prior samples need to have same shape as posterior samples'
        if distr == 'mixture':
            assert len(sample_x.get_shape().as_list()) == 3, \
                'Prior samples need to have shape [batch,nmixtures,zdim] for mixture model'
            dotprod = tf.tensordot(sample_x, tf.transpose(sample_y), [[-1],[0]])
            norm_nk = tf.tensordot(norms_x,tf.ones(tf.shape(tf.transpose(norms_x))),[[-1],[0]])
            norm_lm = tf.tensordot(tf.ones(tf.shape(norms_y)),tf.transpose(norms_y),[[-1],[0]])
            distances = norm_nk + norm_lm - 2. * dotprod
        else:
            assert len(sample_x.get_shape().as_list()) == 2, \
                'Prior samples need to have shape [batch,zdim] for gaussian model'
            dotprod = tf.matmul(sample_x, sample_y, transpose_b=True)
            distances = norms_x + tf.transpose(norms_y) - 2. * dotprod

        return distances

    def reconstruction_loss(self):
        opts = self.opts
        real = self.sample_points
        reconstr = self.reconstructed
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.05 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def compute_blurriness(self):
        images = self.sample_points
        sample_size = tf.shape(self.sample_points)[0]
        # First convert to greyscale
        if self.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter,
                            strides=[1, 1, 1, 1], padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar

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
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        ae_vars = encoder_vars + decoder_vars

        if opts['verbose']:
            logging.error('Param num in AE: %d' % \
                    np.sum([np.prod([int(d) for d in v.get_shape()]) \
                    for v in ae_vars]))

        # Auto-encoder optimizer
        opt = self.optimizer(lr, self.lr_decay)
        self.ae_opt = opt.minimize(loss=self.wae_objective,
                              var_list=encoder_vars + decoder_vars)

    def add_least_gaussian2d_ops(self):
        """ Add ops searching for the 2d plane in z_dim hidden space
            corresponding to the 'least Gaussian' look of the sample
        """

        opts = self.opts

        with tf.variable_scope('leastGaussian2d'):
            # Projection matrix which we are going to tune
            sample = tf.placeholder(
                tf.float32, [None, opts['zdim']], name='sample_ph')
            v = tf.get_variable(
                'proj_v', [opts['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            u = tf.get_variable(
                'proj_u', [opts['zdim'], 1],
                tf.float32, tf.random_normal_initializer(stddev=1.))
            npoints = tf.cast(tf.shape(sample)[0], tf.int32)

            # First we need to make sure projection matrix is orthogonal

            v_norm = tf.nn.l2_normalize(v, 0)
            dotprod = tf.reduce_sum(tf.multiply(u, v_norm))
            u_ort = u - dotprod * v_norm
            u_norm = tf.nn.l2_normalize(u_ort, 0)
            Mproj = tf.concat([v_norm, u_norm], 1)
            sample_proj = tf.matmul(sample, Mproj)
            a = tf.eye(npoints)
            a -= tf.ones([npoints, npoints]) / tf.cast(npoints, tf.float32)
            b = tf.matmul(sample_proj, tf.matmul(a, a), transpose_a=True)
            b = tf.matmul(b, sample_proj)
            # Sample covariance matrix
            covhat = b / (tf.cast(npoints, tf.float32) - 1)
            gcov = opts['pz_scale'] ** 2.  * tf.eye(2)
            # l2 distance between sample cov and the Gaussian cov
            projloss =  tf.reduce_sum(tf.square(covhat - gcov))
            # Also account for the first moment, i.e. expected value
            projloss += tf.reduce_sum(tf.square(tf.reduce_mean(sample_proj, 0)))
            # We are maximizing
            projloss = -projloss
            optim = tf.train.AdamOptimizer(0.001, 0.9)
            optim = optim.minimize(projloss, var_list=[v, u])

        self.proj_u = u_norm
        self.proj_v = v_norm
        self.proj_sample = sample
        self.proj_covhat = covhat
        self.proj_loss = projloss
        self.proj_opt = optim

    def least_gaussian_2d(self, X):
        """
        Given a sample X of shape (n_points, n_z) find 2d plain
        such that projection looks least Gaussian
        """
        opts = self.opts
        with self.sess.as_default(), self.sess.graph.as_default():
            sample = self.proj_sample
            optim = self.proj_opt
            loss = self.proj_loss
            u = self.proj_u
            v = self.proj_v

            covhat = self.proj_covhat
            proj_mat = tf.concat([v, u], 1).eval()
            dot_prod = -1
            best_of_runs = 10e5 # Any positive value would do
            updated = False
            for _ in range(3):
                # We will run 3 times from random inits
                loss_prev = 10e5 # Any positive value would do
                proj_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope='leastGaussian2d')
                self.sess.run(tf.variables_initializer(proj_vars))
                step = 0
                for _ in range(5000):
                    self.sess.run(optim, feed_dict={sample:X})
                    step += 1
                    if step % 10 == 0:
                        loss_cur = loss.eval(feed_dict={sample: X})
                        rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                        if rel_imp < 1e-2:
                            break
                        loss_prev = loss_cur
                loss_final = loss.eval(feed_dict={sample: X})
                if loss_final < best_of_runs:
                    updated = True
                    best_of_runs = loss_final
                    proj_mat = tf.concat([v, u], 1).eval()
                    dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()
        if not updated:
            logging.error('WARNING: possible bug in the worst 2d projection')
        return proj_mat, dot_prod

    def train(self, data):
        opts = self.opts
        if opts['verbose']:
            logging.error(opts)
        logging.error('Training SWAE')
        losses = []
        losses_rec = []
        losses_match = []
        blurr_vals = []
        encoding_changes = []
        batches_num = int(data.num_points / opts['batch_size'])
        train_size = data.num_points
        self.num_pics = opts['plot_num_pics']
        self.fixed_noise = self.sample_pz(opts['plot_num_pics'],sampling = 'per_mixture')

        self.sess.run(self.init)

        self.start_time = time.time()
        counter = 0
        decay = 1.
        wae_lambda = opts['lambda']
        wait = 0
        wait_lambda = 0

        real_blurr = self.sess.run(self.blurriness,
                        feed_dict={self.sample_points: data.data[:self.num_pics]})
        logging.error('Real pictures sharpness = %.5f' % np.min(real_blurr))

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
                self.saver.save(self.sess,
                                 os.path.join(opts['work_dir'],
                                              'checkpoints',
                                              'trained-wae'),
                                 global_step=counter)

            # Iterate over batches
            for it in range(batches_num):

                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size,
                                    opts['batch_size'],
                                    replace=False)
                batch_images = data.data[data_ids].astype(np.float32)
                batch_noise = self.sample_pz(opts['batch_size'],sampling='one_mixture')
                batch_mix_noise = self.sample_pz(opts['batch_size'],sampling='all_mixtures')
                # Update encoder and decoder
                [_, loss, loss_rec, loss_match] = self.sess.run(
                        [self.ae_opt,
                         self.wae_objective,
                         self.loss_reconstruct,
                         self.penalty],
                        feed_dict={self.sample_points: batch_images,
                                   self.sample_noise: batch_noise,
                                   self.sample_mix_noise: batch_mix_noise,
                                   self.lr_decay: decay,
                                   self.wae_lambda: wae_lambda,
                                   self.is_training: True})
                # print("means:")
                # print(enc_mean[:3])
                # print("sigmas:")
                # print(enc_sigmas[:3])
                # print("mix:")
                # print(enc_mixprob[:3])
                # pdb.set_trace()
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
                    [loss_rec_test, enc_test, rec_test, mix_test] = self.sess.run(
                            [self.loss_reconstruct,
                             self.encoded,
                             self.reconstructed,
                             self.enc_mixprob],
                            feed_dict={self.sample_points: data.test_data[:self.num_pics],
                                                                self.is_training: False})

                    # Auto-encoding training images
                    [loss_rec_train, rec_train, mix_train, mean_train] = self.sess.run(
                            [self.loss_reconstruct,
                             self.reconstructed,
                             self.enc_mixprob,
                             self.enc_mean],
                            feed_dict={self.sample_points: data.data[:self.num_pics],
                                                            self.is_training: False})

                    # Random samples generated by the model
                    sample_gen = self.sess.run(
                            self.decoded,
                            feed_dict={self.sample_noise: self.fixed_noise,
                                       self.is_training: False})

                    # # Blurriness measures
                    # gen_blurr = self.sess.run(
                    #         self.blurriness,
                    #         feed_dict={self.sample_points: sample_gen})
                    # blurr_vals.append(np.min(gen_blurr))

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                                epoch + 1, opts['epoch_num'],
                                it + 1, batches_num)
                    debug_str += ' (WAE_LOSS=%.5f, RECON_LOSS_TEST=%.5f, ' \
                                'MATCH_LOSS=%.5f' % (
                                losses[-1], loss_rec_test,
                                losses_match[-1])
                    logging.error(debug_str)

                    # probs = np.exp(mix_test[:10])/np.sum(np.exp(mix_test[:10]),axis=-1,keepdims=True)
                    # debug = np.concatenate((probs,data.labels[:10][:,np.newaxis]),axis=-1)
                    # np.set_printoptions(precision=2,linewidth=200)
                    # print(debug)

                    # Making plots
                    save_plots(opts, data.data[:self.num_pics],
                                    data.labels[:self.num_pics],
                                    data.test_data[:self.num_pics],
                                    data.test_labels[:self.num_pics],
                                    rec_train[:self.num_pics],
                                    rec_test[:self.num_pics],
                                    mix_train[:self.num_pics],
                                    mix_test[:self.num_pics],
                                    enc_test,
                                    self.fixed_noise,
                                    sample_gen,
                                    losses_rec, losses_match, blurr_vals,
                                    'res_e%04d_mb%05d.png' % (epoch, it))

        # # Save the final model
        # if epoch > 0:
        #     self.saver.save(self.sess,
        #                      os.path.join(opts['work_dir'],
        #                                   'checkpoints',
        #                                   'trained-wae-final'),
        #                      global_step=counter)

    def add_sigmas_debug(self):

        # Ops to debug variances of random encoders
        enc_sigmas = self.enc_sigmas
        enc_sigmas = tf.Print(
            enc_sigmas,
            [tf.nn.top_k(tf.reshape(enc_sigmas, [-1]), 1).values[0]],
            'Maximal log sigmas:')
        enc_sigmas = tf.Print(
            enc_sigmas,
            [-tf.nn.top_k(tf.reshape(-enc_sigmas, [-1]), 1).values[0]],
            'Minimal log sigmas:')
        self.enc_sigmas = enc_sigmas

        enc_sigmas_t = tf.transpose(self.enc_sigmas)
        max_per_dim = tf.reshape(tf.nn.top_k(enc_sigmas_t, 1).values, [-1, 1])
        min_per_dim = tf.reshape(-tf.nn.top_k(-enc_sigmas_t, 1).values, [-1, 1])
        per_dim = tf.concat([min_per_dim, max_per_dim], axis=1)
        self.debug_sigmas = per_dim

def save_plots(opts, sample_train, label_train,
                    sample_test, label_test,
                    recon_train, recon_test,
                    mix_train, mix_test,
                    enc_test,
                    sample_prior,
                    sample_gen,
                    losses_rec, losses_match, blurr_vals,
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
    #train_probs = np.exp(mix_train)/np.sum(np.exp(mix_train),axis=-1,keepdims=True)
    test_probs = np.exp(mix_test)/np.sum(np.exp(mix_test),axis=-1,keepdims=True)
    #train_probs_labels = []
    test_probs_labels = []
    for i in range(10):
        # tr_prob = [train_probs[k] for k in range(num_pics) if label_train[k]==i]
        # tr_prob = np.mean(np.stack(tr_prob,axis=0),axis=0)
        te_prob = [test_probs[k] for k in range(num_pics) if label_test[k]==i]
        te_prob = np.mean(np.stack(te_prob,axis=0),axis=0)
        # train_probs_labels.append(tr_prob)
        test_probs_labels.append(te_prob)
    #train_probs_labels = np.stack(train_probs_labels,axis=0).transpose()
    test_probs_labels = np.stack(test_probs_labels,axis=0)

    # ax = plt.subplot(gs[1, 0])
    # plt.imshow(train_probs_labels,cmap='hot', interpolation='none', vmin=0., vmax=1.)
    # plt.colorbar()
    # plt.text(0.47, 1., 'Train means probs',
    #         ha="center", va="bottom", size=20, transform=ax.transAxes)

    ax = plt.subplot(gs[1, 0])
    plt.imshow(test_probs_labels,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.colorbar()
    plt.text(0.47, 1., 'Test means probs',
           ha="center", va="bottom", size=20, transform=ax.transAxes)

    ###UMAP visualization of the embedings
    ax = plt.subplot(gs[1, 1])
    embedding = umap.UMAP(n_neighbors=5,
                            min_dist=0.3,
                            metric='correlation').fit_transform(np.concatenate((enc_test,sample_prior),axis=0))
    plt.scatter(embedding[:np.shape(enc_test)[0], 0], embedding[:np.shape(enc_test)[0], 1],
                c=label_test, s=30, label='Qz test',cmap='Accent')
    plt.colorbar()
    plt.scatter(embedding[np.shape(enc_test)[0]:, 0], embedding[np.shape(enc_test)[0]:, 1],
                            color='navy', s=10, marker='*',label='Pz')

    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.3
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

    y = np.log(np.abs(losses_rec[::x_step]))
    plt.plot(x, y, linewidth=2, color='red', label='log(|rec loss|)')

    y = np.log(np.abs(losses_match[::x_step]))
    plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.text(0.47, 1., 'Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)



    # Saving
    utils.create_dir(opts['work_dir'])
    fig.savefig(utils.o_gfile((opts['work_dir'], filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()
