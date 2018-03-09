# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

""" Wasserstein Auto-Encoder models

"""

import sys
import time
import os
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
        elif opts['e_noise'] in ('gaussian', 'mixture_gaussians'):
            # Encoder outputs means and variances of Gaussians, and mixing probs
            enc_mean, enc_sigmas, enc_mixprob = encoder(opts, inputs=self.sample_points,
                                                            is_training=self.is_training)
            enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
            self.enc_mean, self.enc_sigmas, self.enc_mixprob = enc_mean, enc_sigmas, enc_mixprob
            if opts['verbose']:
                self.add_sigmas_debug()

            # Encoding
            # sampling mixture indices
            if opts['e_noise'] == 'mixture_gaussians':
                eps = tf.random_normal([sample_size, opts['nmixtures'],opts['zdim']],
                                                            0., 1., dtype=tf.float32)
                mixture_idx = tf.reshape(tf.multinomial(self.enc_mixprob, 1),[-1])
                self.mixture = tf.stack([tf.range(sample_size),mixture_idx],axis=-1)
            else:
                eps = tf.random_normal([sample_size, opts['zdim']],
                                                            0., 1., dtype=tf.float32)
                self.mixture = None
            # sampling from all mixtures
            self.mixtures_encoded = self.enc_mean + tf.multiply(
                        eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
            # Select corresponding mixtures
            if opts['e_noise'] == 'mixture_gaussians':
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
        self.add_least_gaussian2d_ops()

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

    def matching_penalty(self):
        opts = self.opts
        sample_qz = self.mixtures_encoded
        if opts['pz'] == 'mixture':
            sample_pz = self.sample_mix_noise
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

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=-1, keepdims=True)
        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        if opts['pz'] == 'mixture':
            assert len(sample_pz.get_shape().as_list()) == 3, \
                'Prior samples need to have shape [batch,nmixtures,zdim] if prior is mixture'
            dotprods_pz = tf.tensordot(sample_pz, tf.transpose(sample_pz), [[-1],[0]])
            norm_nk = tf.tensordot(norms_pz,tf.ones(tf.shape(tf.transpose(norms_pz))),[[-1],[0]])
            norm_lm = tf.tensordot(tf.ones(tf.shape(norms_pz)),tf.transpose(norms_pz),[[-1],[0]])
            distances_pz = norm_nk + norm_lm - 2. * dotprods_pz
        else:
            assert len(sample_pz.get_shape().as_list()) == 2, \
                'Prior samples need to have shape [batch,zdim] if prior is gaussian'
            dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
            distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz
        if opts['e_noise'] == 'mixture_gaussians':
            assert len(sample_qz.get_shape().as_list()) == 3, \
                'latent samples need to have shape [batch,nmixtures,zdim] if model is mixture of gaussians'
            dotprods_qz = tf.tensordot(sample_qz, tf.transpose(sample_qz), [[-1],[0]])
            norm_nk = tf.tensordot(norms_qz,tf.ones(tf.shape(tf.transpose(norms_qz))),[[-1],[0]])
            norm_lm = tf.tensordot(tf.ones(tf.shape(norms_qz)),tf.transpose(norms_qz),[[-1],[0]])
            distances_qz = norm_nk + norm_lm - 2. * dotprods_qz
        else:
            assert len(sample_qz.get_shape().as_list()) == 2, \
                'latent samples need to have shape [batch,zdim] if model is gaussian'
            dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
            distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz
        """
        TODO : add case where qz is different from pz: pz=mixture, qz=gaussian
        """
        if opts['pz'] == 'mixture':
            dotprods = tf.tensordot(sample_qz, tf.transpose(sample_pz), [[-1],[0]])
            norm_nk = tf.tensordot(norms_qz,tf.ones(tf.shape(tf.transpose(norms_qz))),[[-1],[0]])
            norm_lm = tf.tensordot(tf.ones(tf.shape(norms_pz)),tf.transpose(norms_pz),[[-1],[0]])
            distances = norm_nk + norm_lm - 2. * dotprods
        else:
            dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
            distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
        if kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # Maximal heuristic for the sigma^2 of Gaussian kernel
            # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
            # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
            # sigma2_k = opts['latent_space_dim'] * sigma2_p
            if opts['verbose']:
                sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')

            res1 = tf.exp( - distances_qz / 2. / sigma2_k)
            if opts['pz'] == 'mixture':
                res1 += tf.exp( - distances_pz / (2.*sigma2_k)) / opts['nmixtures']**2
                res1 = tf.multiply(tf.transpose(res1,perm=(1,2,0,3)),
                    1.-tf.eye(n,batch_shape=[opts['nmixtures'],opts['nmixtures']]))
            else:
                res1 += tf.exp( - distances_pz / 2. / sigma2_k)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            if opts['pz'] == 'mixture':
                res2 = tf.exp( - distances / 2. / sigma2_k) / opts['nmixtures']
            else:
                res2 = tf.exp( - distances / 2. / sigma2_k)
            res2 = 2*tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        # elif kernel == 'IMQ':
        #     # k(x, y) = C / (C + ||x - y||^2)
        #     # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        #     # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        #     Cbase = 2 * opts['zdim'] * sigma2_p
        #     stat = 0.
        #     for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        #         C = Cbase * scale
        #         res1 = C / (C + distances_qz)
        #         res1 += C / (C + distances_pz)
        #         res1 = tf.multiply(res1, 1. - tf.eye(n))
        #         res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        #         res2 = C / (C + distances)
        #         res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        #         stat += res1 - res2
        else:
            raise ValueError('%s Unknown kernel' % kernel)

        return stat

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

    def sample_pz(self, num=100, sampling='one_mixture'):
        opts = self.opts
        noise = None
        distr = opts['pz']
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [num, opts['zdim']]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(opts['zdim'])
            cov = opts['sigma_prior']*np.identity(opts['zdim'])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        elif distr == 'mixture':
            if opts['zdim']>=opts['nmixtures']:
                cov = opts['sigma_prior']*np.ones((opts['zdim']))
                means = np.zeros([num,opts['nmixtures'], opts['zdim']])
                for k in range(opts['nmixtures']):
                    means[:,k,k] = np.max([2.0*opts['sigma_prior'],means[:,k,k]])
                # sample for each cluster
                eps = np.random.normal(0.,1.,[num, opts['nmixtures'],opts['zdim']])
                noises = means + np.multiply(eps,cov)
                #sample cluster id
                if sampling == 'one_mixture':
                    mixture = np.random.randint(opts['nmixtures'],size=num)
                    noise = noises[np.arange(num),mixture]
                elif sampling == 'per_mixture':
                    samples_per_mixture = int(num / opts['nmixtures'])
                    class_i = np.repeat(np.arange(opts['nmixtures']),samples_per_mixture,axis=0)
                    mixture = np.zeros([num,],dtype='int32')
                    mixture[(num % opts['nmixtures']):] = class_i
                    noise = noises[np.arange(num),mixture]
                elif sampling == 'all_mixtures':
                    noise = noises
            else:
                assert False, 'Too many mixtures in the latents.'
        else:
            assert False, 'Unknown latent model.'
        return opts['pz_scale'] * noise

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
        enc_test_prev = None
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
                batch_images = data.data[data_ids].astype(np.float)
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
                if counter % opts['print_every'] == 0:
                    now = time.time()

                    # Auto-encoding test images
                    [loss_rec_test, enc_test, rec_test] = self.sess.run(
                            [self.loss_reconstruct,
                             self.encoded,
                             self.reconstructed],
                            feed_dict={self.sample_points: data.test_data[:self.num_pics],
                                       self.is_training: False})

                    if enc_test_prev is not None:
                        changes = np.mean((enc_test - enc_test_prev) ** 2.)
                        encoding_changes.append(changes)
                    else:
                        changes = np.mean((enc_test) ** 2.)
                        encoding_changes.append(changes)

                    enc_test_prev = enc_test

                    # Auto-encoding training images
                    [loss_rec_train, enc_train, rec_train] = self.sess.run(
                            [self.loss_reconstruct,
                             self.encoded,
                             self.reconstructed],
                            feed_dict={self.sample_points: data.data[:self.num_pics],
                                       self.is_training: False})

                    # Random samples generated by the model
                    sample_gen = self.sess.run(
                            self.decoded,
                            feed_dict={self.sample_noise: self.fixed_noise,
                                       self.is_training: False})

                    # Blurriness measures
                    gen_blurr = self.sess.run(
                            self.blurriness,
                            feed_dict={self.sample_points: sample_gen})
                    blurr_vals.append(np.min(gen_blurr))

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d, BATCH/SEC:%.2f' % (
                        epoch + 1, opts['epoch_num'],
                        it + 1, batches_num,
                        float(counter) / (now - self.start_time))
                    debug_str += ' (WAE_LOSS=%.5f, RECON_LOSS=%.5f, ' \
                                 'MATCH_LOSS=%.5f, ' \
                                 'RECON_LOSS_TEST=%.5f, ' \
                                 'SHARPNESS=%.5f)' % (
                                    losses[-1], losses_rec[-1],
                                    losses_match[-1], loss_rec_test, np.min(gen_blurr))
                    logging.error(debug_str)

                    # Printing debug info for encoder variances if applicable
                    if False:
                    #if opts['e_noise'] == 'gaussian':
                        logging.error('Per dimension encoder variances:')
                        per_dim_range = self.debug_sigmas.eval(
                            session = self.sess,
                            feed_dict={self.sample_points: data.test_data[:500],
                                       self.is_training: False})
                        for idim in range(per_dim_range.shape[0]):
                            if per_dim_range[idim][1] > 0.:
                                logging.error(
                                    'dim%.4d: [%.2f; %.2f]  <------' % (idim,
                                       per_dim_range[idim][0],
                                       per_dim_range[idim][1]))
                            else:
                                logging.error(
                                    'dim%.4d: [%.2f; %.2f]' % (idim,
                                       per_dim_range[idim][0],
                                       per_dim_range[idim][1]))

                    # Choosing the 2d projection for Pz vs Qz plots
                    pz_noise = self.sample_pz(opts['plot_num_pics'])
                    if opts['pz'] == 'normal' and opts['zdim'] > 2:
                        # Finding the least Gaussian projection for Qz
                        proj_mat, check = self.least_gaussian_2d(
                            np.vstack([enc_train, enc_test]))
                        # Projecting samples from Qz and Pz on this 2d plain
                        Qz_train = np.dot(enc_train, proj_mat)
                        Qz_test = np.dot(enc_test, proj_mat)
                        Pz = np.dot(pz_noise, proj_mat)
                    else:
                        Qz_train = enc_train[:, :2]
                        Qz_test = enc_test[:, :2]
                        Pz = pz_noise[:, :2]

                    # Making plots
                    save_plots(opts, data.data[:self.num_pics],
                               data.test_data[:self.num_pics],
                               rec_train[:self.num_pics],
                               rec_test[:self.num_pics],
                               sample_gen,
                               Qz_train, Qz_test, Pz,
                               losses_rec, losses_match, blurr_vals,
                               encoding_changes,
                               'res_e%04d_mb%05d.png' % (epoch, it))

        # Save the final model
        if epoch > 0:
            self.saver.save(self.sess,
                             os.path.join(opts['work_dir'],
                                          'checkpoints',
                                          'trained-wae-final'),
                             global_step=counter)

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

def save_plots(opts, sample_train, sample_test,
               recon_train, recon_test,
               sample_gen,
               Qz_train, Qz_test, Pz,
               losses_rec, losses_match, blurr_vals,
               encoding_changes,
               filename):
    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img6 | img5

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   Qz vs Pz plots
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

    # Reconstruction plots
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

    # Sample plots
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
    for img, (gi, gj, title) in zip([img1, img2, img3, img5],
                             [(0, 0, 'train reconstruction'),
                              (0, 1, 'test reconstruction'),
                              (0, 2, 'generated samples'),
                              (1, 2, 'data points')]):
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
                 ha="center", va="bottom", size=30, transform=ax.transAxes)

        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    # Then the Pz vs Qz plot
    ax = plt.subplot(gs[1, 0])
    plt.scatter(Pz[:, 0], Pz[:, 1],
                color='red', s=70, marker='*', label='Pz')
    plt.scatter(Qz_train[:, 0], Qz_train[:, 1], color='blue',
                s=20, marker='x', edgecolors='face', label='Qz_train')
    plt.scatter(Qz_test[:, 0], Qz_test[:, 1], color='green',
                s=20, marker='x', edgecolors='face', label='Qz_test')
    plt.text(0.47, 1., 'Pz vs Qz plot',
             ha="center", va="bottom", size=30, transform=ax.transAxes)
    xmin = min(np.min(Qz_train[:,0]),
               np.min(Qz_test[:,0]))
    xmax = max(np.max(Qz_train[:,0]),
               np.max(Qz_test[:,0]))
    magnify = 0.3
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify

    ymin = min(np.min(Qz_train[:,1]),
               np.min(Qz_test[:,1]))
    ymax = max(np.max(Qz_train[:,1]),
               np.max(Qz_test[:,1]))
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(loc='upper left')

    # The loss curves
    ax = plt.subplot(gs[1, 1])
    total_num = len(losses_rec)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(losses_rec) + 1, x_step)

    y = np.log(np.abs(losses_rec[::x_step]))
    plt.plot(x, y, linewidth=2, color='red', label='log(|rec loss|)')

    y = np.log(np.abs(losses_match[::x_step]))
    plt.plot(x, y, linewidth=2, color='blue', label='log(|match loss|)')

    blurr_mod = np.tile(blurr_vals, (opts['print_every'], 1))
    blurr_mod = blurr_mod.transpose().reshape(-1)
    x_step = max(int(len(blurr_mod)/ 100), 1)
    x = np.arange(1, len(blurr_mod) + 1, x_step)
    y = np.log(blurr_mod[::x_step])
    plt.plot(x, y, linewidth=2, color='orange', label='log(sharpness)')
    if len(encoding_changes) > 0:
        x = np.arange(1, len(losses_rec) + 1)
        y = np.log(encoding_changes)
        x_step = int(len(x) / len(y))
        plt.plot(x[::x_step], y, linewidth=2, color='green', label='log(encoding changes)')
    plt.grid(axis='y')
    plt.legend(loc='upper right')

    # Saving
    utils.create_dir(opts['work_dir'])
    fig.savefig(utils.o_gfile((opts['work_dir'], filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()
