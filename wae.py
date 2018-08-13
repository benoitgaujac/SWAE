# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

"""
Wasserstein Auto-Encoder models
"""

import sys
import time
import os
import logging

from math import sqrt, cos, sin, pi
import numpy as np
import tensorflow as tf

import ops
import utils
from priors import init_gaussian_prior, init_cat_prior
from sampling_functions import sample_mixtures, sample_pz, generate_linespace
from loss_functions import matching_penalty, reconstruction_loss, moments_loss
from supervised_functions import accuracy, get_mean_probs, relabelling_mask_from_probs, one_hot
from plot_functions import save_train, save_vizu
from model_nn import label_encoder, cat_encoder, gaussian_encoder
from model_nn import continuous_decoder, discrete_decoder
from datahandler import datashapes

import pdb

class WAE(object):

    def __init__(self, opts):
        logging.error('Building the Tensorflow Graph')

        # --- Create session
        self.sess = tf.Session()
        self.opts = opts

        # --- Some of the parameters for future use
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data_shape = datashapes[opts['dataset']]

        # --- Placeholders
        self.add_model_placeholders()
        self.add_training_placeholders()
        sample_size = tf.shape(self.l_points,out_type=tf.int64)[0]
        range = tf.range(sample_size)
        zero = tf.zeros([tf.cast(sample_size,dtype=tf.int32)],dtype=tf.int64)

        # --- Initialize prior parameters
        self.pz_mean, self.pz_sigma = init_gaussian_prior(opts)
        self.pi0 = init_cat_prior(opts)

        # --- Encoding inputs
        l_pi, self.l_enc_mean, self.l_enc_logSigma = self.encoder(self.l_points,
                                                        False)
        idx_label = tf.stack([range,self.l_labels], axis=-1)
        self.l_pi = tf.gather_nd(l_pi,idx_label)

        u_pi, self.u_enc_mean, self.u_enc_logSigma = self.encoder(self.u_points,
                                                        True)
        self.probs = label_encoder(self.opts, self.u_points, False,
                                                        self.is_training)
        self.u_pi = tf.reduce_sum(tf.multiply(u_pi,tf.expand_dims(self.probs,axis=-1)),axis=1)

        # --- Sampling from encoded MoG prior
        self.l_mixtures_encoded = sample_mixtures(opts, self.l_enc_mean,
                                                        tf.exp(self.l_enc_logSigma),
                                                        sample_size,'tensorflow')
        self.u_mixtures_encoded = sample_mixtures(opts, self.u_enc_mean,
                                                        tf.exp(self.u_enc_logSigma),
                                                        sample_size,'tensorflow')
        # --- Decoding encoded points (i.e. reconstruct)
        self.l_reconstructed, self.l_reconstructed_logits = self.decoder(self.l_mixtures_encoded,
                                                        False,
                                                        True)
        self.u_reconstructed, self.u_reconstructed_logits = self.decoder(self.u_mixtures_encoded,
                                                        True,
                                                        True)
        self.labels_reconstructed, self.labels_reconstructed_logits = discrete_decoder(opts,
                                                        self.label_noise,
                                                        False,
                                                        self.is_training)
        # --- Reconstructing inputs (only for visualization)
        idx = tf.reshape(tf.multinomial(self.u_pi, 1),[-1])
        mix_idx = tf.stack([range,idx],axis=-1)
        self.encoded_point = tf.gather_nd(self.u_mixtures_encoded,mix_idx)
        #mix_idx = tf.stack([range,idx],axis=-1)
        self.reconstructed_point = tf.gather_nd(self.u_reconstructed,mix_idx)
        self.reconstructed_logit = tf.gather_nd(self.u_reconstructed_logits,mix_idx)

        # --- Sampling from model (only for generation)
        self.decoded, self.decoded_logits = self.decoder(self.sample_noise,
                                                        True,
                                                        False)
        # --- Objectives, losses, penalties, pretraining
        # Compute reconstruction cost
        self.l_loss_reconstruct = reconstruction_loss(opts, self.l_pi,
                                                        self.l_points,
                                                        self.l_reconstructed,
                                                        self.l_labels,
                                                        tf.argmax(self.labels_reconstructed,axis=-1))
        self.u_loss_reconstruct = reconstruction_loss(opts, self.u_pi,
                                                        self.u_points,
                                                        self.u_reconstructed)
        # Compute matching penalty cost
        self.kl_g, self.kl_d, self.l_cont_penalty, self.l_disc_penalty = matching_penalty(opts,
                                                        self.pi0, self.l_pi,
                                                        self.l_enc_mean, self.l_enc_logSigma,
                                                        self.pz_mean, self.pz_sigma,
                                                        self.l_sample_mix_noise, self.l_mixtures_encoded)
        self.kl_g, self.kl_d, self.u_cont_penalty, self.u_disc_penalty = matching_penalty(opts,
                                                        self.pi0, self.u_pi,
                                                        self.u_enc_mean, self.u_enc_logSigma,
                                                        self.pz_mean, self.pz_sigma,
                                                        self.u_sample_mix_noise, self.u_mixtures_encoded)
        # Compute Labeled obj
        self.l_loss = self.l_loss_reconstruct\
                         + self.lmbd * self.l_cont_penalty\
                         + self.beta * self.l_disc_penalty
        # Compute Unlabeled obj
        self.u_loss = self.u_loss_reconstruct\
                         + self.lmbd * self.u_cont_penalty\
                         + self.beta * self.u_disc_penalty
        # Compute wae obj
        self.objective = self.l_loss + self.alpha * self.u_loss

        # Pre Training
        self.pretrain_loss()

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.l_points = tf.placeholder(tf.float32,
                                    [None] + shape,
                                    name='l_points_ph')
        self.l_labels = tf.placeholder(tf.int64,
                                    [None,],
                                    name='l_labels_ph')
        self.l_sample_mix_noise = tf.placeholder(tf.float32,
                                    [None] + [opts['nmixtures'],opts['zdim']],
                                    name='l_mix_noise_ph')
        self.u_points = tf.placeholder(tf.float32,
                                    [None] + shape,
                                    name='u_points_ph')
        self.u_sample_mix_noise = tf.placeholder(tf.float32,
                                    [None] + [opts['nmixtures'],opts['zdim']],
                                    name='u_mix_noise_ph')
        # self.l_points = l_data
        # self.l_labels = l_label
        # self.l_sample_mix_noise = l_mix_noise
        # self.u_points = u_data
        # self.u_sample_mix_noise = u_mix_noise
        self.sample_noise = tf.placeholder(tf.float32,
                                    [None] + [opts['zdim']],
                                    name='noise_ph')
        # self.sample_noise = noise
        # self.label_noise = tf.placeholder(tf.float32,
        #                             [None,1],
        #                             name='noise_ph')
        label_noise = tf.range(opts['nmixtures'],
                                    dtype=tf.float32,
                                    name='label_noise_ph')
        self.label_noise = tf.expand_dims(label_noise, axis=0)

        # placeholders fo logistic regression
        self.preds = tf.placeholder(tf.float32, [None, 10], name='predictions') # discrete probabilities
        self.y = tf.placeholder(tf.float32, [None, 10],name='labels') # 0-9 digits recognition => 10 classes
        # self.preds = preds
        # self.y = y

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        alpha = tf.placeholder(tf.float32, name='alpha')
        lmbda = tf.placeholder(tf.float32, name='lambda')
        beta = tf.placeholder(tf.float32, name='beta')

        self.lr_decay = decay
        self.is_training = is_training
        self.alpha = alpha
        self.lmbd = lmbda
        self.beta = beta

    def add_savers(self):
        opts = self.opts
        saver = tf.train.Saver(max_to_keep=10)
        # tf.add_to_collection('real_points_ph', self.sample_points)
        # tf.add_to_collection('noise_ph', self.sample_noise)
        # tf.add_to_collection('is_training_ph', self.is_training)
        # if self.enc_mean is not None:
        #     tf.add_to_collection('encoder_mean', self.enc_mean)
        #     tf.add_to_collection('encoder_var', self.enc_logsigma)
        # tf.add_to_collection('encoder', self.encoded_point)
        # tf.add_to_collection('decoder', self.decoded)
        #tf.add_to_collection('lambda', self.lmbd)
        self.saver = saver

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
        prior_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prior')
        ae_vars = encoder_vars + decoder_vars + prior_vars
        if opts['clip_grad']:
            grad, var = zip(*opt.compute_gradients(loss=self.objective, var_list=ae_vars))
            clip_grad, _ = tf.clip_by_global_norm(grad, opts['clip_norm'])
            self.swae_opt = opt.apply_gradients(zip(clip_grad, var))
        else:
            self.swae_opt = opt.minimize(loss=self.objective, var_list=ae_vars)
        # Pretraining optimizer
        pre_opt = self.optimizer(lr)
        self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_vars)

    def encoder(self, input_points, reuse=False):
        ## Categorical encoding
        pi = cat_encoder(self.opts, inputs=input_points, reuse=reuse,
                            is_training=self.is_training)
        ## Gaussian encoding
        if self.opts['e_means']=='fixed':
            eps = tf.zeros([tf.cast(sample_size, dtype=tf.int32), self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
            enc_mean = self.pz_mean + eps
            enc_logSigma = self.opts['init_e_std']*tf.ones([
                                                    tf.cast(sample_size,dtype=tf.int32),
                                                    self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
        elif self.opts['e_means']=='mean':
            enc_mean, _ = gaussian_encoder(opts, inputs=input_points, reuse=reuse,
                                                    is_training=self.is_training)
            enc_logSigma = tf.exp(self.opts['init_e_std'])*tf.ones([
                                                    tf.cast(sample_size,dtype=tf.int32),
                                                    self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
        elif self.opts['e_means']=='learnable':
            enc_mean, enc_logSigma = gaussian_encoder(self.opts,
                                                    inputs=input_points,
                                                    reuse=reuse,
                                                    is_training=self.is_training)
        return pi, enc_mean, enc_logSigma

    def decoder(self, encoded, reuse=False, reshape=True):
        noise = tf.reshape(encoded,[-1,self.opts['zdim']])
        recon, log = continuous_decoder(self.opts, noise=noise,
                                                        reuse=reuse,
                                                        is_training=self.is_training)
        if reshape:
            reconstructed = tf.reshape(recon,
                            [-1,self.opts['nmixtures']]+self.data_shape)
            logits = tf.reshape(log,
                            [-1,self.opts['nmixtures']]+self.data_shape)
        else:
            reconstructed = recon
            logits = log

        return reconstructed, logits

    def pretrain_loss(self):
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        l_pre_loss = moments_loss(self.l_sample_mix_noise, self.l_mixtures_encoded)
        u_pre_loss = moments_loss(self.u_sample_mix_noise, self.u_mixtures_encoded)
        # Loss
        self.pre_loss = l_pre_loss + u_pre_loss

    def pretrain_encoder(self, data):
        opts=self.opts
        steps_max = 500
        batch_size = opts['e_pretrain_sample_size']
        full_train_size = data.num_points
        l_train_size = max(int(full_train_size*opts['lu_split']),5)
        u_train_size = full_train_size-l_train_size
        for step in range(steps_max):
            data_ids = np.random.choice(l_train_size,
                                batch_size,
                                replace=True)
            l_batch_images = data.data[data_ids].astype(np.float32)
            l_batch_mix_noise = sample_pz(opts, self.pz_mean,
                                              self.pz_sigma,
                                              batch_size,
                                              sampling_mode='all_mixtures')
            data_ids = l_train_size + np.random.choice(u_train_size,
                                                       batch_size,
                                                       replace=False)
            u_batch_images = data.data[data_ids].astype(np.float32)
            u_batch_mix_noise = sample_pz(opts, self.pz_mean,
                                              self.pz_sigma,
                                              batch_size,
                                              sampling_mode='all_mixtures')
            [_, pre_loss] = self.sess.run(
                                [self.pre_opt, self.pre_loss],
                                feed_dict={self.l_points: l_batch_images,
                                           self.l_sample_mix_noise: l_batch_mix_noise,
                                           self.u_points: u_batch_images,
                                           self.u_sample_mix_noise: u_batch_mix_noise,
                                           self.is_training: True})
        logging.error('Pretraining the encoder done.')
        logging.error ('Loss after %d iterations: %.3f' % (steps_max,pre_loss))

    def train(self, data, MODEL_DIR, WEIGHTS_FILE):
        """
        Train MoG model with chosen method
        """

        opts = self.opts
        if opts['method']=='swae':
            logging.error('Training WAE')
        elif opts['method']=='vae':
            logging.error('Training VAE')
        print('')

        # Create work_dir
        utils.create_dir(opts['method'])
        work_dir = os.path.join(opts['method'],opts['work_dir'])

        # Split data set
        full_train_size = data.num_points
        l_train_size = max(int(full_train_size*opts['lu_split']),5)
        u_train_size = full_train_size-l_train_size
        debug_str = 'Total:%d, Unlabelled:%d, Labelled:%d' % (
                    full_train_size, u_train_size, l_train_size)
        logging.error(debug_str)
        print('')

        # Init sess and load trained weights if needed
        if opts['use_trained']:
            if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.init)
            if opts['e_pretrain']:
                logging.error('Pretraining the encoder')
                self.pretrain_encoder(data)
        print('')

        batches_num = int(max(l_train_size,u_train_size)/opts['batch_size'])
        npics = opts['plot_num_pics']
        fixed_noise = sample_pz(opts, self.pz_mean, self.pz_sigma,
                                                        opts['plot_num_pics'],
                                                        sampling_mode = 'per_mixture')
        self.start_time = time.time()
        losses, losses_rec, losses_match, losses_xent = [], [], [], []
        kl_gau, kl_dis  = [], []
        decay, counter = 1., 0
        if opts['method']=='swae':
            alpha = opts['alpha']
            lmbda = opts['lambda']
            beta = opts['beta']
        else:
            assert False, 'to implement VAE'
            wae_lmbda = 1
        wait = 0
        for epoch in range(opts['epoch_num']):
            # Update learning rate if necessary
            if epoch == 30:
                decay = decay / 2.
            if epoch == 50:
                decay = decay / 5.
            if epoch == 100:
                decay = decay / 10.
            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(
                                                        work_dir,'checkpoints',
                                                        'trained-wae'),
                                                        global_step=counter)
            # # Initialize mean probs for each epoch
            # mean_probs = np.zeros((10,10))
            ##### TRAINING LOOP #####
            for it in range(batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(l_train_size,
                                                        opts['batch_size'],
                                                        replace=True)
                l_batch_images = data.data[data_ids].astype(np.float32)
                l_batch_labels = data.labels[data_ids].astype(np.float32)
                l_batch_mix_noise = sample_pz(opts, self.pz_mean,
                                                        self.pz_sigma,
                                                        opts['batch_size'],
                                                        sampling_mode='all_mixtures')
                data_ids = l_train_size + np.random.choice(u_train_size,
                                                           opts['batch_size'],
                                                           replace=False)
                u_batch_images = data.data[data_ids].astype(np.float32)
                u_batch_labels = data.labels[data_ids].astype(np.float32)
                u_batch_mix_noise = sample_pz(opts, self.pz_mean,
                                                        self.pz_sigma,
                                                        opts['batch_size'],
                                                        sampling_mode='all_mixtures')
                # Feeding dictionary
                feed_dict={self.l_points: l_batch_images,
                           self.l_labels: l_batch_labels,
                           self.l_sample_mix_noise: l_batch_mix_noise,
                           self.u_points: u_batch_images,
                           self.u_sample_mix_noise: u_batch_mix_noise,
                           self.lr_decay: decay,
                           self.alpha: alpha,
                           self.lmbd: lmbda,
                           self.beta: beta,
                           self.is_training: True}
                # Update encoder and decoder
                if opts['method']=='swae':
                    outputs = self.sess.run([self.swae_opt, self.objective,
                                                        self.l_loss_reconstruct,
                                                        self.l_cont_penalty,
                                                        self.l_disc_penalty,
                                                        self.u_loss_reconstruct,
                                                        self.u_cont_penalty,
                                                        self.u_disc_penalty,
                                                        self.probs],
                                             feed_dict=feed_dict)

                    loss = outputs[1]
                    l_loss_rec, l_loss_match, l_loss_xent = outputs[2:5]
                    u_loss_rec, u_loss_match, u_loss_xent = outputs[5:8]
                    probs_labels = outputs[-1]
                elif opts['method']=='vae':
                    assert False, 'to implement VAE'
                    [_, loss, loss_rec, loss_match, enc_mw, kl_g, kl_d] = self.sess.run(
                                                        [self.swae_opt,
                                                         self.objective,
                                                         self.loss_reconstruct,
                                                         self.penalty,
                                                         self.enc_mixweight,
                                                         self.kl_g,
                                                         self.kl_d],
                                                        feed_dict=feed_dict)
                    kl_gau.append(kl_g)
                    kl_dis.append(kl_d)
                losses.append(loss)
                losses_rec.append([l_loss_rec,u_loss_rec])
                losses_match.append([l_loss_match,u_loss_match])
                losses_xent.append([l_loss_xent,u_loss_xent])
                #mean_probs += get_mean_probs(u_batch_labels,probs_labels) / batches_num
                ##### TESTING LOOP #####
                if counter % opts['print_every'] == 0:
                    now = time.time()
                    test_size = np.shape(data.test_data)[0]
                    te_size = max(int(test_size*0.1),opts['batch_size'])
                    te_batches_num = int(te_size/opts['batch_size'])
                    tr_size = test_size - te_size
                    tr_batches_num = int(tr_size/opts['batch_size'])
                    # Determine clusters ID
                    mean_probs = np.zeros((10,10))
                    for it_ in range(tr_batches_num):
                        # Sample batches of data points
                        data_ids = te_size + np.random.choice(tr_size,
                                                       opts['batch_size'],
                                                       replace=False)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        batch_labels = data.test_labels[data_ids].astype(np.float32)
                        probs_train = self.sess.run(self.probs,
                                                        feed_dict={self.l_points:batch_images,
                                                                   self.u_points:batch_images,
                                                                   self.is_training:False})
                        mean_prob = get_mean_probs(batch_labels,probs_train)
                        mean_probs += mean_prob / tr_batches_num
                    # Determine clusters given mean probs
                    labelled_clusters = relabelling_mask_from_probs(mean_probs)
                    # Test accuracy & loss
                    u_loss_rec_test, l_loss_rec_test = 0., 0.
                    u_acc_test = 0.
                    for it_ in range(te_batches_num):
                        # Sample batches of data points
                        data_ids = te_size + np.random.choice(tr_size,
                                                        opts['batch_size'],
                                                        replace=False)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        batch_labels = data.test_labels[data_ids].astype(np.float32)
                        [u_loss_rec, l_loss_rec, probs_test] = self.sess.run(
                                                        [self.u_loss_reconstruct,
                                                         self.l_loss_reconstruct,
                                                         self.probs],
                                                        feed_dict={self.l_points:batch_images,
                                                                   self.l_labels:batch_labels,
                                                                   self.u_points:batch_images,
                                                                   self.is_training:False})
                        # Computing accuracy
                        u_acc = accuracy(batch_labels, probs_test, labelled_clusters)
                        u_acc_test += u_acc / te_batches_num
                        u_loss_rec_test += u_loss_rec / te_batches_num
                        l_loss_rec_test += l_loss_rec / te_batches_num

                    # Auto-encoding unlabeled test images
                    [rec_test, encoded, labeling, probs_test] = self.sess.run(
                                                        [self.reconstructed_point,
                                                         self.encoded_point,
                                                         self.labels_reconstructed,
                                                         self.probs],
                                                        feed_dict={self.l_points:data.test_data[:npics],
                                                                   self.u_points:data.test_data[:npics],
                                                                   self.is_training:False})
                    # Auto-encoding training images
                    [rec_train, probs_train] = self.sess.run(
                                                        [self.reconstructed_point,
                                                         self.probs],
                                                        feed_dict={self.l_points:data.data[:npics],
                                                                   self.u_points:data.data[:npics],
                                                                   self.is_training:False})
                    # Random samples generated by the model
                    sample_gen = self.sess.run(self.decoded,
                                                        feed_dict={self.l_points:data.data[:npics],
                                                                   self.sample_noise: fixed_noise,
                                                                   self.is_training: False})
                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d, LOSS=%.3f' % (
                                epoch + 1, opts['epoch_num'],
                                it + 1, batches_num,
                                losses[-1])
                    logging.error(debug_str)
                    debug_str = 'Clusters ID: %s' % (str(labelled_clusters))
                    logging.error(debug_str)
                    labs = np.argmax(labeling,axis=-1)
                    debug_str = 'Labelling: %s' % (str(labs))
                    logging.error(debug_str)
                    debug_str = 'TEST REC(L/U)=%.3f/%.3f, TEST ACC=%.2f' % (
                                                        l_loss_rec_test,
                                                        u_loss_rec_test,
                                                        100*u_acc_test)
                    logging.error(debug_str)
                    debug_str = 'MATCH(L/U)=%.3f/%.3f, XENT(L/U)=%.3f/%.3f' % (
                                                        losses_match[-1][0],
                                                        losses_match[-1][1],
                                                        losses_xent[-1][0],
                                                        losses_xent[-1][1])
                    logging.error(debug_str)
                    print('')
                    # Making plots
                    #logging.error('Saving images..')
                    save_train(opts, data.data[:npics], data.test_data[:npics],                 # images
                                     data.test_labels[:npics],                                  # labels
                                     rec_train[:npics], rec_test[:npics],                       # reconstructions
                                     probs_train, probs_test,                                   # mixweights
                                     encoded,                                                   # encoded points
                                     fixed_noise,                                               # prior samples
                                     sample_gen,                                                # samples
                                     losses, losses_rec, losses_match, losses_xent,             # loses
                                     kl_gau, kl_dis,                                            # KL terms
                                     work_dir,                                                  # working directory
                                     'res_e%04d_mb%05d.png' % (epoch, it))                      # filename

                # Update learning rate if necessary and counter
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
                counter += 1

        # Save the final model
        if epoch > 0:
            self.saver.save(self.sess,
                             os.path.join(work_dir,
                                          'checkpoints',
                                          'trained-wae-final'),
                             global_step=counter)

    def test(self, data, MODEL_DIR, WEIGHTS_FILE):
        """
        Test trained MoG model with chosen method
        """
        opts = self.opts
        # Load trained weights
        MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        # Set up
        batch_size = 100
        tr_batches_num = int(data.num_points / batch_size)
        train_size = data.num_points
        te_batches_num = int(np.shape(data.test_data)[0] / batch_size)
        test_size = np.shape(data.test_data)[0]
        debug_str = 'test data size: %d' % (np.shape(data.test_data)[0])
        logging.error(debug_str)

        ### Compute probs
        # Iterate over batches
        logging.error('Determining clusters ID using training..')
        mean_probs = np.zeros((10,10))
        for it in range(tr_batches_num):
            # Sample batches of data points and Pz noise
            data_ids = np.random.choice(train_size,
                                opts['batch_size'],
                                replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_labels = data.labels[data_ids].astype(np.float32)
            prob = self.sess.run(self.enc_mixweight,
                                  feed_dict={self.sample_points: batch_images,
                                             self.is_training: False})

            mean_prob = get_mean_probs(batch_labels,prob)
            mean_probs += mean_prob / tr_batches_num
        # Determine clusters given mean probs
        labelled_clusters = relabelling_mask_from_probs(mean_probs)
        logging.error('Clusters ID:')
        print(labelled_clusters)

        ### Accuracy
        logging.error('Computing losses & accuracy..')
        # Training accuracy & loss
        acc_tr = 0.
        loss_rec_tr, loss_match_tr = 0., 0.
        for it in range(tr_batches_num):
            # Sample batches of data points and Pz noise
            data_ids = np.random.choice(train_size,
                                        batch_size,
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_labels = data.labels[data_ids].astype(np.float32)
            # Accuracy
            probs = self.sess.run(self.enc_mixweight,
                                  feed_dict={self.sample_points: batch_images,
                                             self.is_training: False})
            acc = accuracy(batch_labels,probs,labelled_clusters)
            acc_tr += acc / tr_batches_num
            # loss
            batch_mix_noise = sample_pz(opts, self.pz_mean,
                                              self.pz_cov,
                                              opts['batch_size'],
                                              sampling_mode='all_mixtures')
            [loss_rec, loss_match] = self.sess.run(
                                                [self.loss_reconstruct,
                                                 self.penalty],
                                                feed_dict={self.sample_points: batch_images,
                                                           self.sample_mix_noise: batch_mix_noise,
                                                           self.is_training: False})
            loss_rec_tr += loss_rec / tr_batches_num
            loss_match_tr += loss_match / tr_batches_num

        # Testing acc
        acc_te = 0.
        loss_rec_te, loss_match_te = 0., 0.
        for it in range(te_batches_num):
            # Sample batches of data points and Pz noise
            data_ids = np.random.choice(test_size,
                                        batch_size,
                                        replace=False)
            batch_images = data.test_data[data_ids].astype(np.float32)
            batch_labels = data.test_labels[data_ids].astype(np.float32)
            # Accuracy
            probs = self.sess.run(self.enc_mixweight,
                                  feed_dict={self.sample_points: batch_images,
                                             self.is_training: False})
            acc = accuracy(batch_labels,probs,labelled_clusters)
            acc_te += acc / te_batches_num
            # Testing loss
            batch_mix_noise = sample_pz(opts, self.pz_mean,
                                              self.pz_cov,
                                              batch_size,
                                              sampling_mode='all_mixtures')
            [loss_rec, loss_match] = self.sess.run(
                                                [self.loss_reconstruct,
                                                 self.penalty],
                                                feed_dict={self.sample_points: batch_images,
                                                           self.sample_mix_noise: batch_mix_noise,
                                                           self.is_training: False})
            loss_rec_te += loss_rec / te_batches_num
            loss_match_te += loss_match / te_batches_num

        ### Logs
        debug_str = 'rec train: %.4f, rec test: %.4f' % (loss_rec_tr,
                                                       loss_rec_te)
        logging.error(debug_str)
        debug_str = 'match train: %.4f, match test: %.4f' % (loss_match_tr,
                                                           loss_match_te)
        logging.error(debug_str)
        debug_str = 'acc train: %.2f, acc test: %.2f' % (100.*acc_tr,
                                                             100.*acc_te)
        logging.error(debug_str)

        ### Saving
        filename = 'res_test'
        res_test = np.array((loss_rec_tr, loss_rec_te,
                            loss_match_tr, loss_match_te,
                            acc_tr, acc_te))
        np.save(os.path.join(MODEL_PATH,filename),res_test)

    def reg(self, data, MODEL_DIR, WEIGHTS_FILE):
        """
        Trained a logistic regression on the trained MoG model
        """

        opts = self.opts
        # Load trained weights
        MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        # set up
        epoch_num = 20
        print_every = 2
        batch_size = 100
        tr_batches_num = int(data.num_points / batch_size)
        train_size = data.num_points
        te_batches_num = int(np.shape(data.test_data)[0] / batch_size)
        test_size = np.shape(data.test_data)[0]
        lr = 0.001

        ### Logistic regression model
        # Construct model
        linear_layer = ops.linear(opts, self.preds, 10, scope='log_reg')
        logreg_preds = tf.nn.softmax(linear_layer) # Softmax
        # Minimize error using cross entropy
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(logreg_preds), reduction_indices=1))
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(logreg_preds, 1),tf.argmax(self.y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        ### Optimizer
        opt = tf.train.GradientDescentOptimizer(lr)
        logreg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='log_reg')
        logreg_opt = opt.minimize(loss=cross_entropy, var_list=logreg_vars)
        for var in logreg_vars:
            self.sess.run(var.initializer)

        ### Training loop
        costs, acc_train, acc_test  = [], [], []
        counter = 0
        logging.error('Start training..')
        self.start_time = time.time()
        for epoch in range(epoch_num):
            cost = 0.
            # Iterate over batches
            for it_ in range(tr_batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size,
                                            batch_size,
                                            replace=False)
                batch_images = data.data[data_ids].astype(np.float32)
                # Get preds
                preds = self.sess.run(self.enc_mixweight,
                            feed_dict={self.sample_points: batch_images,
                                                self.is_training: False})
                # linear reg
                batch_labels = one_hot(data.labels[data_ids])
                [_ , c] = self.sess.run([logreg_opt,cross_entropy],
                                        feed_dict={self.preds: preds,
                                                   self.y: batch_labels})
                cost += c / tr_batches_num
                costs.append(cost)
                counter += 1

            if counter==1 or counter % print_every == 0:
                # Testing and logging info
                acc_tr, acc_te  = 0., 0.
                # Training Acc
                for it in range(tr_batches_num):
                    # Sample batches of data points and Pz noise
                    data_ids = np.random.choice(train_size,
                                                batch_size,
                                                replace=False)
                    batch_images = data.data[data_ids].astype(np.float32)
                    preds = self.sess.run(self.enc_mixweight,
                                          feed_dict={self.sample_points: batch_images,
                                                     self.is_training: False})
                    batch_labels = one_hot(data.labels[data_ids])
                    a = self.sess.run(acc,
                                feed_dict={self.preds: preds,
                                            self.y: batch_labels})
                    acc_tr += a/ tr_batches_num
                # Testing Acc
                for it in range(te_batches_num):
                    data_ids = np.random.choice(test_size,
                                                batch_size,
                                                replace=False)
                    batch_images = data.test_data[data_ids].astype(np.float32)
                    preds = self.sess.run(self.enc_mixweight,
                                          feed_dict={self.sample_points: batch_images,
                                                     self.is_training: False})
                    batch_labels = one_hot(data.test_labels[data_ids])
                    a = self.sess.run(acc,
                               feed_dict={self.preds: preds,
                                          self.y: batch_labels})
                    acc_te += a/ te_batches_num

                acc_train.append(acc_tr)
                acc_test.append(acc_te)
                # logs
                debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                            epoch + 1, epoch_num,
                            it_ + 1, tr_batches_num)
                logging.error(debug_str)
                debug_str = 'cost=%.3f, TRAIN ACC=%.2f, TEST ACC=%.2f' % (
                            costs[-1], 100*acc_tr, 100*acc_te)
                logging.error(debug_str)

        ### Saving
        filename = 'logreg'
        xstep = int(len(costs)/100)
        np.savez(os.path.join(MODEL_PATH,filename),
                    costs=np.array(costs[::xstep]),
                    acc_tr=np.array(acc_train),
                    acc_te=np.array(acc_test))

    def vizu(self, data, MODEL_DIR, WEIGHTS_FILE):
        """
        Plot and save different visualizations
        """

        opts = self.opts
        # Load trained weights
        MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
        if not tf.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
            raise Exception("weights file doesn't exist")
        self.saver.restore(self.sess, WEIGHTS_PATH)
        # Set up
        num_pics = 200
        test_size = np.shape(data.test_data)[0]
        step_inter = 20
        num_anchors = 10
        imshape = datashapes[opts['dataset']]
        # Auto-encoding training images
        logging.error('Encoding and decoding train images..')
        rec_train = self.sess.run(self.reconstructed_point,
                                  feed_dict={self.sample_points: data.data[:num_pics],
                                             self.is_training: False})
        # Auto-encoding test images
        logging.error('Encoding and decoding test images..')
        [rec_test, encoded, enc_mw_test] = self.sess.run(
                                [self.reconstructed_point,
                                 self.encoded_point,
                                 self.enc_mixweight],
                                feed_dict={self.sample_points:data.test_data[:num_pics],
                                           self.is_training:False})
        # Encode anchors points and interpolate
        logging.error('Encoding anchors points and interpolating..')
        anchors_ids = np.random.choice(test_size,2*num_anchors,replace=False)
        anchors = data.test_data[anchors_ids]
        enc_anchors = self.sess.run(self.encoded_point,
                                    feed_dict={self.sample_points: anchors,
                                               self.is_training: False})
        enc_interpolation = generate_linespace(opts, step_inter,
                                            'points_interpolation',
                                            anchors=enc_anchors)
        noise = enc_interpolation.reshape(-1,opts['zdim'])
        decoded = self.sess.run(self.decoded,
                                feed_dict={self.sample_noise: noise,
                                           self.is_training: False})
        interpolation = decoded.reshape([-1,step_inter]+imshape)
        start_anchors = anchors[::2]
        end_anchors = anchors[1::2]
        interpolation = np.concatenate((start_anchors[:,np.newaxis],
                                        np.concatenate((interpolation,end_anchors[:,np.newaxis]), axis=1)),
                                        axis=1)
        # Random samples generated by the model
        logging.error('Decoding random samples..')
        prior_noise = sample_pz(opts, self.pz_mean,
                                      self.pz_cov,
                                      num_pics,
                                      sampling_mode = 'per_mixture')
        samples = self.sess.run(self.decoded,
                                   feed_dict={self.sample_noise: prior_noise,
                                              self.is_training: False})
        # Encode prior means and interpolate
        logging.error('Generating latent linespace and decoding..')
        if opts['zdim']==2:
            pz_mean_interpolation = generate_linespace(opts, step_inter,
                                                       'transformation',
                                                   anchors=self.pz_mean)
        else:
            pz_mean_interpolation = generate_linespace(opts, step_inter,
                                                 'priors_interpolation',
                                                   anchors=self.pz_mean)
        noise = pz_mean_interpolation.reshape(-1,opts['zdim'])
        decoded = self.sess.run(self.decoded,
                                feed_dict={self.sample_noise: noise,
                                           self.is_training: False})
        prior_interpolation = decoded.reshape([-1,step_inter]+imshape)


        # Making plots
        logging.error('Saving images..')
        save_vizu(opts, data.data[:num_pics], data.test_data[:num_pics],    # images
                        data.test_labels[:num_pics],                        # labels
                        rec_train, rec_test,                                # reconstructions
                        enc_mw_test,                                        # mixweights
                        encoded,                                            # encoded points
                        prior_noise,                                        # prior samples
                        samples,                                            # samples
                        interpolation, prior_interpolation,                 # interpolations
                        MODEL_PATH)                                         # working directory
