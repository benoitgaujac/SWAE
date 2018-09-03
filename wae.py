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
from loss_functions import matching_penalty, reconstruction_loss, vae_recons_loss, moments_loss
from supervised_functions import accuracy, get_mean_probs, relabelling_mask_from_probs, one_hot
from plot_functions import save_train, save_vizu
from model_nn import cat_encoder, gaussian_encoder
from model_nn import continuous_decoder
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
        sample_size = tf.shape(self.points,out_type=tf.int64)[0]
        range = tf.range(sample_size)
        zero = tf.zeros([tf.cast(sample_size,dtype=tf.int32)],dtype=tf.int64)
        # --- Initialize prior parameters
        self.pz_mean, self.pz_sigma = init_gaussian_prior(opts)
        self.pi0 = init_cat_prior(opts)
        # --- Encoding inputs
        logits, self.enc_mean, self.enc_logSigma = self.encoder(
                                                        self.points,
                                                        False)
        self.pi = ops.softmax(logits,axis=-1)
        # --- Sampling from encoded MoG prior
        self.mixtures_encoded = sample_mixtures(opts, self.enc_mean,
                                                        tf.exp(self.enc_logSigma),
                                                        sample_size,'tensorflow')
        # --- Decoding encoded points (i.e. reconstruct)
        self.reconstructed, _ = self.decoder(self.mixtures_encoded,False,False)
        # --- Reconstructing inputs (only for visualization)
        idx = tf.reshape(tf.multinomial(tf.nn.log_softmax(logits),1),[-1])
        mix_idx = tf.stack([range,idx],axis=-1)
        self.encoded_point = tf.gather_nd(self.mixtures_encoded[:,:,0],mix_idx)
        self.reconstructed_point = tf.gather_nd(self.reconstructed[:,:,0],mix_idx)
        #self.encoded_point = tf.gather_nd(self.mixtures_encoded,mix_idx)
        #self.reconstructed_point = tf.gather_nd(self.reconstructed,mix_idx)
        # --- Sampling from model (only for generation)
        self.decoded, self.logits_decoded = self.decoder(self.sample_noise,
                                                        True,
                                                        True)
        flat_logits = tf.reshape(self.logits_decoded,[-1])
        log_probs = tf.stack([flat_logits,tf.log(1.-tf.exp(flat_logits))],axis=-1)
        flat_vae_decoded = tf.multinomial(logits=log_probs, num_samples=1)
        self.vae_decoded = tf.reshape(flat_vae_decoded, tf.shape(self.decoded))

        # --- Objectives, losses, penalties, pretraining
        # Compute reconstruction cost
        self.loss_reconstruct = reconstruction_loss(opts, self.pi,
                                                        self.points,
                                                        self.reconstructed)
        # self.wae_log_reconstruct = vae_recons_loss(opts, self.pi,
        #                                                 self.points,
        #                                                 self.reconstructed)
        self.wae_log_reconstruct = tf.zeros([1])
        # Compute matching penalty cost
        self.kl_g, self.kl_d, self.match_penalty, dpz, dqz, d= matching_penalty(opts,
                                                        self.pi0, self.pi,
                                                        self.enc_mean, self.enc_logSigma,
                                                        self.pz_mean, self.pz_sigma,
                                                        self.sample_mix_noise, self.mixtures_encoded)
        Mdpz = tf.reduce_max(dpz,axis=[0,2,3,4,-1])
        mdpz = tf.reduce_min(dpz,axis=[0,2,3,4,-1])
        Mdqz = tf.reduce_max(dqz,axis=[0,2,3,4,-1])
        mdqz = tf.reduce_min(dqz,axis=[0,2,3,4,-1])
        Md = tf.reduce_max(d,axis=[0,2,3,4,-1])
        md = tf.reduce_min(d,axis=[0,2,3,4,-1])
        self.Mdistances = tf.stack([Mdpz, Mdqz, Md],axis=-1)
        self.mdistances = tf.stack([mdpz, mdqz, md],axis=-1)
        # Compute Unlabeled obj
        self.objective = self.loss_reconstruct + self.lmbd * self.match_penalty
        # FID score
        self.blurriness = self.compute_blurriness()
        # Pre Training
        self.pretrain_loss()

        # --- Optimizers, savers, etc
        self.add_optimizers()
        self.add_savers()
        self.init = tf.global_variables_initializer()

    def add_model_placeholders(self):
        opts = self.opts
        shape = self.data_shape
        self.points = tf.placeholder(tf.float32,
                                    [None] + shape,
                                    name='points_ph')
        self.sample_mix_noise = tf.placeholder(tf.float32,
                                    [None] + [opts['nmixtures'],opts['nsamples'],opts['zdim']],
                                    name='mix_noise_ph')
        # self.sample_mix_noise = tf.placeholder(tf.float32,
        #                             [None] + [opts['nmixtures'],opts['zdim']],
        #                             name='mix_noise_ph')
        self.sample_noise = tf.placeholder(tf.float32,
                                    [None] + [opts['nmixtures'],opts['zdim']],
                                    name='noise_ph')
        # placeholders fo logistic regression
        self.preds = tf.placeholder(tf.float32, [None, 10], name='predictions') # discrete probabilities
        self.y = tf.placeholder(tf.float32, [None, 10],name='labels') # 0-9 digits recognition => 10 classes

    def add_training_placeholders(self):
        opts = self.opts
        decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        is_training = tf.placeholder(tf.bool, name='is_training_ph')
        alpha = tf.placeholder(tf.float32, name='alpha')
        lmbda = tf.placeholder(tf.float32, name='lambda')

        self.lr_decay = decay
        self.is_training = is_training
        self.lmbd = lmbda

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
        e_cat_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder/cat_params')
        e_gaus_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder/gaus_params')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        prior_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prior')
        #ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if opts['clip_grad_cat']:
        # Clipping gradient if necessary
            grad_cat, var_cat = zip(*opt.compute_gradients(loss=self.objective,
                                                    var_list=e_cat_vars))
            clip_grad, _ = tf.clip_by_global_norm(grad_cat, opts['clip_norm'])
            grad, var = zip(*opt.compute_gradients(loss=self.objective,
                                                    var_list=e_gaus_vars+decoder_vars))
            self.swae_opt = opt.apply_gradients(zip(grad+tuple(clip_grad), var+var_cat))
        elif opts['different_lr_cat']:
        # Different lr for cat if necessary
            opt_cat = self.optimizer(opts['lr_cat'], self.lr_decay)
            grad_cat, var_cat = zip(*opt_cat.compute_gradients(loss=self.objective, var_list=e_cat_vars))
            grad, var = zip(*opt.compute_gradients(loss=self.objective, var_list=e_gaus_vars+decoder_vars))
            self.swae_opt = opt.apply_gradients(zip(grad+tuple(grad_cat), var+var_cat))
        else:
            self.swae_opt = opt.minimize(loss=self.objective, var_list=e_cat_vars+e_gaus_vars+decoder_vars)

        # Pretraining optimizer
        if opts['e_pretrain']:
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_vars+prior_vars)

    def encoder(self, input_points, reuse=False):
        ## Categorical encoding
        logit = cat_encoder(self.opts, inputs=input_points, reuse=reuse,
                                                    is_training=self.is_training)
        ## Gaussian encoding
        if self.opts['e_means']=='fixed':
            sample_size = tf.shape(self.points,out_type=tf.int64)[0]
            eps = tf.zeros([tf.cast(sample_size, dtype=tf.int32), self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
            enc_mean = self.pz_mean + eps
            enc_logSigma = tf.log(self.opts['sigma_prior'])*tf.ones([
                                                    tf.cast(sample_size,dtype=tf.int32),
                                                    self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
        elif self.opts['e_means']=='mean':
            sample_size = tf.shape(self.points,out_type=tf.int64)[0]
            enc_mean, _ = gaussian_encoder(opts, inputs=input_points, reuse=reuse,
                                                    is_training=self.is_training)
            enc_logSigma = tf.log(self.opts['sigma_prior'])*tf.ones([
                                                    tf.cast(sample_size,dtype=tf.int32),
                                                    self.opts['nmixtures'],
                                                    self.opts['zdim']],dtype=tf.float32)
        elif self.opts['e_means']=='learnable':
            enc_mean, enc_logSigma = gaussian_encoder(self.opts,
                                                    inputs=input_points,
                                                    reuse=reuse,
                                                    is_training=self.is_training)
        return logit, enc_mean, enc_logSigma

    def decoder(self, encoded, decode_sample=False, reuse=False):
        noise = tf.reshape(encoded,[-1,self.opts['zdim']])
        recon, log = continuous_decoder(self.opts, noise=noise,
                                                        reuse=reuse,
                                                        is_training=self.is_training)
        if decode_sample:
            out_shape = [-1,self.opts['nmixtures']]+self.data_shape
        else:
            out_shape = [-1,self.opts['nmixtures'],self.opts['nsamples']]+self.data_shape
        reconstructed = tf.reshape(recon,out_shape)
        logits = tf.reshape(log,out_shape)
        # reconstructed = tf.reshape(recon,
        #                 [-1,self.opts['nmixtures'],self.opts['nsamples']]+self.data_shape)
        # logits = tf.reshape(log,
        #                 [-1,self.opts['nmixtures'],self.opts['nsamples']]+self.data_shape)
        return reconstructed, logits

    def pretrain_loss(self):
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        self.pre_loss = moments_loss(self.sample_mix_noise, self.mixtures_encoded)

    def pretrain_encoder(self, data):
        opts=self.opts
        steps_max = 1000
        batch_size = opts['e_pretrain_sample_size']
        train_size = data.num_points
        for step in range(steps_max):
            data_ids = np.random.choice(train_size, batch_size,
                                                       replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_mix_noise = sample_pz(opts, self.pz_mean,
                                              self.pz_sigma,
                                              batch_size,
                                              sampling_mode='all_mixtures')
            [_, pre_loss] = self.sess.run(
                                [self.pre_opt, self.pre_loss],
                                feed_dict={self.points: batch_images,
                                           self.sample_mix_noise: batch_mix_noise,
                                           self.is_training: True})
        logging.error('Pretraining the encoder done.')
        logging.error ('Loss after %d iterations: %.3f' % (steps_max,pre_loss))

    def compute_blurriness(self):
        images = self.points
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

        # data set size
        train_size = data.num_points

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

        batches_num = int(train_size/opts['batch_size'])
        npics = opts['plot_num_pics']
        fixed_noise = sample_pz(opts, self.pz_mean, self.pz_sigma,
                                                        opts['plot_num_pics'],
                                                        sampling_mode = 'per_mixture')
        self.start_time = time.time()

        # Compute bluriness of real data
        real_blurr = self.sess.run(self.blurriness, feed_dict={
                                                        self.points: data.data[:npics]})
        logging.error('Real pictures sharpness = %.5f' % np.min(real_blurr))
        print('')

        losses, losses_rec, losses_match, losses_VAE = [], [], [], []
        kl_gau, kl_dis  = [], []
        loss_rec_test, accuracies, mean_blurr, true_blurr = [], [], [], []
        decay, counter = 1., 0
        wait = 0
        for epoch in range(opts['epoch_num']):
            # Update learning rate if necessary
            if epoch == 50:
                decay = decay / 2.
            if epoch == 100:
                decay = decay / 5.
            if epoch == 150:
                decay = decay / 10.
            # Save the model
            if epoch > 0 and epoch % opts['save_every_epoch'] == 0:
                self.saver.save(self.sess, os.path.join(
                                                        work_dir,'checkpoints',
                                                        'trained-wae'),
                                                        global_step=counter)
            ##### TRAINING LOOP #####
            for it in range(batches_num):
                # Sample batches of data points and Pz noise
                data_ids = np.random.choice(train_size, opts['batch_size'],
                                                        replace=False)
                batch_images = data.data[data_ids].astype(np.float32)
                batch_mix_noise = sample_pz(opts, self.pz_mean,
                                                        self.pz_sigma,
                                                        opts['batch_size'],
                                                        sampling_mode='all_mixtures')
                # Feeding dictionary
                feed_dict={self.points: batch_images,
                           self.sample_mix_noise: batch_mix_noise,
                           self.lr_decay: decay,
                           self.lmbd: opts['lambda'],
                           self.is_training: True}
                # Update encoder and decoder
                if opts['method']=='swae':
                    [_, loss, loss_rec, loss_vae, loss_match] = self.sess.run([self.swae_opt,
                                                        self.objective,
                                                        self.loss_reconstruct,
                                                        self.wae_log_reconstruct,
                                                        self.match_penalty],
                                                        feed_dict=feed_dict)
                    losses_VAE.append(loss_vae)
                    [max_dist, min_dist] = self.sess.run([self.Mdistances,self.mdistances],
                                                        feed_dict=feed_dict)
                    print('')
                    debug_str = 'Max pz: %s' % (np.array2string(max_dist[:,0],precision=4))
                    logging.error(debug_str)
                    debug_str = 'min pz: %s' % (np.array2string(min_dist[:,0],precision=4))
                    logging.error(debug_str)
                    debug_str = 'Max qz: %s' % (np.array2string(max_dist[:,1],precision=4))
                    logging.error(debug_str)
                    debug_str = 'min qz: %s' % (np.array2string(min_dist[:,1],precision=4))
                    logging.error(debug_str)
                    debug_str = 'Max dist: %s' % (np.array2string(max_dist[:,-1],precision=4))
                    logging.error(debug_str)
                    debug_str = 'min dist: %s' % (np.array2string(min_dist[:,-1],precision=4))
                    logging.error(debug_str)
                    print('')
                elif opts['method']=='vae':
                    [_, loss, loss_rec, loss_match, kl_g, kl_d] = self.sess.run(
                                                        [self.swae_opt,
                                                         self.objective,
                                                         self.loss_reconstruct,
                                                         self.match_penalty,
                                                         self.kl_g,
                                                         self.kl_d],
                                                        feed_dict=feed_dict)
                    kl_gau.append(kl_g)
                    kl_dis.append(kl_d)
                losses.append(loss)
                losses_rec.append(loss_rec)
                losses_match.append(loss_match)
                ##### TESTING LOOP #####
                if counter % opts['print_every'] == 0:
                    now = time.time()
                    test_size = np.shape(data.test_data)[0]
                    te_size = max(int(test_size*0.2),opts['batch_size'])
                    te_batches_num = int(te_size/opts['batch_size'])
                    tr_size = test_size - te_size
                    tr_batches_num = int(tr_size/opts['batch_size'])
                    # Determine clusters ID
                    mean_probs = np.zeros((opts['nclasses'],opts['nmixtures']))
                    for it_ in range(tr_batches_num):
                        # Sample batches of data points
                        data_ids = te_size + np.random.choice(tr_size,
                                                        opts['batch_size'],
                                                        replace=True)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        batch_labels = data.test_labels[data_ids].astype(np.float32)
                        pi_train = self.sess.run(self.pi, feed_dict={
                                                        self.points:batch_images,
                                                        self.is_training:False})
                        mean_prob = get_mean_probs(opts,batch_labels,pi_train)
                        mean_probs += mean_prob / tr_batches_num
                    # Determine clusters given mean probs
                    labelled_clusters = relabelling_mask_from_probs(opts, mean_probs)
                    # Test accuracy & loss
                    test_rec = 0.
                    acc_test = 0.
                    for it_ in range(te_batches_num):
                        # Sample batches of data points
                        data_ids =  np.random.choice(te_size,
                                                        opts['batch_size'],
                                                        replace=True)
                        batch_images = data.test_data[data_ids].astype(np.float32)
                        batch_labels = data.test_labels[data_ids].astype(np.float32)
                        [l, pi_test] = self.sess.run([self.loss_reconstruct,
                                                         self.pi],
                                                        feed_dict={self.points:batch_images,
                                                                   self.is_training:False})
                        # Computing accuracy
                        acc = accuracy(batch_labels, pi_test, labelled_clusters)
                        acc_test += acc / te_batches_num
                        test_rec += l / te_batches_num
                    accuracies.append(acc_test)
                    loss_rec_test.append(test_rec)
                    # Auto-encoding unlabeled test images
                    [decoded_test, encoded, p_test] = self.sess.run(
                                                        [self.reconstructed_point,
                                                         self.encoded_point,
                                                         self.pi],
                                                        feed_dict={self.points:data.test_data[:npics],
                                                                   self.is_training:False})
                    # Auto-encoding training images
                    [decoded_train, p_train] = self.sess.run(
                                                        [self.reconstructed_point,
                                                         self.pi],
                                                        feed_dict={self.points:data.data[200:200+npics],
                                                                   self.is_training:False})

                    # Random samples generated by the model
                    sample_gen = self.sess.run(self.decoded,
                                                        feed_dict={self.points:data.data[200:200+npics],
                                                                   self.sample_noise: fixed_noise,
                                                                   self.is_training: False})
                    flat_samples = np.reshape(sample_gen,[-1]+self.data_shape)
                    gen_blurr = self.sess.run(self.blurriness,
                                                        feed_dict={self.points: flat_samples})
                    mean_blurr.append(np.min(gen_blurr))
                    if opts['method']=='vae':
                        true_sample_gen = self.sess.run(self.vae_decoded,
                                                            feed_dict={self.points:data.data[200:200+npics],
                                                                       self.sample_noise: fixed_noise,
                                                                       self.is_training: False})
                        flat_samples = np.reshape(true_sample_gen,[-1]+self.data_shape)
                        gen_blurr = self.sess.run(self.blurriness,
                                                            feed_dict={self.points: flat_samples})
                        true_blurr.append(np.min(gen_blurr))

                    # Prior parameter
                    pi0 = self.sess.run(self.pi0,feed_dict={})

                    # Printing various loss values
                    debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
                                                        epoch + 1, opts['epoch_num'],
                                                        it + 1, batches_num)
                    logging.error(debug_str)
                    debug_str = 'TRAIN LOSS=%.3f' % (losses[-1])
                    logging.error(debug_str)
                    debug_str = 'ACC=%.2f, BLUR=%.3f ' % (100*accuracies[-1],
                                                        mean_blurr[-1])
                    logging.error(debug_str)
                    if opts['method']=='swae':
                        debug_str = 'TEST REC=%.3f, TRAIN REC=%.3f, '\
                                                        'VAE REC=%.3f' % (
                                                        loss_rec_test[-1],
                                                        losses_rec[-1],
                                                        losses_VAE[-1])
                    else:
                        debug_str = 'TEST REC=%.3f, TRAIN REC=%.3f' % (
                                                        loss_rec_test[-1],
                                                        losses_rec[-1])

                    logging.error(debug_str)
                    debug_str = 'MATCH=%.3f' % (opts['lambda']*losses_match[-1])
                    logging.error(debug_str)
                    debug_str = 'Clusters ID: %s' % (str(labelled_clusters))
                    logging.error(debug_str)
                    debug_str = 'Priors: %s' % (np.array2string(pi0,precision=3))
                    logging.error(debug_str)
                    print('')
                    # Making plots
                    #logging.error('Saving images..')
                    save_train(opts, data.data[200:200+npics], data.test_data[:npics],          # images
                                     data.test_labels[:npics],                                  # labels
                                     decoded_train[:npics], decoded_test[:npics],               # reconstructions
                                     p_train, p_test,                                           # mixweights
                                     encoded,                                                   # encoded points
                                     fixed_noise,                                               # prior samples
                                     sample_gen,                                                # samples
                                     losses, losses_match,                                      # losses
                                     losses_rec, loss_rec_test, losses_VAE,                     # rec losses
                                     kl_gau, kl_dis,                                            # KL terms
                                     mean_blurr, true_blurr,                                    # FID score
                                     accuracies,                                                # acc
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

        # # Save the final model
        # if epoch > 0:
        #     self.saver.save(self.sess,
        #                      os.path.join(work_dir,
        #                                   'checkpoints',
        #                                   'trained-wae-final'),
        #                      global_step=counter)

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

            mean_prob = get_mean_probs(opts,batch_labels,prob)
            mean_probs += mean_prob / tr_batches_num
        # Determine clusters given mean probs
        labelled_clusters = relabelling_mask_from_probs(opts, mean_probs)
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
