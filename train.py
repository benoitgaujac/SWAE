import sys
import os
import logging

import numpy as np
import tensorflow as tf

import utils
import models
from priors import init_gaussian_prior, init_cat_prior
from sampling_functions import sample_gmm, generate_linespace, generate_latent_grid
from loss_functions import moments_loss
from supervised_functions import accuracy, get_mean_probs, relabelling_mask_from_probs, one_hot
from plot_functions import save_train, save_vizu

import pdb

class Run(object):

    def __init__(self, opts, data):
        logging.error('Building the Tensorflow Graph')

        # --- Create session
        self.sess = tf.Session()
        self.opts = opts
        self.data = data

        # --- Placeholders
        self.add_ph()

        # --- Initialize prior parameters
        self.pz_mean, self.pz_sigma = init_gaussian_prior(self.opts)
        self.pi0 = init_cat_prior(self.opts)

        # --- Instantiate Model
        if self.opts['model'] == 'VAE':
            self.model = models.VAE(self.opts, self.pi0, self.pz_mean, self.pz_sigma)
        elif self.opts['model'] == 'WAE':
            self.model = models.WAE(self.opts, self.pi0, self.pz_mean, self.pz_sigma)
        else:
            raise ValueError('Unknown {} model' % self.opts['model'])

        # --- obj & losses
        self.rec, self.reg, self.kl_g, self.kl_c = self.model.loss(
                                    inputs=self.data.next_element,
                                    is_training=self.is_training)
        self.objective = self.rec + self.beta * self.reg

        # --- Pre Training
        # self.pretrain_loss()

        # --- Get batchnorm ops for training only
        self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # --- encode & decode pass for vizu
        cat_logits, encoded, _, _, reconstructed, _ = self.model.forward_pass(
                                    inputs=self.obs_points,
                                    is_training=self.is_training,
                                    reuse=True)
        self.pi = tf.nn.softmax(cat_logits,axis=-1)
        idx = tf.reshape(tf.multinomial(cat_logits,1,output_dtype=tf.int32),[-1]) #[batch,]
        batch_size = tf.cast(tf.shape(self.obs_points)[0], tf.int32)
        mix_idx = tf.stack([tf.range(batch_size,dtype=tf.int32),idx],axis=-1)
        self.encoded = tf.gather_nd(encoded, mix_idx) #[batch,zdim]
        if self.opts['model']=='VAE' and self.opts['decoder']=='bernoulli':
            reconstructed = tf.math.sigmoid(reconstructed)
        reconstructed = tf.gather_nd(reconstructed, mix_idx)
        self.reconstructed = tf.reshape(reconstructed, [-1,]+self.data.data_shape)

        # --- Sampling
        self.generated = self.model.sample_x_from_prior(noise=self.pz_samples) #[batch,imdim]

        # --- Optimizers, savers, etc
        self.add_optimizers()

        # --- Init iteratorssess, saver and load trained weights if needed, else init variables
        self.sess = tf.compat.v1.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.initializer = tf.global_variables_initializer()
        self.sess.graph.finalize()

    def add_ph(self):
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.beta = tf.placeholder(tf.float32, name='beta_ph')
        self.obs_points = tf.placeholder(tf.float32,
                                    [None] + self.data.data_shape,
                                    name='points_ph')
        self.pz_samples = tf.placeholder(tf.float32,
                                    [None, self.opts['zdim']],
                                    name='noise_ph')

    def optimizer(self, lr, decay=1.):
        lr *= decay
        if self.opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif self.opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=self.opts['adam_beta1'])
        else:
            assert False, 'Unknown optimizer.'

    def add_optimizers(self):
        # SWAE optimizer
        lr = self.opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='encoder')
        decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='decoder')
        with tf.control_dependencies(self.extra_update_ops):
            self.opt = opt.minimize(loss=self.objective, var_list=encoder_vars + decoder_vars)

        # Pretraining optimizer
        if self.opts['e_pretrain']:
            encoder_gaus_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder/gaus')
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_gaus_vars)

    def pretrain_loss(self):
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        self.pre_loss = moments_loss(self.sample_mix_noise, self.mixtures_encoded)

    def pretrain_encoder(self):
        steps_max = 1000
        batch_size = self['e_pretrain_sample_size']
        train_size = self.data.num_points
        for step in range(steps_max):
            data_ids = np.random.choice(train_size, batch_size,
                                    replace=False)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_mix_noise = sample_gmm(self, self.pz_mean,
                                    self.pz_sigma,
                                    batch_size,
                                    sampling_mode='all')
            [_, pre_loss] = self.sess.run([self.pre_opt, self.pre_loss],
                                    feed_dict={self.points: batch_images,
                                               self.sample_mix_noise: batch_mix_noise,
                                               self.is_training: True})
        logging.error('Pretraining the encoder done.')
        logging.error ('Loss after %d iterations: %.3f' % (steps_max,pre_loss))

    def get_classes(self):
        train_size = 20000
        batch_size = 500
        batch_num = int(train_size / batch_size)
        mean_probs = 0.
        for it_ in range(batch_num):
            idx = np.random.choice(np.arange(self.data.train_size), batch_size, False)
            data, labels = self.data.sample_observations(idx, True)
            pi = self.sess.run(self.pi, feed_dict={
                                self.obs_points: data,
                                self.is_training: False})
            probs = get_mean_probs(self.opts, labels, pi)
            mean_probs += probs / batch_num

        return relabelling_mask_from_probs(self.opts, mean_probs)

    def train(self, WEIGHTS_FILE=None):
        """
        Train MoG model with chosen method
        """

        logging.error('\nTraining {}'.format(self.opts['model']))
        exp_dir = self.opts['exp_dir']
        # - Load trained model
        if self.opts['use_trained']:
            if WEIGHTS_FILE is None:
                    raise Exception("No model/weights provided")
            else:
                if not tf.gfile.IsDirectory(self.opts['exp_dir']):
                                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(self.opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
                if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                                    raise Exception("weights file doesn't exist")
                self.saver.restore(self.sess, WEIGHTS_PATH)
        else:
            self.sess.run(self.initializer)
        # - Set up for training
        train_size = self.data.train_size
        trBatch_num = int(self.data.train_size / self.opts['batch_size'])
        teBatch_num = int(self.data.test_size / self.opts['batch_size'])
        logging.error('\nTrain size: {}, trBatch num.: {}, It num: {}'.format(
                                    self.data.train_size,
                                    trBatch_num,
                                    self.opts['it_num']))
        # - set up for testing
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_gmm(self.opts, self.pz_mean, self.pz_sigma,
                                    5*npics, sampling_mode='mixtures')
        np.random.seed(123)
        idx = np.random.choice(np.arange(self.data.test_size), npics, False)
        data_vizu, labels_vizu = self.data.sample_observations(idx)
        np.random.seed()
        # - Init all monitoring variables
        Losses, Losses_test = [], []
        KL, KL_test = [], []
        Acc, Acc_test = [], []
        decay, decay_rate, fix_decay_steps = 1., .9, 25000
        # - Training
        for it in range(self.opts['it_num']):
            # - Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess, os.path.join(exp_dir,'checkpoints','trained-wae'),
                                    global_step=it)
            #####  TRAINING LOOP #####
            it += 1
            # - Training step
            feed_dict={self.data.handle: self.train_handle,
                                    self.lr_decay: decay,
                                    self.beta: self.opts['beta'],
                                    self.is_training: True}
            _ = self.sess.run(self.opt,feed_dict=feed_dict)
            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                # - Get classes for supervised eval
                classes = self.get_classes()
                # - Train
                feed_dict={self.data.handle: self.train_handle,
                                    self.beta: self.opts['beta'],
                                    self.is_training: False}
                losses = self.sess.run([self.objective,
                                    self.rec, self.reg,
                                    self.kl_g, self.kl_c],
                                    feed_dict=feed_dict)
                Losses.append(losses[:3])
                KL.append(losses[3:])
                # - Test
                loss, kl = np.zeros(3), np.zeros(2)
                for it_ in range(teBatch_num):
                    test_feed_dict={self.data.handle: self.test_handle,
                                    self.beta: self.opts['beta'],
                                    self.is_training: False}
                    losses = self.sess.run([self.objective,
                                    self.rec, self.reg,
                                    self.kl_g, self.kl_c],
                                    feed_dict=test_feed_dict)
                    loss += np.array(losses[:3]) / teBatch_num
                    kl += np.array(losses[3:]) / teBatch_num
                Losses_test.append(loss)
                KL_test.append(kl)
                # - Accurcay
                # training acc
                idx = np.random.choice(np.arange(self.data.train_size), npics, False)
                data_train, labels_train = self.data.sample_observations(idx, True)
                while np.unique(labels_train).shape[0]<self.opts['nclasses']:
                    # resample if needed
                    data_train, labels_train = self.data.sample_observations(idx, True)
                pi = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data_train,
                                    self.is_training: False})
                Acc.append(accuracy(labels_train, pi, classes))
                # testing acc
                acc, means_probs = 0., 0.
                for it_ in range(teBatch_num):
                    idx = np.random.choice(np.arange(self.data.test_size), npics, False)
                    data_test, labels_test = self.data.sample_observations(idx)
                    while np.unique(labels_test).shape[0]<self.opts['nmixtures']:
                        # resample if needed
                        data_test, labels_test = self.data.sample_observations(idx)
                    pi = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data_test,
                                    self.is_training: False})
                    acc += accuracy(labels_test, pi, classes) / teBatch_num
                    means_probs += get_mean_probs(self.opts, labels_test, pi) / teBatch_num
                Acc_test.append(acc)
                # - Printing various loss values
                logging.error('')
                debug_str = 'it: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (
                                    Losses[-1][0],
                                    Losses_test[-1][0])
                logging.error(debug_str)
                debug_str = 'REC=%.3f, TEST REC=%.3f, l*REG=%.3f, l*TEST REG=%.3f' % (
                                    Losses[-1][1],
                                    Losses_test[-1][1],
                                    self.opts['beta']*Losses[-1][2],
                                    self.opts['beta']*Losses_test[-1][2])
                logging.error(debug_str)
                debug_str = 'Acc=%.3f, TEST Acc=%.3f' % (Acc[-1], Acc_test[-1])
                logging.error(debug_str)
                if self.opts['model'] == 'VAE':
                    debug_str = 'gauss KL=%10.3e, cat KL=%10.3e'  % (
                                    KL_test[-1][0],
                                    KL_test[-1][1])
                    logging.error(debug_str)
            ##### Vizu #####
            if it % self.opts['print_every'] == 0:
                # - Encode, decode and sample
                # Auto-encoding test images & samples generated by the model
                [rec_vizu, enc_vizu, mixwise_gen] = self.sess.run([
                                    self.reconstructed,
                                    self.encoded,
                                    self.generated],
                                    feed_dict={self.obs_points: data_vizu,
                                               self.pz_samples: fixed_noise[:npics],
                                               self.is_training: False})
                # Auto-encoding training images
                rec_train = self.sess.run(self.reconstructed, feed_dict={
                                    self.obs_points: data_train,
                                    self.is_training: False})
                # Latent grid
                if self.opts['zdim']==2:
                    nsteps = 10
                    latent_grid = generate_latent_grid(self.opts, nsteps, #[nsteps,nsteps,zdim]
                                        self.pz_mean,
                                        self.pz_sigma)
                    latent_grid = latent_grid.reshape([-1,self.opts['zdim']])
                    latent_interpolation = self.sess.run(self.generated,
                                        feed_dict={self.pz_samples: latent_grid,
                                                   self.is_training: False})
                    latent_interpolation = latent_interpolation.reshape([nsteps,nsteps]+self.data.data_shape)
                else:
                    latent_interpolation = None
                # - Saving plots
                save_train(self.opts, data_train, data_vizu,
                                    rec_train, rec_vizu, mixwise_gen,
                                    enc_vizu, labels_vizu,
                                    fixed_noise, self.pz_mean,
                                    latent_interpolation,
                                    means_probs,
                                    Losses, Losses_test,
                                    Acc, Acc_test,
                                    exp_dir, 'res_it%07d.png' % (it))
            # - Update learning rate if necessary
            if self.opts['lr_decay']:
                # decaying every fix_decay_steps
                if it % fix_decay_steps == 0:
                    decay = decay_rate ** (int(it / fix_decay_steps))
                    logging.error('Reduction in lr: %f\n' % decay)
            # - Logging
            if (it)%50000==0 :
                logging.error('')
                logging.error('Train it.: {}/{}'.format(it,self.opts['it_num']))
        # - Save the final model
        if self.opts['save_final'] and it > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                    'checkpoints',
                                    'trained-{}-final'.format(self.opts['model'])),
                                    global_step=it)
        # - Finale losses & scores
        classes = self.get_classes()
        feed_dict={self.data.handle: self.train_handle,
                    self.beta: self.opts['beta'],
                    self.is_training: False}
        losses = self.sess.run([self.objective,
                                    self.rec, self.reg,
                                    self.kl_g, self.kl_c],
                                    feed_dict=feed_dict)
        Losses.append(losses[:3])
        KL.append(losses[3:])
        loss, kl = np.zeros(3), np.zeros(2)
        for it_ in range(teBatch_num):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.beta: self.opts['beta'],
                            self.is_training: False}
            losses = self.sess.run([self.objective,
                                    self.rec, self.reg,
                                    self.kl_g, self.kl_c],
                                    feed_dict=test_feed_dict)
            loss += np.array(losses[:3]) / teBatch_num
            kl += np.array(losses[3:]) / teBatch_num
        Losses_test.append(loss)
        KL_test.append(kl)
        idx = np.random.choice(np.arange(self.data.train_size), npics, False)
        data_train, labels_train = self.data.sample_observations(idx, True)
        while np.unique(labels_train).shape[0]<self.opts['nmixtures']:
            # resample if needed
            data_train, labels_train = self.data.sample_observations(idx, True)
        pi = self.sess.run(self.pi, feed_dict={
                            self.obs_points: data_train,
                            self.is_training: False})
        Acc.append(accuracy(labels_train, pi, classes))
        means_pi = 0.
        for it_ in range(teBatch_num):
            idx = np.random.choice(np.arange(self.data.test_size), npics, False)
            data_test, labels_test = self.data.sample_observations(idx)
            while np.unique(labels_test).shape[0]<self.opts['nmixtures']:
                # resample if needed
                data_test, labels_test = self.data.sample_observations(idx)
            pi = self.sess.run(self.pi, feed_dict={
                                self.obs_points: data_test,
                                self.is_training: False})
            acc += accuracy(labels_test, pi, classes) / teBatch_num
            means_pi += pi / teBatch_num
        Acc_test.append(acc)
        logging.error('')
        debug_str = 'Training done. '
        logging.error(debug_str)
        debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Losses[-1][0],
                            Losses_test[-1][0])
        logging.error(debug_str)
        debug_str = 'REC=%.3f, TEST REC=%.3f, l*REG=%.3f, l TEST REG=%.3f' % (
                            Losses[-1][1],
                            Losses_test[-1][1],
                            self.opts['beta']*Losses[-1][2],
                            self.opts['beta']*Losses_test[-1][2])
        logging.error(debug_str)
        debug_str = 'Acc=%.3f, TEST Acc=%.3f' % (Acc[-1], Acc_test[-1])
        logging.error(debug_str)
        if self.opts['model'] == 'VAE':
            debug_str = 'gauss KL=%10.3e, cat KL=%10.3e'  % (
                            KL_test[-1][0],
                            KL_test[-1][1])
            logging.error(debug_str)

        # - save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Losses), loss_test=np.array(Losses_test),
                    kl=np.array(KL), kl_test=np.array(KL_test),
                    acc=np.array(Acc), acc_test=np.array(Acc_test))


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
            data_ids = np.random.choice(train_size, opts['batch_size'],
                                                replace=True)
            batch_images = data.test_data[data_ids].astype(np.float32)
            batch_labels = data.test_labels[data_ids].astype(np.float32)
            pi_train = self.sess.run(self.pi, feed_dict={
                                                self.points:batch_images,
                                                self.is_training:False})
            mean_prob = get_mean_probs(self.opts,batch_labels,pi_train)
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
            data_ids = np.random.choice(train_size, batch_size,
                                                replace=True)
            batch_images = data.data[data_ids].astype(np.float32)
            batch_labels = data.labels[data_ids].astype(np.float32)
            batch_mix_noise = sample_gmm(opts, self.pz_mean,
                                                self.pz_cov,
                                                batch_size,
                                                sampling_mode='all')
            # Accuracy & losses
            [loss_rec, loss_match, pi] = self.sess.run([self.loss_reconstruct,
                                                self.match_penalty,
                                                self.pi],
                                                feed_dict={self.points:batch_images,
                                                           self.sample_mix_noise: batch_mix_noise,
                                                           self.is_training:False})
            acc = accuracy(batch_labels,pi,labelled_clusters)
            acc_tr += acc / tr_batches_num
            loss_rec_tr += loss_rec / tr_batches_num
            loss_match_tr += loss_match / tr_batches_num
        # Testing accuracy and losses
        acc_te = 0.
        loss_rec_te, loss_match_te = 0., 0.
        for it in range(te_batches_num):
            # Sample batches of data points and Pz noise
            data_ids = np.random.choice(test_size,
                                        batch_size,
                                        replace=True)
            batch_images = data.test_data[data_ids].astype(np.float32)
            batch_labels = data.test_labels[data_ids].astype(np.float32)
            batch_mix_noise = sample_gmm(opts, self.pz_mean,
                                                self.pz_cov,
                                                batch_size,
                                                sampling_mode='all')
            # Accuracy & losses
            [loss_rec, loss_match, pi] = self.sess.run([self.loss_reconstruct,
                                                self.match_penalty,
                                                self.pi],
                                                feed_dict={self.points:batch_images,
                                                           self.sample_mix_noise: batch_mix_noise,
                                                           self.is_training:False})
            acc = accuracy(batch_labels,probs,labelled_clusters)
            acc_te += acc / tr_batches_num
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
        num_pics = 1000
        test_size = np.shape(data.test_data)[0]
        step_inter = 20
        num_anchors = opts['nmixtures']
        imshape = self.data.data_shape
        # Auto-encoding training images
        logging.error('Encoding and decoding train images..')
        rec_train = self.sess.run(self.reconstructed_point,
                                  feed_dict={self.points: data.data[:num_pics],
                                             self.is_training: False})
        # Auto-encoding test images
        logging.error('Encoding and decoding test images..')
        [rec_test, encoded, pi] = self.sess.run(
                                [self.reconstructed_point,
                                 self.encoded_point,
                                 self.pi],
                                feed_dict={self.points:data.test_data[:num_pics],
                                           self.is_training:False})
        # Encode anchors points and interpolate
        logging.error('Encoding anchors points and interpolating..')
        anchors_ids = np.random.choice(test_size,2*num_anchors,replace=False)
        anchors = data.test_data[anchors_ids]
        enc_anchors = self.sess.run(self.encoded_point,
                                feed_dict={self.points: anchors,
                                           self.is_training: False})
        enc_interpolation = generate_linespace(opts, step_inter,
                                'points_interpolation',
                                anchors=enc_anchors)
        #noise = enc_interpolation.reshape(-1,opts['zdim'])
        noise = np.transpose(enc_interpolation,(1,0,2))
        decoded = self.sess.run(self.decoded,
                                feed_dict={self.sample_noise: noise,
                                           self.is_training: False})
        #interpolation = decoded.reshape([-1,step_inter]+imshape)
        interpolation = np.transpose(decoded,(1,0,2,3,4))
        start_anchors = anchors[::2]
        end_anchors = anchors[1::2]
        interpolation = np.concatenate((start_anchors[:,np.newaxis],
                                        np.concatenate((interpolation,end_anchors[:,np.newaxis]), axis=1)),
                                        axis=1)
        # Random samples generated by the model
        logging.error('Decoding random samples..')
        prior_noise = sample_gmm(opts, self.pz_mean,
                                self.pz_sigma,
                                num_pics,
                                sampling_mode = 'mixtures')
        samples = self.sess.run(self.decoded,
                               feed_dict={self.sample_noise: prior_noise,
                                          self.is_training: False})
        # Encode prior means and interpolate
        logging.error('Generating latent linespace and decoding..')
        ancs = np.concatenate((self.pz_mean,self.pz_mean[0][np.newaxis,:]),axis=0)
        if opts['zdim']==2:
            pz_mean_interpolation = generate_linespace(opts, step_inter+2,
                                                       'transformation',
                                                   anchors=ancs)
        else:
            pz_mean_interpolation = generate_linespace(opts, step_inter+2,
                                                 'priors_interpolation',
                                                   anchors=ancs)
        #noise = pz_mean_interpolation.reshape(-1,opts['zdim'])
        noise = np.transpose(pz_mean_interpolation,(1,0,2))
        decoded = self.sess.run(self.decoded,
                                feed_dict={self.sample_noise: noise,
                                           self.is_training: False})
        #prior_interpolation = decoded.reshape([-1,step_inter]+imshape)
        prior_interpolation = np.transpose(decoded,(1,0,2,3,4))



        # Making plots
        logging.error('Saving images..')
        save_vizu(opts, data.data[:num_pics], data.test_data[:num_pics],    # images
                        data.test_labels[:num_pics],                        # labels
                        rec_train, rec_test,                                # reconstructions
                        pi,                                                 # mixweights
                        encoded,                                            # encoded points
                        prior_noise,                                        # prior samples
                        samples,                                            # samples
                        interpolation, prior_interpolation,                 # interpolations
                        MODEL_PATH)                                         # working directory

    # def reg(self, data, MODEL_DIR, WEIGHTS_FILE):
    #     """
    #     Trained a logistic regression on the trained MoG model
    #     """
    #
    #     opts = self.opts
    #     # Load trained weights
    #     MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
    #     if not tf.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
    #         raise Exception("weights file doesn't exist")
    #     self.saver.restore(self.sess, WEIGHTS_PATH)
    #     # set up
    #     epoch_num = 20
    #     print_every = 2
    #     batch_size = 100
    #     tr_batches_num = int(data.num_points / batch_size)
    #     train_size = data.num_points
    #     te_batches_num = int(np.shape(data.test_data)[0] / batch_size)
    #     test_size = np.shape(data.test_data)[0]
    #     lr = 0.001
    #
    #     ### Logistic regression model
    #     # Construct model
    #     linear_layer = ops.linear(opts, self.preds, 10, scope='log_reg')
    #     logreg_preds = tf.nn.softmax(linear_layer) # Softmax
    #     # Minimize error using cross entropy
    #     cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(logreg_preds), reduction_indices=1))
    #     # Accuracy
    #     correct_prediction = tf.equal(tf.argmax(logreg_preds, 1),tf.argmax(self.y, 1))
    #     acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    #     ### Optimizer
    #     opt = tf.train.GradientDescentOptimizer(lr)
    #     logreg_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='log_reg')
    #     logreg_opt = opt.minimize(loss=cross_entropy, var_list=logreg_vars)
    #     for var in logreg_vars:
    #         self.sess.run(var.initializer)
    #
    #     ### Training loop
    #     costs, acc_train, acc_test  = [], [], []
    #     counter = 0
    #     logging.error('Start training..')
    #     self.start_time = time.time()
    #     for epoch in range(epoch_num):
    #         cost = 0.
    #         # Iterate over batches
    #         for it_ in range(tr_batches_num):
    #             # Sample batches of data points and Pz noise
    #             data_ids = np.random.choice(train_size,
    #                                         batch_size,
    #                                         replace=False)
    #             batch_images = data.data[data_ids].astype(np.float32)
    #             # Get preds
    #             preds = self.sess.run(self.enc_mixweight,
    #                         feed_dict={self.points: batch_images,
    #                                             self.is_training: False})
    #             # linear reg
    #             batch_labels = one_hot(data.labels[data_ids])
    #             [_ , c] = self.sess.run([logreg_opt,cross_entropy],
    #                                     feed_dict={self.preds: preds,
    #                                                self.y: batch_labels})
    #             cost += c / tr_batches_num
    #             costs.append(cost)
    #             counter += 1
    #
    #         if counter==1 or counter % print_every == 0:
    #             # Testing and logging info
    #             acc_tr, acc_te  = 0., 0.
    #             # Training Acc
    #             for it in range(tr_batches_num):
    #                 # Sample batches of data points and Pz noise
    #                 data_ids = np.random.choice(train_size,
    #                                             batch_size,
    #                                             replace=False)
    #                 batch_images = data.data[data_ids].astype(np.float32)
    #                 preds = self.sess.run(self.enc_mixweight,
    #                                       feed_dict={self.points: batch_images,
    #                                                  self.is_training: False})
    #                 batch_labels = one_hot(data.labels[data_ids])
    #                 a = self.sess.run(acc,
    #                             feed_dict={self.preds: preds,
    #                                         self.y: batch_labels})
    #                 acc_tr += a/ tr_batches_num
    #             # Testing Acc
    #             for it in range(te_batches_num):
    #                 data_ids = np.random.choice(test_size,
    #                                             batch_size,
    #                                             replace=False)
    #                 batch_images = data.test_data[data_ids].astype(np.float32)
    #                 preds = self.sess.run(self.enc_mixweight,
    #                                       feed_dict={self.points: batch_images,
    #                                                  self.is_training: False})
    #                 batch_labels = one_hot(data.test_labels[data_ids])
    #                 a = self.sess.run(acc,
    #                            feed_dict={self.preds: preds,
    #                                       self.y: batch_labels})
    #                 acc_te += a/ te_batches_num
    #
    #             acc_train.append(acc_tr)
    #             acc_test.append(acc_te)
    #             # logs
    #             debug_str = 'EPOCH: %d/%d, BATCH:%d/%d' % (
    #                         epoch + 1, epoch_num,
    #                         it_ + 1, tr_batches_num)
    #             logging.error(debug_str)
    #             debug_str = 'cost=%.3f, TRAIN ACC=%.2f, TEST ACC=%.2f' % (
    #                         costs[-1], 100*acc_tr, 100*acc_te)
    #             logging.error(debug_str)
    #
    #     ### Saving
    #     filename = 'logreg'
    #     xstep = int(len(costs)/100)
    #     np.savez(os.path.join(MODEL_PATH,filename),
    #                 costs=np.array(costs[::xstep]),
    #                 acc_tr=np.array(acc_train),
    #                 acc_te=np.array(acc_test))
    #
