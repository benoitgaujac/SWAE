import sys
import os
import logging

import numpy as np
import tensorflow as tf

import utils
import models
from priors import init_gaussian_prior, init_cat_prior
from sampling_functions import sample_gmm, sample_all_gmm, generate_linespace, generate_latent_grid
from loss_functions import moments_loss, moments_loss_empirical_pz
from supervised_functions import accuracy, get_mean_probs, relabelling_mask_from_probs, one_hot
from plot_functions import save_train, save_plot

import pdb

class Run(object):

    def __init__(self, opts, data):
        logging.error('Building the Tensorflow Graph')

        # --- Create session
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
        self.pretrain_loss()

        # --- Get batchnorm ops for training only
        self.extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        # --- encode & decode pass for vizu
        cat_logits, encoded, _, _, reconstructed, _ = self.model.forward_pass(
                                    inputs=self.obs_points,
                                    is_training=self.is_training,
                                    reuse=True)
        self.pi = tf.nn.softmax(cat_logits,axis=-1)
        idx = tf.reshape(tf.compat.v1.multinomial(cat_logits,1,output_dtype=tf.int32),[-1]) #[batch,]
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
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)
        self.initializer = tf.compat.v1.global_variables_initializer()
        self.sess.graph.finalize()

    def add_ph(self):
        self.lr_decay = tf.compat.v1.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training_ph')
        self.beta = tf.compat.v1.placeholder(tf.float32, name='beta_ph')
        self.obs_points = tf.compat.v1.placeholder(tf.float32,
                                    [None] + self.data.data_shape,
                                    name='points_ph')
        self.pz_samples = tf.compat.v1.placeholder(tf.float32,
                                    [None, self.opts['zdim']],
                                    name='noise_ph')

    def optimizer(self, lr, decay=1.):
        lr *= decay
        if self.opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif self.opts['optimizer'] == 'adam':
            return tf.compat.v1.train.AdamOptimizer(lr, beta1=self.opts['adam_beta1'])
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
        if self.opts['pretrain_encoder']:
            encoder_gaus_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='encoder/gaus')
            pre_opt = self.optimizer(0.001)
            self.pre_opt = pre_opt.minimize(loss=self.pre_loss, var_list=encoder_gaus_vars)

    def pretrain_loss(self):
        # Adding ops to pretrain the encoder so that mean and covariance
        # of Qz will try to match those of Pz
        _, encoded, _, _, _, _ = self.model.forward_pass(
                                    inputs=self.data.next_element,
                                    is_training=self.is_training,
                                    reuse=True)
        if self.opts['pretrain_empirical_pz']:
            pz = sample_all_gmm(self.opts, self.pz_mean, self.pz_sigma,
                                        self.opts['batch_size'], False)
            self.pre_loss = moments_loss_empirical_pz(encoded, pz)
        else:
            self.pre_loss = moments_loss(encoded, self.pz_mean, self.pz_sigma)

    def pretrain_encoder(self):
        logging.error('\nPre training encoder...')
        it_num = 10000
        for _ in range(it_num):
            _, pre_loss = self.sess.run([self.pre_opt, self.pre_loss],
                                    feed_dict={self.data.handle: self.train_handle,
                                               self.is_training: True})
        logging.error('Pretraining the done.')
        logging.error ('Loss after %d iterations: %.3f' % (it_num,pre_loss))

    def get_classes(self):
        batch_size = 1000
        batch_num = int(self.data.train_size / batch_size)
        mean_probs, c = 0., 0
        for it_ in range(batch_num):
            idx = np.random.choice(np.arange(self.data.train_size), batch_size, False)
            data, labels = self.data.sample_observations(idx, True)
            if np.unique(labels).shape[0]==self.opts['nclasses']:
                pi = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data,
                                    self.is_training: False})
                probs = get_mean_probs(self.opts, labels, pi)
                mean_probs += probs
                c += 1
        mean_probs /= c
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
                if not tf.io.gfile.IsDirectory(self.opts['exp_dir']):
                                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(self.opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
                if not tf.io.gfile.Exists(WEIGHTS_PATH+".meta"):
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
        wait, decay = 1, 1.
        # - Pre training encoder if needed
        if self.opts['pretrain_encoder']:
            self.pretrain_encoder()
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
                # while np.unique(labels_train).shape[0]<self.opts['nmixtures']:
                #     # resample if needed
                #     data_train, labels_train = self.data.sample_observations(idx, True)
                if np.unique(labels_train).shape[0]==self.opts['nclasses']:
                    pi = self.sess.run(self.pi, feed_dict={
                                        self.obs_points: data_train,
                                        self.is_training: False})
                    Acc.append(accuracy(labels_train, pi, classes))
                # testing acc
                acc, c, means_pi = 0., 0, 0.
                for it_ in range(teBatch_num):
                    idx = np.random.choice(np.arange(self.data.test_size), npics, False)
                    data_test, labels_test = self.data.sample_observations(idx)
                    # while np.unique(labels_test).shape[0]<self.opts['nmixtures']:
                    #     # resample if needed
                    #     data_test, labels_test = self.data.sample_observations(idx)
                    if np.unique(labels_test).shape[0]==self.opts['nmixtures']:
                        pi = self.sess.run(self.pi, feed_dict={
                                        self.obs_points: data_test,
                                        self.is_training: False})
                        acc += accuracy(labels_test, pi, classes)
                        means_pi += get_mean_probs(self.opts, labels_test, pi)
                        c += 1
                Acc_test.append(acc / c)
                means_pi /= c
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
                [rec_vizu, mixwise_gen] = self.sess.run([
                                    self.reconstructed,
                                    self.generated],
                                    feed_dict={self.obs_points: data_vizu,
                                               self.pz_samples: fixed_noise[:npics],
                                               self.is_training: False})
                idx = np.random.choice(np.arange(self.data.test_size), 10*npics, False)
                data_enc_vizu, labels_enc_vizu = self.data.sample_observations(idx)
                enc_vizu = self.sess.run(self.encoded,
                                    feed_dict={self.obs_points: data_enc_vizu,
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
                                    enc_vizu, labels_enc_vizu,
                                    fixed_noise, self.pz_mean,
                                    latent_interpolation,
                                    means_pi,
                                    Losses, Losses_test,
                                    Acc, Acc_test,
                                    exp_dir, 'res_it%07d.png' % (it))
            # - Update learning rate if necessary
            if self.opts['lr_decay']:
                # First 200 epochs do nothing
                if it >= 200*trBatch_num:
                    # If no significant progress was made in last 50 epochs
                    # then decrease the learning rate.
                    last_100 = int(100 * trBatch_num / self.opts['evaluate_every'])
                    if Losses[-1][1] < np.amin(np.array(Losses)[-last_100:,1]):
                        wait = 0
                    else:
                        wait += 1
                    if wait > 100 * trBatch_num:
                        decay *= 0.5
                        logging.error('Reduction in lr: %f' % decay)
                        wait = 0
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
        # while np.unique(labels_train).shape[0]<self.opts['nmixtures']:
        #     # resample if needed
        #     data_train, labels_train = self.data.sample_observations(idx, True)
        if np.unique(labels_train).shape[0]==self.opts['nmixtures']:
            pi = self.sess.run(self.pi, feed_dict={
                                self.obs_points: data_train,
                                self.is_training: False})
            Acc.append(accuracy(labels_train, pi, classes))
        acc, c = 0., 0
        for it_ in range(teBatch_num):
            idx = np.random.choice(np.arange(self.data.test_size), npics, False)
            data_test, labels_test = self.data.sample_observations(idx)
            # while np.unique(labels_test).shape[0]<self.opts['nmixtures']:
            #     # resample if needed
            #     data_test, labels_test = self.data.sample_observations(idx)
            if np.unique(labels_test).shape[0]==self.opts['nmixtures']:
                pi = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data_test,
                                    self.is_training: False})
                acc += accuracy(labels_test, pi, classes)
                c += 1
        Acc_test.append(acc / c)
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
        if not tf.io.gfile.IsDirectory(MODEL_PATH):
            raise Exception("model doesn't exist")
        WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
        if not tf.io.gfile.Exists(WEIGHTS_PATH+".meta"):
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


    def plot(self, WEIGHTS_FILE=None):
        """
        Plots reconstructions, latent transversals and model samples
        """

        # - Load trained model
        if WEIGHTS_FILE is None:
                raise Exception("No model/weights provided")
        else:
            if not tf.compat.v1.gfile.IsDirectory(self.opts['exp_dir']):
                raise Exception("model doesn't exist")
            WEIGHTS_PATH = os.path.join(self.opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
            if not tf.compat.v1.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)

        # - set up
        npics = self.opts['plot_num_pics']
        nsteps = 20
        fixed_noise = sample_gmm(self.opts, self.pz_mean,
                                    self.pz_sigma,
                                    5*npics,
                                    sampling_mode='mixtures')
        np.random.seed(123)
        idx = np.random.choice(np.arange(self.data.test_size), npics, False)
        data_vizu, _ = self.data.sample_observations(idx)
        np.random.seed()

        # - encode, decode and sample
        [rec_vizu, mixwise_gen] = self.sess.run([self.reconstructed,
                                    self.generated],
                                    feed_dict={self.obs_points: data_vizu,
                                               self.pz_samples: fixed_noise[:npics],
                                               self.is_training: False})
        batch_size = 1000
        teBatch_num = int(self.data.test_size / batch_size)
        labels_enc_vizu, enc_vizu = [], []
        for n in range(teBatch_num):
            idx = np.random.choice(np.arange(self.data.test_size), batch_size, False)
            data, labels = self.data.sample_observations(idx)
            enc = self.sess.run(self.encoded, feed_dict={self.obs_points: data,
                                    self.is_training: False})
            labels_enc_vizu.append(labels)
            enc_vizu.append(enc)
        labels_enc_vizu = np.concatenate(labels_enc_vizu, axis=0)
        enc_vizu = np.concatenate(enc_vizu, axis=0)

        # - Latent grid/interpolation
        if self.opts['zdim']==2:
            latent_grid = generate_latent_grid(self.opts, nsteps, #[nsteps,nsteps,zdim]
                                    self.pz_mean,
                                    self.pz_sigma)
            latent_grid = latent_grid.reshape([-1,self.opts['zdim']])
            latent_interpolation = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: latent_grid,
                                               self.is_training: False})
            latent_interpolation = latent_interpolation.reshape([nsteps,nsteps]+self.data.data_shape)
        else:
            prior_modes_inter = generate_linespace(self.opts, nsteps, #[nmixtures-1,nsteps,zdim]
                                    'priors_interpolation',
                                    anchors=self.pz_mean)
            prior_modes_inter = prior_modes_inter.reshape([-1,self.opts['zdim']])
            decoded = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: prior_modes_inter,
                                               self.is_training: False})
            latent_interpolation = decoded.reshape([-1,nsteps]+self.data.data_shape)

        # - means_probs
        # train
        trProbs = np.zeros([self.opts['nclasses'], self.opts['nmixtures']])
        trBatch_num = int(self.data.train_size / batch_size)
        c = 0
        for n in range(trBatch_num):
            idx = np.random.choice(np.arange(self.data.train_size), batch_size, False)
            data, labels = self.data.sample_observations(idx,True)
            if np.unique(labels).shape[0]==self.opts['nmixtures']:
                probs = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data,
                                    self.is_training: False})
                trProbs += get_mean_probs(self.opts, labels, probs)
                c += 1
        trProbs /= c
        # test
        teProbs = np.zeros([self.opts['nclasses'], self.opts['nmixtures']])
        c = 0
        for n in range(teBatch_num):
            idx = np.random.choice(np.arange(self.data.test_size), batch_size, False)
            data, labels = self.data.sample_observations(idx)
            if np.unique(labels).shape[0]==self.opts['nmixtures']:
                probs = self.sess.run(self.pi, feed_dict={
                                    self.obs_points: data,
                                    self.is_training: False})
                teProbs += get_mean_probs(self.opts, labels, probs)
                c += 1
        teProbs /= c
        # Making plots
        save_plot(self.opts, data_vizu, rec_vizu, mixwise_gen, # rec, samples
                                    enc_vizu, labels_enc_vizu,   # enc, labels
                                    fixed_noise, self.pz_mean,   # prior noise, means
                                    latent_interpolation,   # interpolations
                                    trProbs, teProbs,   # mean probs
                                    self.opts['exp_dir'])   # working directory

    # def reg(self, data, MODEL_DIR, WEIGHTS_FILE):
    #     """
    #     Trained a logistic regression on the trained MoG model
    #     """
    #
    #     opts = self.opts
    #     # Load trained weights
    #     MODEL_PATH = os.path.join(opts['method'],MODEL_DIR)
    #     if not tf.io.gfile.IsDirectory(MODEL_PATH):
    #         raise Exception("model doesn't exist")
    #     WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
    #     if not tf.io.gfile.Exists(WEIGHTS_PATH+".meta"):
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
