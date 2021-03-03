import numpy as np
import tensorflow as tf
import math

from networks import encoder, decoder
from datahandler import datashapes
from sampling_functions import sample_all_gmm
from loss_functions import cross_entropy_loss, KL, ground_cost, MMD
from utils import get_batch_size

import pdb

class Model(object):

    def __init__(self, opts, pi0, pz_mean, pz_sigma):
        self.opts = opts
        self.output_dim = datashapes[self.opts['dataset']][:-1] \
                          + [2 * datashapes[self.opts['dataset']][-1], ]
        # --- Initialize prior parameters
        self.pi0, self.pz_mean, self.pz_sigma = pi0, pz_mean, pz_sigma

    def forward_pass(self, inputs, is_training, reuse=False):
        """Performs a full pass over the model.

        inputs:                                 [batch,imgdim]
        return:
        enc_cat_logits:                         [batch,K]
        enc_z/enc_gauss_mean/enc_gauss_Sigma:   [batch,K,zdim]
        dec_mean, dec_Sigma:                    [batch,K,imgdim]

        """
        # Encode
        enc_cat_logits, enc_gauss_mean, enc_gauss_Sigma = encoder(
                                        self.opts,
                                        input=inputs,
                                        cat_output_dim = self.opts['nmixtures'],
                                        gaus_output_dim=2*self.opts['nmixtures']*self.opts['zdim'],
                                        scope='encoder',
                                        reuse=reuse,
                                        is_training=is_training)
        enc_gauss_mean = tf.reshape(enc_gauss_mean,[-1,self.opts['nmixtures'],self.opts['zdim']])
        enc_gauss_Sigma = tf.reshape(enc_gauss_Sigma,[-1,self.opts['nmixtures'],self.opts['zdim']])
        enc_z = sample_all_gmm(self.opts, enc_gauss_mean, enc_gauss_Sigma) #[batch,nmixtures,zdim]
        # Decode
        dec_mean, dec_Sigma = decoder(self.opts,
                                        input=enc_z,
                                        nmixtures = self.opts['nmixtures'],
                                        output_dim=self.output_dim,
                                        scope='decoder',
                                        reuse=reuse,
                                        is_training=is_training)

        return enc_cat_logits, enc_z, enc_gauss_mean, enc_gauss_Sigma, dec_mean, dec_Sigma

    def sample_x_from_prior(self, noise):
        """
        Sample is taken to be the mean parameters of the decoder.
        In the case of WAE, this correspond to determinitic decoder,
        for VAE, discrepency between decoder and samples as
        we consider the mean param as the samples from the model

        noise:      [batch,K,zdim]
        return:
        sample_x:   [batch,K,imgdim]
        """
        sample_x, _, = decoder(self.opts, input=noise,
                                        nmixtures = self.opts['nmixtures'],
                                        output_dim=self.output_dim,
                                        scope='decoder',
                                        reuse=True,
                                        is_training=False)
        output_shape = [-1,self.opts['nmixtures']]+datashapes[self.opts['dataset']]

        return tf.reshape(sample_x, output_shape)


class VAE(Model):

    def __init__(self, opts, pi0, pz_mean, pz_sigma):
        super().__init__(opts, pi0, pz_mean, pz_sigma)

    def latent_penalty(self, logits, encoded_mean, encoded_sigma): # To check implementation
        """
        Compute KL divergence between prior and variational distribution
        """
        kl, kl_gauss, kl_cat = KL(self.opts, tf.nn.softmax(logits,axis=-1),
                                    encoded_mean,
                                    encoded_sigma,
                                    self.pi0,
                                    self.pz_mean,
                                    self.pz_sigma,)

        return kl, kl_gauss, kl_cat

    def reconstruction_loss(self, inputs, logits, mean, sigma):
        """
        Compute VAE rec. loss
        """
        pi = tf.nn.softmax(logits,axis=-1)
        rec = cross_entropy_loss(self.opts, inputs, tf.nn.softmax(logits,axis=-1),
                                    mean,
                                    sigma)

        return rec

    def loss(self, inputs, is_training, reuse=False):
        """
        Compute the reconstruction and latent regularizarion of the VAE
        """

        # --- Encoding and reconstructing
        enc_cat_logits, _, enc_mean, enc_Sigma, dec_mean, dec_Sigma = self.forward_pass(
                                    inputs=inputs,
                                    is_training=is_training,
                                    reuse=reuse)

        rec = self.reconstruction_loss(inputs, enc_cat_logits, dec_mean, dec_Sigma)
        kl, kl_gauss, kl_cat = self.latent_penalty(enc_cat_logits,
                                    enc_mean,
                                    enc_Sigma)

        return rec, kl, kl_gauss, kl_cat


class WAE(Model):

    def __init__(self, opts, pi0, pz_mean, pz_sigma):
        super().__init__(opts, pi0, pz_mean, pz_sigma)

    def latent_penalty(self, logits, qz):
        """
        Compute MMD between prior and aggregated posterior
        """
        pz = sample_all_gmm(self.opts, self.pz_mean, self.pz_sigma, self.opts['batch_size'], False)
        mmd = MMD(self.opts, tf.nn.softmax(logits,axis=-1), qz, self.pi0, pz)

        return mmd

    def reconstruction_loss(self, inputs, logits, reconstruction):
        """
        Compute WAE rec. loss
        """
        reconstruction = tf.reshape(reconstruction, [-1,self.opts['nmixtures']]+datashapes[self.opts['dataset']])
        rec = ground_cost(self.opts, inputs, tf.nn.softmax(logits,axis=-1), reconstruction)

        return rec

    def loss(self, inputs, is_training, reuse=False):
        """
        Compute the reconstruction and latent regularizarion of the WAE
        """
        # --- Encoding and reconstructing
        enc_cat_logits, enc_z, _, _, dec_mean, dec_Sigma = self.forward_pass(
                                    inputs=inputs,
                                    is_training=is_training)

        rec = self.reconstruction_loss(inputs, enc_cat_logits, dec_mean)
        mmd = self.latent_penalty(enc_cat_logits, enc_z)


        return rec, mmd, tf.zeros([]), tf.zeros([])
