# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import shutil
import random
import logging
import gzip
import zipfile
import tensorflow as tf
import numpy as np
from six.moves import cPickle
import urllib.request
import requests
from scipy.io import loadmat
from sklearn.feature_extraction import image
import struct
from tqdm import tqdm
from PIL import Image
import sys
import tarfile
import h5py
from math import ceil

import utils

import pdb

# Path to data
# data_dir = '../data'

datashapes = {}
datashapes['mnist'] = [32, 32, 1]
datashapes['svhn'] = [32, 32, 3]


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """


    def __init__(self, opts):
        """Load the dataset and fill all the necessary variables.

        """
        self.dataset = opts['dataset']
        self.data_dir = os.path.join(opts['data_dir'], self.dataset)
        # load data
        logging.error('\n Loading {}.'.format(self.dataset))
        if self.dataset == 'mnist':
            self._load_mnist(opts)
        elif self.dataset == 'svhn':
            self._load_svhn(opts)
        else:
            raise ValueError('Unknown {} dataset' % self.dataset)
        logging.error('Loading Done.')

    def _load_mnist(self, opts):
        """Load data from MNIST or ZALANDO files.

        """
        # loading label
        tr_Y, te_Y = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(60000*1), dtype=np.uint8)
            tr_Y = loaded.reshape((60000,)).astype(np.int64)
        with gzip.open(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
            fd.read(8)
            loaded = np.frombuffer(fd.read(10000*1), dtype=np.uint8)
            te_Y = loaded.reshape((10000,)).astype(np.int64)
        y = np.concatenate((tr_Y, te_Y), axis=0)
        all_labels = y
        # loading images
        tr_X, te_X = None, None
        with gzip.open(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(60000*28*28*1), dtype=np.uint8)
            tr_X = loaded.reshape((60000, 28, 28, 1)).astype(np.float32)
        with gzip.open(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
            fd.read(16)
            loaded = np.frombuffer(fd.read(10000*28*28*1), dtype=np.uint8)
            te_X = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)
        X = np.concatenate((tr_X, te_X), axis=0)
        all_data = X / 255.
        # shuffling data
        np.random.seed()
        idx_random = np.random.permutation(all_labels.shape[0])
        # split train/test
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>all_labels.shape[0]-10000:
            tr_stop = all_labels.shape[0] - 10000
        else:
            tr_stop = opts['train_dataset_size']
        self.data_train = all_data[idx_random[:tr_stop]]
        self.labels_train = all_labels[idx_random[:tr_stop]]
        self.data_test = all_data[idx_random[-10000:]]
        self.labels_test = all_labels[idx_random[-10000:]]
        # dataset size
        self.train_size = self.data_train.shape[0]
        self.test_size = self.data_test.shape[0]
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.data.Dataset.from_tensor_slices(self.data_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(self.data_test)
        # pad data to 32x32
        def pad_mnist(x):
            paddings = [[2,2], [2,2], [0,0]]
            return tf.pad(x, paddings, mode='CONSTANT', constant_values=0.)
        dataset_train = dataset_train.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = dataset_test.map(pad_mnist,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = tf.compat.v1.data.make_initializable_iterator(dataset_train)
        self.iterator_test = tf.compat.v1.data.make_initializable_iterator(dataset_test)

        # Global iterator
        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        self.next_element = tf.compat.v1.data.Iterator.from_string_handle(
            self.handle, tf.compat.v1.data.get_output_types(dataset_train), tf.compat.v1.data.get_output_shapes(dataset_train)).get_next()

    def _load_svhn(self, opts):
        """Load data from SVHN files.

        """
        # Helpers to process raw data
        def convert_imgs_to_array(img_array):
            rows = datashapes['svhn'][0]
            cols = datashapes['svhn'][1]
            chans = datashapes['svhn'][2]
            num_imgs = img_array.shape[3]
            # Note: not the most efficent way but can monitor what is happening
            new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
            for x in range(0, num_imgs):
                # TODO reuse normalize_img here
                chans = img_array[:, :, :, x]
                # # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
                # norm_vec = (255-chans)*1.0/255.0
                new_array[x] = chans
            return new_array

        # loading Extra data
        file_path = os.path.join(self.data_dir,'extra_32x32.mat')
        file = open(file_path, 'rb')
        data = loadmat(file)
        imgs = data['X']
        X = convert_imgs_to_array(imgs)
        y = data['y']
        file.close()
        # select randomly 100000 data points
        seed = 123
        np.random.seed(seed)
        idx = np.random.randint(y.shape[0], size=100000)
        all_data = X[idx] / 255.
        all_labels = y[idx]
        # split train/test
        if opts['train_dataset_size']==-1 or opts['train_dataset_size']>all_labels.shape[0]-10000:
            tr_stop = all_labels.shape[0] - 10000
        else:
            tr_stop = opts['train_dataset_size']
        self.data_train = all_data[idx_random[:tr_stop]]
        self.labels_train = all_labels[idx_random[:tr_stop]]
        self.data_test = all_data[idx_random[-10000:]]
        self.labels_test = all_labels[idx_random[-10000:]]
        # dataset size
        self.train_size = self.data_train.shape[0]
        self.test_size = self.data_test.shape[0]
        # datashape
        self.data_shape = datashapes[self.dataset]
        # Create tf.dataset
        dataset_train = tf.compat.v1.data.Dataset.from_tensor_slices(self.data_train)
        dataset_test = tf.compat.v1.data.Dataset.from_tensor_slices(self.data_test)
        # Shuffle dataset
        dataset_train = dataset_train.shuffle(buffer_size=50*opts['batch_size'])
        dataset_test = dataset_test.shuffle(buffer_size=50*opts['batch_size'])
        # repeat for multiple epochs
        dataset_train = dataset_train.repeat()
        dataset_test = dataset_test.repeat()
        # Random batching
        dataset_train = dataset_train.batch(batch_size=opts['batch_size'])
        dataset_test = dataset_test.batch(batch_size=opts['batch_size'])
        # Prefetch
        self.dataset_train = dataset_train.prefetch(buffer_size=4*opts['batch_size'])
        self.dataset_test = dataset_test.prefetch(buffer_size=4*opts['batch_size'])
        # Iterator for each split
        self.iterator_train = dataset_train.make_initializable_iterator()
        self.iterator_test = dataset_test.make_initializable_iterator()

        # Global iterator
        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])
        self.next_element = tf.compat.v1.data.Iterator.from_string_handle(
            self.handle, dataset_train.output_types, dataset_train.output_shapes).get_next()

    def init_iterator(self, sess):
        sess.run([self.iterator_train.initializer,self.iterator_test.initializer])
        # handle = sess.run(iterator.string_handle())
        train_handle, test_handle = sess.run([self.iterator_train.string_handle(),self.iterator_test.string_handle()])

        return train_handle, test_handle

    def sample_observations(self, keys, train=False):
        # all_data is an np.ndarray already loaded into the memory
        if train:
            all_data = self.data_train
            all_labels = self.labels_train
        else:
            all_data = self.data_test
            all_labels = self.labels_test
        data = all_data[keys]
        labels = all_labels[keys]
        if self.dataset=='mnist':
            paddings = ((0,0),(2,2), (2,2), (0,0))
            data = np.pad(data, paddings, mode='constant', constant_values=0.)

        return data, labels
