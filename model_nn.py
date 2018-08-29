import numpy as np
import tensorflow as tf
import ops
from datahandler import datashapes
from math import ceil

import pdb


def cat_encoder(opts, inputs, reuse=False, is_training=False):
    with tf.variable_scope("encoder", reuse=reuse):
        pi = encoder(opts, inputs, opts['e_cat_arch'], opts['e_cat_nlayers'],
                                                       opts['e_cat_nfilters'],
                                                       opts['nmixtures'],
                                                       1,
                                                       'cat_params',
                                                       opts['batch_norm'],
                                                       reuse,
                                                       is_training)
        return pi[0]

def gaussian_encoder(opts, inputs, reuse=False, is_training=False):
    with tf.variable_scope('encoder', reuse=reuse):
        mean_params = encoder(opts, inputs, opts['e_gaus_arch'], opts['e_gaus_nlayers'],
                                                                 opts['e_gaus_nfilters'],
                                                                 2*opts['zdim'],
                                                                 opts['nmixtures'],
                                                                 'gaus_params',
                                                                 opts['batch_norm'],
                                                                 reuse,
                                                                 is_training)
        mean_params = tf.stack(mean_params,axis=1)
        mean, logSigma = tf.split(mean_params,2,axis=-1)
        logSigma = tf.clip_by_value(logSigma, -50, 50)
        return mean, logSigma

def encoder(opts, inputs, archi, num_layers, num_units, output_dim,
                                                        num_mixtures,
                                                        scope,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs = mlp_encoder(opts, inputs, num_layers, num_units,
                                                            output_dim,
                                                            num_mixtures,
                                                            batch_norm,
                                                            reuse,
                                                            is_training)
        elif archi == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            outputs = dcgan_encoder(opts, inputs, num_layers, num_units,
                                                              output_dim,
                                                              num_mixtures,
                                                              batch_norm,
                                                              reuse,
                                                              is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % archi)
    return outputs

def mlp_encoder(opts, inputs, num_layers, num_units, output_dim,
                                                     num_mixtures,
                                                     batch_norm=False,
                                                     reuse=False,
                                                     is_training=False):
    outputs = []
    for k in range(num_mixtures):
        layer_x = inputs
        for i in range(num_layers):
            layer_x = ops.linear(opts, layer_x, num_units, scope='mix{}/hid{}/lin'.format(k,i))
            if batch_norm:
                layer_x = ops.batch_norm(opts, layer_x, is_training,
                                    reuse, scope='mix{}/hid{}/bn'.format(k,i))
            layer_x = tf.nn.relu(layer_x)
        output = ops.linear(opts, layer_x, output_dim, scope='mix{}/hid_final'.format(k))
        outputs.append(output)
    return outputs

def dcgan_encoder(opts, inputs, num_layers, num_units, output_dim,
                                                       num_mixtures,
                                                       batch_norm=False,
                                                       reuse=False,
                                                       is_training=False):
    outputs = []
    for k in range(num_mixtures):
        layer_x = inputs
        for i in range(num_layers):
            scale = 2**(num_layers - i - 1)
            layer_x = ops.conv2d(opts, layer_x, int(num_units / scale),
                                 scope='mix{}/hid{}/conv'.format(k,i))
            if batch_norm:
                layer_x = ops.batch_norm(opts, layer_x, is_training,
                                         reuse, scope='mix{}/hid{}/bn'.format(k,i))
            layer_x = tf.nn.relu(layer_x)
        output = ops.linear(opts, layer_x, output_dim, scope='mix{}/hid_final'.format(k))
        outputs.append(output)
    return outputs

def continuous_decoder(opts, noise, reuse=False, is_training=True):
    assert opts['dataset'] in datashapes, 'Unknown dataset!'
    with tf.variable_scope("generator", reuse=reuse):
        outputs, logits = decoder(opts, noise, opts['g_cont_arch'],
                                               opts['g_cont_nlayers'],
                                               opts['g_cont_nfilters'],
                                               datashapes[opts['dataset']],
                                               'cont_gen',
                                               opts['batch_norm'],
                                               reuse,
                                               is_training)
    return outputs, logits

def decoder(opts, inputs, archi, num_layers, num_units, output_shape,
                                                        scope,
                                                        batch_norm=False,
                                                        reuse=False,
                                                        is_training=False):
    with tf.variable_scope(scope, reuse=reuse):
        if archi == 'mlp':
            # Encoder uses only fully connected layers with ReLus
            outputs, logits = mlp_decoder(opts, inputs, num_layers, num_units,
                                                                    output_shape,
                                                                    batch_norm,
                                                                    reuse,
                                                                    is_training)
        elif archi == 'dcgan' or archi == 'dcgan_mod':
            # Fully convolutional architecture similar to DCGAN
            outputs, logits = dcgan_decoder(opts, inputs, archi, num_layers,
                                                                 num_units,
                                                                 output_shape,
                                                                 batch_norm,
                                                                 reuse,
                                                                 is_training)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % archi)
    return outputs, logits


def mlp_decoder(opts, inputs, num_layers, num_units, output_shape,
                                                     batch_norm,
                                                     reuse,
                                                     is_training):
    # Architecture with only fully connected layers and ReLUs
    layer_x = inputs
    for i in range(num_layers):
        layer_x = ops.linear(opts, layer_x, num_units, 'hid%d/lin' % i)
        layer_x = tf.nn.relu(layer_x)
        if batch_norm:
            layer_x = ops.batch_norm(
                opts, layer_x, is_training, reuse, scope='hid%d/bn' % i)
    out = ops.linear(opts, layer_x,
                     np.prod(output_shape), 'hid_final')
    out = tf.reshape(out, [-1] + list(output_shape))
    if opts['input_normalize_sym']:
        return tf.nn.tanh(out), out
    else:
        return tf.nn.sigmoid(out), out

def  dcgan_decoder(opts, inputs, archi, num_layers, num_units,
                                                    output_shape,
                                                    batch_norm,
                                                    reuse,
                                                    is_training):
    batch_size = tf.shape(inputs)[0]
    if archi == 'dcgan':
        height = output_shape[0] / 2**num_layers
        width = output_shape[1] / 2**num_layers
    elif archi == 'dcgan_mod':
        height = output_shape[0] / 2**(num_layers - 1)
        width = output_shape[1] / 2**(num_layers - 1)

    h0 = ops.linear(opts, inputs, num_units * ceil(height) * ceil(width),
                                            scope='hid0/lin')
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='hid%d/deconv' % i)
        if batch_norm:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='hid%d/bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    if archi == 'dcgan':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, scope='hid_final/deconv')
    elif archi == 'dcgan_mod':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hid_final/deconv')
    if opts['input_normalize_sym']:
        return tf.nn.tanh(last_h), last_h
    else:
        return tf.nn.sigmoid(last_h), last_h
