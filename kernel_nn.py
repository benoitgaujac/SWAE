import numpy as np
import tensorflow as tf
import ops
from datahandler import datashapes
from math import ceil

def k_encoder(opts, inputs, reuse=False, is_training=False):
    with tf.variable_scope("kernel_encoder", reuse=reuse):
        if opts['k_e_arch'] == 'mlp':
            k_encoded = mlp_encoder(opts['k_e_num_filters'],
                                    opts['k_e_num_layers'],
                                    opts['k_outdim'], inputs,
                                    opts, is_training, reuse)
        elif opts['k_e_arch'] == 'dcgan':
            # Fully convolutional architecture similar to DCGAN
            k_encoded = dcgan_encoder(opts['k_e_num_filters'],
                                        opts['k_e_num_layers'],
                                        opts['k_outdim'], inputs,
                                        opts, is_training, reuse)
        elif opts['k_e_arch'] == 'began':
            # Architecture similar to the BEGAN paper
            k_encoded = began_encoder(opts['k_e_num_filters'],
                                        opts['k_e_num_layers'],
                                        opts['k_outdim'], inputs,
                                        opts, is_training, reuse)
        else:
            raise ValueError('%s Unknown encoder architecture for mixtures' % opts['k_e_arch'])

    return k_encoded


def mlp_encoder(num_units, num_layers, output_dim, inputs, opts, is_training=False, reuse=False):
    layer_x = inputs
    for i in range(num_layers):
        layer_x = ops.linear(opts, layer_x, num_units, scope='h{}_lin'.format(i))
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                reuse, scope='h{}_bn'.format(i))
        layer_x = tf.nn.relu(layer_x)
    layer_x = ops.linear(opts, layer_x, output_dim, scope='out_lin')

    return layer_x

def dcgan_encoder(num_units, num_layers, output_dim, inputs, opts, is_training=False, reuse=False):
    layer_x = inputs
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = ops.conv2d(opts, layer_x, int(num_units / scale),
                             scope='h{}_conv'.format(i))
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                     reuse, scope='h{}_bn'.format(i))
        layer_x = tf.nn.relu(layer_x)
    layer_x = ops.linear(opts, layer_x, output_dim, scope='out_lin')

    return layer_x

def began_encoder(num_units, num_layers, output_dim, inputs, opts, is_training=False, reuse=False):
    layer_x = inputs
    layer_x = ops.conv2d(opts, layer_x, num_units, scope='hfirst_conv')
    for i in range(num_layers):
        if i % 3 < 2:
            if i != num_layers - 2:
                ii = i - int(i / 3)
                scale = (ii + 1 - int(ii / 2))
            else:
                ii = i - int(i / 3)
                scale = (ii - int((ii - 1) / 2))
            layer_x = ops.conv2d(opts, layer_x, num_units * scale, d_h=1, d_w=1,
                                 scope='_h{}_conv'.format(i))
            layer_x = tf.nn.relu(layer_x)
        else:
            if i != num_layers - 1:
                layer_x = ops.downsample(layer_x, scope='h{}_maxpool'.format(i),
                                                                    reuse=reuse)
    # Tensor should be [N, 8, 8, filters] at this point
    layer_x = ops.linear(opts, layer_x, output_dim, scope='out_lin')

    return layer_x


def k_decoder(opts, noise, output_dim, reuse=False, is_training=True):
    num_units = opts['k_g_num_filters']

    with tf.variable_scope("kernel_generator", reuse=reuse):
        if opts['k_g_arch'] == 'mlp':
            # Architecture with only fully connected layers and ReLUs
            layer_x = noise
            for i in range(opts['k_g_num_layers']):
                layer_x = ops.linear(opts, layer_x, num_units, 'h%d_lin' % i)
                layer_x = tf.nn.relu(layer_x)
                if opts['batch_norm']:
                    layer_x = ops.batch_norm(
                        opts, layer_x, is_training, reuse, scope='h%d_bn' % i)
            out = ops.linear(opts, layer_x,
                             output_dim, 'h%d_lin' % (i + 1))
            if opts['input_normalize_sym']:
                return tf.nn.tanh(out)
            else:
                return tf.nn.sigmoid(out)
        elif opts['k_g_arch'] in ['dcgan', 'dcgan_mod']:
            # Fully convolutional architecture similar to DCGAN
            res = dcgan_decoder(opts, noise, is_training, reuse)
        elif opts['k_g_arch'] == 'ali':
            # Architecture smilar to "Adversarially learned inference" paper
            res = ali_decoder(opts, noise, is_training, reuse)
        elif opts['k_g_arch'] == 'began':
            # Architecture similar to the BEGAN paper
            res = began_decoder(opts, noise, is_training, reuse)
        else:
            raise ValueError('%s Unknown decoder architecture' % opts['g_arch'])

        return res

def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    output_shape = datashapes[opts['dataset']]
    num_units = opts['k_g_num_filters']
    batch_size = tf.shape(noise)[0]
    num_layers = opts['k_g_num_layers']
    if opts['k_g_arch'] == 'dcgan':
        height = output_shape[0] / 2**num_layers
        width = output_shape[1] / 2**num_layers
    elif opts['k_g_arch'] == 'dcgan_mod':
        height = output_shape[0] / 2**(num_layers - 1)
        width = output_shape[1] / 2**(num_layers - 1)

    h0 = ops.linear(opts, noise, num_units * ceil(height) * ceil(width),
                                            scope='h0_lin')
    h0 = tf.reshape(h0, [-1, ceil(height), ceil(width), num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, ceil(height * scale),
                      ceil(width * scale), int(num_units / scale)]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    if opts['k_g_arch'] == 'dcgan':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, scope='hfinal_deconv')
    elif opts['k_g_arch'] == 'dcgan_mod':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hfinal_deconv')
    if opts['input_normalize_sym']:
        return tf.nn.tanh(last_h), last_h
    else:
        return tf.nn.sigmoid(last_h), last_h

def began_decoder(opts, noise, is_training=False, reuse=False):

    output_shape = datashapes[opts['dataset']]
    num_units = opts['k_g_num_filters']
    num_layers = opts['k_g_num_layers']
    batch_size = tf.shape(noise)[0]

    h0 = ops.linear(opts, noise, num_units * 8 * 8, scope='h0_lin')
    h0 = tf.reshape(h0, [-1, 8, 8, num_units])
    layer_x = h0
    for i in range(num_layers):
        if i % 3 < 2:
            # Don't change resolution
            layer_x = ops.conv2d(opts, layer_x, num_units,
                                 d_h=1, d_w=1, scope='h%d_conv' % i)
            layer_x = tf.nn.relu(layer_x)
        else:
            if i != num_layers - 1:
                # Upsampling by factor of 2 with NN
                scale = 2 ** (int(i / 3) + 1)
                layer_x = ops.upsample_nn(layer_x, [scale * 8, scale * 8],
                                          scope='h%d_upsample' % i, reuse=reuse)
                # Skip connection
                append = ops.upsample_nn(h0, [scale * 8, scale * 8],
                                          scope='h%d_skipup' % i, reuse=reuse)
                layer_x = tf.concat([layer_x, append], axis=3)

    last_h = ops.conv2d(opts, layer_x, output_shape[-1],
                        d_h=1, d_w=1, scope='hfinal_conv')
    if opts['input_normalize_sym']:
        return tf.nn.tanh(last_h), last_h
    else:
        return tf.nn.sigmoid(last_h), last_h
