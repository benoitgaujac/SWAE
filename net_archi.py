import numpy as np
import tensorflow as tf

from ops.linear import Linear
from ops.batchnorm import Batchnorm_layers
from ops.conv2d import Conv2d
from ops.deconv2d import Deconv2D
import ops._ops


#################################### Encoder/Decoder ####################################

######### mlp #########
def mlp_encoder(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    layer_x = input
    # hidden 0
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    256, init=opts['mlp_init'],
                                    scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # gaussian output layer
    gaus_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    gaus_output_dim,
                                    init=opts['mlp_init'],
                                    scope='gaus/final')
    # hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    256, init=opts['mlp_init'],
                                    scope='cat/hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'cat/hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    256, init=opts['mlp_init'],
                                    scope='cat/hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'cat/hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    cat_output_dim,
                                    init=opts['mlp_init'],
                                    scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mlp_encoder_per_mixtures(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    layer_x = input
    ### gaussian encoder
    gaus_outputs = []
    for n in range(opts['nmixtures']):
        # hidden 0
        gaus_layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        64, init=opts['mlp_init'],
                                        scope='gaus_{}/hid/lin'.format(n))
        if opts['normalization']=='batchnorm':
            gaus_layer_x = Batchnorm_layers(opts, gaus_layer_x,
                                        'gaus_{}/hid/bn'.format(n),
                                        is_training, reuse)
        gaus_layer_x = ops._ops.non_linear(gaus_layer_x,'relu')
        gaus_output = Linear(opts, gaus_layer_x, np.prod(gaus_layer_x.get_shape().as_list()[1:]),
                                        gaus_output_dim,
                                        init=opts['mlp_init'],
                                        scope='gaus_{}/hid_final'.format(n))
        gaus_outputs.append(gaus_output)
    gaus_outputs = tf.stack(gaus_outputs,axis=1)
    ### cat encoder
    # hidden 0
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='cat/hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'cat/hid0/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='cat/hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'cat/hid1/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    cat_output_dim,
                                    init=opts['mlp_init'],
                                    scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mlp_decoder(opts, input, output_dim, reuse=False, is_training=False):
    layer_x = input
    # hidden 0
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    1024, init=opts['mlp_init'],
                                    scope='hid1/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid1/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    # layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
    #                                 512, init=opts['mlp_init'],
    #                                 scope='hid3/lin')
    # if opts['normalization']=='batchnorm':
    #     layer_x = Batchnorm_layers(opts, layer_x, 'hid3/bn',
    #                                 is_training, reuse)
    # layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Linear(opts, layer_x,np.prod(layer_x.get_shape().as_list()[1:]),
                                    np.prod(output_dim), init=opts['mlp_init'],
                                    scope='hid_final')

    return outputs

######### conv #########
def mnist_conv_encoder_v0(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=8, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=16, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # gaussian output layer
    gaus_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    gaus_output_dim,
                                    init=opts['mlp_init'],
                                    scope='gaus/final')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='cat/hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'cat/hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='cat/hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'cat/hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    cat_output_dim,
                                    init=opts['conv_init'],
                                    scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mnist_conv_encoder(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # gaus hidden
    layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    128, init=opts['mlp_init'],
                                    scope='gaus/hid/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'gaus/hid/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # gaussian output layer
    gaus_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    gaus_output_dim,
                                    init=opts['mlp_init'],
                                    scope='gaus/final')
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=16, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                    cat_output_dim,
                                    init=opts['conv_init'],
                                    scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mnist_conv_encoder_per_mix(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    layer_x = input
    ### gaussian encoder
    gaus_outputs = []
    for n in range(opts['nmixtures']):
        # hidden 0
        gaus_layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        32, init=opts['mlp_init'],
                                        scope='gaus_{}/hid/lin'.format(n))
        if opts['normalization']=='batchnorm':
            gaus_layer_x = Batchnorm_layers(opts, gaus_layer_x,
                                        'gaus_{}/hid/bn'.format(n),
                                        is_training, reuse)
        gaus_layer_x = ops._ops.non_linear(gaus_layer_x,'relu')
        gaus_output = Linear(opts, gaus_layer_x, np.prod(gaus_layer_x.get_shape().as_list()[1:]),
                                        gaus_output_dim,
                                        init=opts['mlp_init'],
                                        scope='gaus_{}/hid_final'.format(n))
        gaus_outputs.append(gaus_output)
    gaus_outputs = tf.stack(gaus_outputs,axis=1)
    ### cat encoder
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=128, filter_size=4,
                                        stride=2, scope='hid0/conv',
                                        init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn',
                                        is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=256, filter_size=4,
                                        stride=2, scope='hid1/conv',
                                        init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid1/bn',
                                        is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        cat_output_dim,
                                        init=opts['conv_init'],
                                        scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mnist_conv_encoder_per_mix_v2(opts, input, cat_output_dim, gaus_output_dim, reuse=False, is_training=False):
    layer_x = input
    ### gaussian encoder
    gaus_outputs = []
    for n in range(opts['nmixtures']):
        # hidden 0
        gaus_layer_x = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        32, init=opts['mlp_init'],
                                        scope='gaus_{}/hid/lin'.format(n))
        if opts['normalization']=='batchnorm':
            gaus_layer_x = Batchnorm_layers(opts, gaus_layer_x,
                                        'gaus_{}/hid/bn'.format(n),
                                        is_training, reuse)
        gaus_layer_x = ops._ops.non_linear(gaus_layer_x,'relu')
        gaus_output = Linear(opts, gaus_layer_x, np.prod(gaus_layer_x.get_shape().as_list()[1:]),
                                        gaus_output_dim,
                                        init=opts['mlp_init'],
                                        scope='gaus_{}/hid_final'.format(n))
        gaus_outputs.append(gaus_output)
    gaus_outputs = tf.stack(gaus_outputs,axis=1)
    ### cat encoder
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=256, filter_size=4,
                                        stride=2, scope='hid0/conv',
                                        init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn',
                                        is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                        output_dim=512, filter_size=4,
                                        stride=2, scope='hid1/conv',
                                        init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid1/bn',
                                        is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # cat output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    cat_outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                        cat_output_dim,
                                        init=opts['conv_init'],
                                        scope='cat/hid_final')

    return cat_outputs, gaus_outputs

def mnist_conv_decoder(opts, input, output_dim, reuse=False, is_training=False):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                    8*8*256, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 256])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    128]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape, filter_size=4,
                                    stride=2, scope='hid1/deconv',
                                    init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x, 'hid1/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape, filter_size=4,
                                    stride=2, scope='hid2/deconv',
                                    init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x, 'hid2/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')

    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+output_dim,
                                    filter_size=1, stride=1,
                                    scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def mnist_conv_decoder_v2(opts, input, output_dim, reuse=False, is_training=False):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                    8*8*128, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x, 'hid0/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 128])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape, filter_size=4,
                                    stride=2, scope='hid1/deconv',
                                    init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x, 'hid1/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                    2*layer_x.get_shape().as_list()[2],
                                    32]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=_out_shape, filter_size=4,
                                    stride=2, scope='hid2/deconv',
                                    init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x, 'hid2/bn',
                                    is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')

    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                    output_shape=[batch_size,]+output_dim,
                                    filter_size=1, stride=1,
                                    scope='hid_final/deconv',
                                    init= opts['conv_init'])

    return outputs

def svhn_conv_encoder(opts, input, output_dim, reuse=False, is_training=False):
    """
    Archi used by Ghosh & al.
    """
    layer_x = input
    # hidden 0
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=32, filter_size=4,
                                stride=2, scope='hid0/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 1
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=64, filter_size=4,
                                stride=2, scope='hid1/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=128, filter_size=4,
                                stride=2, scope='hid2/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 3
    layer_x = Conv2d(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_dim=256, filter_size=4,
                                stride=2, scope='hid3/conv',
                                init=opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid3/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    layer_x = tf.reshape(layer_x, [-1,np.prod(layer_x.get_shape().as_list()[1:])])
    outputs = Linear(opts, layer_x, np.prod(layer_x.get_shape().as_list()[1:]),
                                output_dim, scope='hid_final')

    return outputs

def svhn_conv_decoder(opts, input, output_dim, reuse, is_training):
    """
    Archi used by Ghosh & al.
    """
    # batch_size
    batch_size = tf.shape(input)[0]
    layer_x = input
    # Linear layers
    layer_x = Linear(opts, layer_x, np.prod(input.get_shape().as_list()[1:]),
                                8*8*128, scope='hid0/lin')
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers(opts, layer_x,
                                'hid0/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    layer_x = tf.reshape(layer_x, [-1, 8, 8, 128])
    # hidden 1
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                64]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid1/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid1/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # hidden 2
    _out_shape = [batch_size, 2*layer_x.get_shape().as_list()[1],
                                2*layer_x.get_shape().as_list()[2],
                                32]
    layer_x = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=_out_shape, filter_size=4,
                                stride=2, scope='hid2/deconv',
                                init= opts['conv_init'])
    if opts['normalization']=='batchnorm':
        layer_x = Batchnorm_layers( opts, layer_x,
                                'hid2/bn', is_training, reuse)
    layer_x = ops._ops.non_linear(layer_x,'relu')
    # output layer
    outputs = Deconv2D(opts, layer_x, layer_x.get_shape().as_list()[-1],
                                output_shape=[batch_size,]+output_dim, filter_size=1,
                                stride=1, scope='hid_final/deconv',
                                init= opts['conv_init'])

    return outputs

net_archi = {'mnist': {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
                    'mlp_per_mix': {'encoder': mlp_encoder_per_mixtures, 'decoder': mlp_decoder},
                    'conv':{'encoder': mnist_conv_encoder, 'decoder': mlp_decoder},
                    'conv_per_mix':{'encoder': mnist_conv_encoder_per_mix, 'decoder': mlp_decoder},
                    # 'conv_per_mix_conv_dec':{'encoder': mnist_conv_encoder_per_mix, 'decoder': mnist_conv_decoder_v2}},
                    'conv_per_mix_conv_dec':{'encoder': mnist_conv_encoder_per_mix_v2, 'decoder': mnist_conv_decoder_v2}},
            'svhn': {'mlp': {'encoder': mlp_encoder, 'decoder': mlp_decoder},
                    'conv': {'encoder': svhn_conv_encoder, 'decoder': svhn_conv_decoder}}
            }
