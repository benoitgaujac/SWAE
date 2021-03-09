import tensorflow as tf
from net_archi import net_archi

import pdb

def encoder(opts, input, cat_output_dim, gaus_output_dim, scope=None, reuse=False, is_training=False):

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # build encoder network
        if opts['dataset'] in net_archi:
            if opts['net_archi'] in net_archi[opts['dataset']]:
                encoder_net = net_archi[opts['dataset']][opts['net_archi']]['encoder']
            else:
                ValueError('Unknown {} net. archi. for {} dataset'.format(
                                    opts['net_archi'], opts['dataset']))
        else:
            ValueError('{} dataset'.format(opts['dataset']))
        # encode
        cat_outputs, gaus_outputs = encoder_net(opts, input, cat_output_dim,
                                    gaus_output_dim, reuse, is_training)
    mean, logSigma = tf.split(gaus_outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500) #[batch,nmixtures*zdim]
    Sigma = tf.nn.softplus(logSigma) #[batch,nmixtures*zdim]

    return cat_outputs, mean, Sigma


def decoder(opts, input, output_dim, scope=None, reuse=False, is_training=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # build decoder network
        if opts['dataset'] in net_archi:
            if opts['net_archi'] in net_archi[opts['dataset']]:
                decoder_net = net_archi[opts['dataset']][opts['net_archi']]['decoder']
            else:
                ValueError('Unknown {} net. archi. for {} dataset'.format(
                                    opts['net_archi'], opts['dataset']))
        # decode
        outputs = decoder_net(opts, input, output_dim, reuse, is_training) #[batch,nmixtures,2*outdim]

    mean, logSigma = tf.split(outputs,2,axis=-1)
    logSigma = tf.clip_by_value(logSigma, -20, 500)
    Sigma = tf.nn.softplus(logSigma)

    return mean, Sigma
