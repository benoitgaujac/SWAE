import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils

import tensorflow as tf

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='vizu',
                    help='mode to run [train/test/vizu]')
parser.add_argument("--exp", default='mnist',
                    help='dataset [mnist/celebA/dsprites]')
parser.add_argument("--alg",
                    help='algo to train [swae/vae]')
parser.add_argument("--work_dir")
parser.add_argument("--weights_file")


parser.add_argument("--zdim",
                    help='dimensionality of the latent space',
                    type=int)
parser.add_argument("--z_test",
                    help='method of choice for verifying Pz=Qz [mmd/gan]')
parser.add_argument("--wae_lambda", help='WAE regularizer', type=int)
parser.add_argument("--stop_grad", help='Stop gradient for debug')
parser.add_argument("--lambda_schedule",
                    help='constant or adaptive')
parser.add_argument("--enc_noise",
                    help="type of encoder noise:"\
                         " 'deterministic': no noise whatsoever,"\
                         " 'gaussian': gaussian encoder,"\
                         " 'implicit': implicit encoder,"\
                         " 'add_noise': add noise before feeding "\
                         "to deterministic encoder")

FLAGS = parser.parse_args()

def main():

    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment configuration'

    if FLAGS.alg:
        opts['method'] = FLAGS.alg
    if FLAGS.zdim:
        opts['zdim'] = FLAGS.zdim
    if FLAGS.z_test:
        opts['penalty'] = FLAGS.z_test
    if FLAGS.lambda_schedule:
        opts['lambda_schedule'] = FLAGS.lambda_schedule
    if FLAGS.work_dir:
        opts['work_dir'] = FLAGS.work_dir
    if FLAGS.wae_lambda:
        opts['lambda'] = FLAGS.wae_lambda
    if FLAGS.enc_noise:
        opts['e_noise'] = FLAGS.enc_noise

    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['method'])
    work_dir = os.path.join(opts['method'],opts['work_dir'])
    utils.create_dir(work_dir)
    utils.create_dir(os.path.join(work_dir,
                     'checkpoints'))
    # Dumping all the configs to the text file
    with utils.o_gfile((work_dir, 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    #Reset tf graph
    tf.reset_default_graph()

    # build WAE
    wae = WAE(opts)

    # Training/testing/vizu
    if FLAGS.mode=="train":
        logging.error('Training')
        wae.train(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="test":
        logging.error('Testing')
        wae.test(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="reg":
        logging.error('Logistic regression')
        wae.reg(data, opts['work_dir'], FLAGS.weights_file)
    elif FLAGS.mode=="vizu":
        logging.error('Visualization')
        wae.vizu(data, opts['work_dir'], FLAGS.weights_file)

main()
