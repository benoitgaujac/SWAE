import copy
from math import sqrt

# MNIST config from ICLR paper

config_mnist = {}
# Outputs set up
config_mnist['verbose'] = False
config_mnist['save_every_epoch'] = 1000
config_mnist['print_every'] = 30 #250
config_mnist['work_dir'] = 'results_mnist'
config_mnist['plot_num_pics'] = 120
config_mnist['plot_num_cols'] = 10

# Data set up
config_mnist['dataset'] = 'mnist'
config_mnist['input_normalize_sym'] = False
config_mnist['data_dir'] = 'mnist'
config_mnist['data_source_url'] = 'http://yann.lecun.com/exdb/mnist/'

# Experiment set up
config_mnist['train_dataset_size'] = 1500
config_mnist['batch_size'] = 150
config_mnist['epoch_num'] = 100
config_mnist['method'] = 'swae' #vae, swae
config_mnist['use_trained'] = False #train from pre-trained model
config_mnist['e_pretrain'] = False #pretrained the encoder parameters
config_mnist['e_pretrain_sample_size'] = 500

# Opt set up
config_mnist['optimizer'] = 'adam' # adam, sgd
config_mnist['adam_beta1'] = 0.5
config_mnist['lr'] = 0.0001
config_mnist['lr_adv'] = 0.0008
config_mnist['clip_grad'] = False
config_mnist['clip_norm'] = 50.
config_mnist['batch_norm'] = True
config_mnist['batch_norm_eps'] = 1e-05
config_mnist['batch_norm_decay'] = 0.9

# Objective set up
config_mnist['cost'] = 'l2sq' #l2, l2sq, l1, l2sq_wrong
config_mnist['sqrt_MMD'] = False #use true MMD estimator or square estimator
config_mnist['lambda'] = 1000.
config_mnist['mmd_kernel'] = 'RBF' # RBF, IMQ

# Model set up
config_mnist['zdim'] = 8
config_mnist['nmixtures'] = 12
config_mnist['nclasses'] = 10
config_mnist['sigma_prior'] = .1**2
coef = 2. * config_mnist['sigma_prior'] / config_mnist['zdim'] / 3.
config_mnist['prior_threshold'] = coef * 3.**2
config_mnist['pz_scale'] = 1.

# NN set up
config_mnist['conv_filters_dim'] = 3
config_mnist['init_std'] = 0.0099999
config_mnist['init_bias'] = 0.0

config_mnist['e_means'] = 'learnable'
config_mnist['init_e_std'] = -2.0

config_mnist['e_gaus_arch'] = 'mlp' # mlp, dcgan, ali, began
config_mnist['e_gaus_nlayers'] = 1
config_mnist['e_gaus_nfilters'] = 16
config_mnist['e_cat_arch'] = 'mlp' # mlp, dcgan, ali, began
config_mnist['e_cat_nlayers'] = 2
config_mnist['e_cat_nfilters'] = 32

config_mnist['g_cont_arch'] = 'mlp' # mlp, dcgan, dcgan_mod, ali, began
config_mnist['g_cont_nlayers'] = 2
config_mnist['g_cont_nfilters'] = 32
