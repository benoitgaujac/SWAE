import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import umap

import utils
from supervised_functions import relabelling_mask_from_probs

import pdb

def save_train(opts, data_train, data_test,
                     rec_train, rec_test, samples,
                     encoded, label_test,
                     samples_prior, prior_means,
                     latent_interpolation,
                     mean_probs,
                     losses, losses_test,
                     acc, acc_test,
                     work_dir, filename):


    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img5 | img6
        img4 | img5 | img6

        img1    -   train reconstructions
        img2    -   test reconstructions
        img3    -   samples
        img4    -   loss curves
        img5    -   rec/reg curbes
        img6    -   acc curves
        img7    -   embeddings
        img8    -   mean probs
        img9    -   latent space traversal

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = data_train.shape[-1] == 1

    images = []
    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test)]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        assert len(sample) == num_pics
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        # Figuring out a layout
        image = np.split(merged[:num_pics], num_cols)
        image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=2)
        image = np.split(image,image.shape[0],axis=0)
        image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=1)
        image = image[0]
        if greyscale:
            image = 1. - image
        images.append(image)
    ### Sample plots
    samples_shape = samples.shape
    samples = samples.reshape((-1,)+samples_shape[-3:])
    assert len(samples) == num_pics
    # Figuring out a layout
    image = np.split(samples, num_cols)
    image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=2)
    image = np.split(image,image.shape[0],axis=0)
    image = [np.pad(img, ((0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
    image = np.concatenate(image, axis=1)
    image = image[0]
    if greyscale:
        image = 1. - image
    images.append(image)

    img1, img2, img3 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 10*height_pic / float(dpi)
    fig_width = 10*width_pic / float(dpi)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(fig_width, fig_height))

    # First samples and reconstructions
    for img, (i, j, title) in zip([img1, img2, img3],
                             [(0, 0, 'Train rec'),
                              (0, 1, 'Test rec'),
                              (0, 2, 'Samples')]):
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            axes[i,j].imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            axes[i,j] = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing ticks
        axes[i,j].axes.get_xaxis().set_ticks([])
        axes[i,j].axes.get_yaxis().set_ticks([])
        axes[i,j].axes.set_xlim([0, width_pic])
        axes[i,j].axes.set_ylim([height_pic, 0])
        axes[i,j].axes.set_aspect(1)
        axes[i,j].set_title(title)

    ### The loss curves
    total_num = len(losses)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(losses) + 1, x_step)
    losses = np.array(losses)
    losses_test = np.array(losses_test)
    # obj
    y = np.log(losses_test[::x_step,0])
    axes[1,0].plot(x, y, linewidth=3, color='black', label='teLoss')
    y = np.log(losses[::x_step,0])
    axes[1,0].plot(x, y, linewidth=1, color='black', linestyle=':', label='trLoss')
    axes[1,0].grid(axis='y')
    axes[1,0].legend(loc='best')
    axes[1,0].set_title('Loss')
    # rec/reg
    y = np.log(losses_test[::x_step,1])
    axes[1,1].plot(x, y, linewidth=3, color='blue', label=r'$\beta$ teRec')
    y = np.log(losses[::x_step,1])
    axes[1,1].plot(x, y, linewidth=1, color='blue', linestyle=':', label=r'$\beta$ trRec')
    y = np.log(opts['beta']*losses_test[::x_step,2])
    axes[1,1].plot(x, y, linewidth=3, color='red', label=r'$\beta$ teReg')
    y = np.log(opts['beta']*losses[::x_step,1])
    axes[1,1].plot(x, y, linewidth=1, color='red', linestyle=':', label=r'$\beta$ trReg')
    axes[1,1].grid(axis='y')
    axes[1,1].legend(loc='best')
    axes[1,1].set_title('Rec/Reg split')
    # acc
    y = acc_test[::x_step]
    axes[1,2].plot(x, y, linewidth=3, color='black', label='teAcc')
    y = acc[::x_step]
    axes[1,2].plot(x, y, linewidth=1, color='black', linestyle=':', label='trAcc')
    axes[1,2].grid(axis='y')
    axes[1,2].legend(loc='best')
    axes[1,2].set_title('Accuracy')

    ### UMAP visualization of the embedings
    samples_prior = samples_prior.reshape([-1, opts['zdim']])
    num_pz = samples_prior.shape[0]
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,samples_prior,prior_means),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded,samples_prior),axis=0))

    axes[2,0].scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
                c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='tab10'))
                #c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    # axes[2,0].colorbar()
    axes[2,0].scatter(embedding[num_pics:num_pics+num_pz, 0], embedding[num_pics:num_pics+num_pz, 1],
                            color='navy', s=10, marker='*',label='Pz')
    axes[2,0].scatter(embedding[num_pics+num_pz:, 0], embedding[num_pics+num_pz:, 1],
                            color='red', s=15, marker='*',label='Pz means')

    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.1
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify

    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    axes[2,0].set_xlim(xmin, xmax)
    axes[2,0].set_ylim(ymin, ymax)
    axes[2,0].legend(loc='upper left')
    axes[2,0].set_title('Latent vizu')

    ### Latent space interpolation
    if latent_interpolation is not None:
        num_cols = latent_interpolation.shape[0]
        num_rows = latent_interpolation.shape[1]
        # Figuring out a layout, latent_interpolation is shape [nsteps, nsteps, imdim]
        image = np.split(latent_interpolation, num_cols,axis=1)
        image = [np.pad(img, ((0,0),(0,0),(0,0),(0,1),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=3)
        image = np.split(image,num_rows,axis=0)
        image = [np.pad(img, ((0,0),(0,0),(0,1),(0,0),(0,0)), mode='constant', constant_values=1.0) for img in image]
        image = np.concatenate(image, axis=2)
        image = image[0,0]
        if greyscale:
            image = 1. - image
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            axes[2,1].imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            axes[2,1] = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing ticks
        axes[2,1].get_xaxis().set_ticks([])
        axes[2,1].get_yaxis().set_ticks([])
        axes[2,1].set_xlim([0, image.shape[0]])
        axes[2,1].set_ylim([image.shape[1], 0])
        axes[2,1].set_aspect(1)
        axes[2,1].set_title('Latent manifold')

    ### Then the mean mixtures plots
    cluster_to_digit = relabelling_mask_from_probs(opts,mean_probs)
    digit_to_cluster = np.argsort(cluster_to_digit)
    mean_probs = mean_probs[:,digit_to_cluster]
    axes[2,2].imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    # plt.text(0.47, 1., 'Test means probs',
    #        ha="center", va="bottom", size=20, transform=ax.transAxes)
    axes[2,2].set_xticks(np.arange(10))
    axes[2,2].set_xticklabels(digit_to_cluster)
    axes[2,2].set_yticks(np.arange(10))
    axes[2,2].set_yticklabels(np.arange(10))
    axes[2,2].set_title('$\mathrm{\mathbb{E}}_x q_D(k\vert{X})$')

    ### Saving plots and data
    # Plot
    plots_dir = 'train_plots'
    save_path = os.path.join(work_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()


def save_vizu(opts, data_train, data_test,              # images
                    label_test,                         # labels
                    rec_train, rec_test,                # reconstructions
                    pi,                                 # mixweights
                    encoded,                            # encoded points
                    samples_prior,                      # prior samples
                    samples,                            # samples
                    interpolation, prior_interpolation, # interpolations
                    work_dir):                          # working directory

    """ Generates and saves the following plots:
        img1    -   train reconstruction
        img2    -   test reconstruction
        img3    -   samples
        img4    -   test interpolation
        img5    -   prior interpolation
        img6    -   discrete latents
        img7    -   UMAP
    """

    # Create saving directory
    plots_dir = 'vizu_plots'
    save_path = os.path.join(work_dir,plots_dir)
    utils.create_dir(save_path)

    greyscale = np.shape(prior_interpolation)[-1] == 1
    images = []
    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test)]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        num_pics = np.shape(sample)[0]
        size_pics = np.shape(sample)[1]
        num_cols = 10
        num_to_keep = 10
        assert len(sample) == len(recon)
        pics = []
        merged = np.vstack([recon, sample])
        r_ptr = 0
        w_ptr = 0
        for _ in range(int(num_pics / 2)):
            merged[w_ptr] = sample[r_ptr]
            merged[w_ptr + 1] = recon[r_ptr]
            r_ptr += 1
            w_ptr += 2
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        image = image[:num_to_keep*size_pics]
        images.append(image)

    ### Points Interpolation plots
    white_pix = 4
    num_pics = np.shape(interpolation)[0]
    num_cols = np.shape(interpolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            white = np.zeros((white_pix,)+np.shape(pic)[2:])
            pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            white = np.zeros((white_pix,)+np.shape(pic)[1:])
            pic = np.concatenate(white,pic)
            pics.append(pic)
    image = np.concatenate(pics, axis=0)
    images.append(image)

    ###Prior Interpolation plots
    white_pix = 4
    num_pics = np.shape(prior_interpolation)[0]
    num_cols = np.shape(prior_interpolation)[1]
    pics = []
    for idx in range(num_pics):
        if greyscale:
            pic = 1. - prior_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=2)
            if opts['zdim']!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[2:])
                pic = np.concatenate((white,pic[0]),axis=0)
            pics.append(pic)
        else:
            pic = prior_interpolation[idx, :, :, :, :]
            pic = np.concatenate(np.split(pic, num_cols),axis=1)
            if opts['zdim']!=2:
                white = np.zeros((white_pix,)+np.shape(pic)[1:])
                pic = np.concatenate(white,pic)
            pics.append(pic)
    # Figuring out a layout
    image = np.concatenate(pics, axis=0)
    images.append(image)

    img1, img2, img3, img4 = images

    ###Settings for pyplot fig
    dpi = 100
    for img, title, filename in zip([img1, img2, img3, img4],
                         ['Train reconstruction',
                         'Test reconstruction',
                         'Points interpolation',
                         'Priors interpolation'],
                         ['train_recon',
                         'test_recon',
                         'point_inter',
                         'prior_inter']):
        height_pic = img.shape[0]
        width_pic = img.shape[1]
        fig_height = height_pic / 10
        fig_width = width_pic / 10
        fig = plt.figure(figsize=(fig_width, fig_height))
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
        # Removing axes, ticks, labels
        plt.axis('off')
        # # placing subplot
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
        # Saving
        filename = filename + '.png'
        plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                    dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
        plt.close()

    #Set size for following plots
    height_pic= img1.shape[0]
    width_pic = img1.shape[1]

    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)

    ###The mean mixtures plots
    mean_probs = []
    num_pics = np.shape(pi)[0]
    for i in range(10):
        probs = [pi[k] for k in range(num_pics) if label_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    #entropies = calculate_row_entropy(mean_probs)
    #cluster_to_digit = relabelling_mask_from_entropy(mean_probs, entropies)
    cluster_to_digit = relabelling_mask_from_probs(opts,mean_probs)
    digit_to_cluster = np.argsort(cluster_to_digit)
    mean_probs = mean_probs[::-1,digit_to_cluster]
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.title('Average probs')
    plt.yticks(np.arange(10),np.arange(10)[::-1])
    plt.xticks(np.arange(10))
    # Saving
    filename = 'probs.png'
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png', bbox_inches='tight')
    plt.close()

    ###Sample plots
    pics = []
    num_cols = 10
    samples = np.transpose(samples,(1,0,2,3,4))
    samples = samples.reshape((-1,)+np.shape(samples)[2:])
    num_pics = np.shape(samples)[0]
    size_pics = np.shape(samples)[1]
    num_to_keep = 10
    for idx in range(num_pics):
        if greyscale:
            pics.append(1. - samples[idx, :, :, :])
        else:
            pics.append(samples[idx, :, :, :])
    # Figuring out a layout
    pics = np.array(pics)
    cluster_pics = np.array(np.split(pics, num_cols))[digit_to_cluster]
    img = np.concatenate(cluster_pics.tolist(), axis=2)
    img = np.concatenate(img, axis=0)
    img = img[:num_to_keep*size_pics]
    fig = plt.figure(figsize=(img.shape[1]/10, img.shape[0]/10))
    #fig = plt.figure()
    if greyscale:
        image = img[:, :, 0]
        # in Greys higher values correspond to darker colors
        plt.imshow(image, cmap='Greys',
                        interpolation='none', vmin=0., vmax=1.)
    else:
        plt.imshow(img, interpolation='none', vmin=0., vmax=1.)
    # Removing axes, ticks, labels
    plt.axis('off')
    # # placing subplot
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    # Saving
    filename = 'samples.png'
    plt.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png', box_inches='tight', pad_inches=0.0)
    plt.close()

    ###UMAP visualization of the embedings
    pdb.set_trace()
    samples_prior_flat = samples_prior.reshape(-1,np.shape(samples_prior)[-1])
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,samples_prior_flat),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],samples_prior_flat),axis=0))
                                #metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],enc_mean[:num_pics],sample_prior),axis=0))
    fig_height = height_pic / float(dpi)
    fig_width = width_pic / float(dpi)
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
               c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    plt.colorbar()
    plt.scatter(embedding[num_pics:, 0], embedding[num_pics:, 1],
                            color='navy', s=3, alpha=0.5, marker='*',label='Pz')
    # plt.scatter(embedding[num_pics:(2*num_pics-1), 0], embedding[num_pics:(2*num_pics-1), 1],
    #            color='aqua', s=3, alpha=0.5, marker='x',label='mean Qz test')
    # plt.scatter(embedding[2*num_pics:, 0], embedding[2*num_pics:, 1],
    #                         color='navy', s=3, alpha=0.5, marker='*',label='Pz')
    xmin = np.amin(embedding[:,0])
    xmax = np.amax(embedding[:,0])
    magnify = 0.1
    width = abs(xmax - xmin)
    xmin = xmin - width * magnify
    xmax = xmax + width * magnify
    ymin = np.amin(embedding[:,1])
    ymax = np.amax(embedding[:,1])
    width = abs(ymin - ymax)
    ymin = ymin - width * magnify
    ymax = ymax + width * magnify
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off',
                    right='off',
                    left='off',
                    labelleft='off')
    plt.legend(loc='upper left')
    plt.title('UMAP latents')
    # Saving
    filename = 'umap.png'
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()

    ###Saving data
    data_dir = 'vizu_data'
    save_path = os.path.join(work_dir,data_dir)
    utils.create_dir(save_path)
    filename = 'final_plots'
    np.savez(os.path.join(save_path,filename),
                data_train=data_train,
                data_test=data_test,
                labels_test=label_test,
                smples_pr=samples_prior,
                smples=samples,
                rec_tr=rec_train,
                rec_te=rec_test,
                enc=encoded,
                points=interpolation,
                priors=prior_interpolation,
                pi=pi,
                lmbda=np.array(opts['beta']))

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
