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

def save_train(opts, sample_train, sample_test,
                     label_test,
                     rec_train, rec_test,
                     probs_train, probs_test,
                     encoded,
                     samples_prior,
                     samples,
                     losses, losses_match,
                     losses_rec, loss_rec_test, losses_vae,
                     kl_gau, kl_dis,
                     blurr_vals,
                     accuracies,
                     work_dir,
                     filename):

    """ Generates and saves the plot of the following layout:
        img1 | img2 | img3
        img4 | img6 | img5

        img1    -   test reconstructions
        img2    -   train reconstructions
        img3    -   samples
        img4    -   Means mixture weights
        img5    -   real pics
        img6    -   loss curves

    """
    num_pics = opts['plot_num_pics']
    num_cols = opts['plot_num_cols']
    assert num_pics % num_cols == 0
    assert num_pics % 2 == 0
    greyscale = sample_train.shape[-1] == 1

    if opts['input_normalize_sym']:
        sample_train = sample_train / 2. + 0.5
        sample_test = sample_test / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        rec_test = rec_test / 2. + 0.5
        samples = samples / 2. + 0.5

    images = []
    ### Reconstruction plots
    for pair in [(sample_train, rec_train),
                 (sample_test, rec_test)]:

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

        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - merged[idx, :, :, :])
            else:
                pics.append(merged[idx, :, :, :])

        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    ### Sample plots
    # reshaping samples & prior samples
    sample_shape = np.shape(samples)
    new_shape = (-1,) + sample_shape[2:]
    samples = np.reshape(np.transpose(samples,(1,0,2,3,4)),new_shape)
    sample_prior_shape = np.shape(samples_prior)
    new_shape = (-1,) + sample_prior_shape[2:]
    samples_prior = np.reshape(np.transpose(samples_prior,(1,0,2)),new_shape)

    for sample in [samples, sample_train]:
        assert len(sample) == num_pics
        pics = []
        for idx in range(num_pics):
            if greyscale:
                pics.append(1. - sample[idx, :, :, :])
            else:
                pics.append(sample[idx, :, :, :])
        # Figuring out a layout
        pics = np.array(pics)
        image = np.concatenate(np.split(pics, num_cols), axis=2)
        image = np.concatenate(image, axis=0)
        images.append(image)

    img1, img2, img3, img5 = images

    # Creating a pyplot fig
    dpi = 100
    height_pic = img1.shape[0]
    width_pic = img1.shape[1]
    fig_height = 4 * 2*height_pic / float(dpi)
    fig_width = 6 * 2*width_pic / float(dpi)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = matplotlib.gridspec.GridSpec(2, 3)

    # Filling in separate parts of the plot

    # First samples and reconstructions
    for img, (gi, gj, title) in zip([img1, img2, img3],
                             [(0, 0, 'Train reconstruction'),
                              (0, 1, 'Test reconstruction'),
                              (0, 2, 'Generated samples')]):
        plt.subplot(gs[gi, gj])
        if greyscale:
            image = img[:, :, 0]
            # in Greys higher values correspond to darker colors
            ax = plt.imshow(image, cmap='Greys',
                            interpolation='none', vmin=0., vmax=1.)
        else:
            ax = plt.imshow(img, interpolation='none', vmin=0., vmax=1.)

        ax = plt.subplot(gs[gi, gj])
        plt.text(0.47, 1., title,
                 ha="center", va="bottom", size=20, transform=ax.transAxes)

        # Removing ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.set_xlim([0, width_pic])
        ax.axes.set_ylim([height_pic, 0])
        ax.axes.set_aspect(1)

    ### Then the mean mixtures plots
    mean_probs = []
    for i in range(10):
        probs = [probs_test[k] for k in range(num_pics) if label_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # # entropy
    # entropies = calculate_row_entropy(mean_probs)
    # relab_mask = relabelling_mask_from_entropy(mean_probs, entropies)
    # mean_probs = mean_probs[relab_mask]
    ax = plt.subplot(gs[1, 0])
    plt.imshow(mean_probs,cmap='hot', interpolation='none', vmax=1.,vmin=0.)
    plt.text(0.47, 1., 'Test means probs',
           ha="center", va="bottom", size=20, transform=ax.transAxes)
    #plt.yticks(np.arange(10),relab_mask)
    plt.xticks(np.arange(10))

    # ###UMAP visualization of the embedings
    ax = plt.subplot(gs[1, 1])
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,samples_prior),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded,samples_prior),axis=0))

    plt.scatter(embedding[:num_pics, 0], embedding[:num_pics, 1],
                #c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='tab10'))
                c=label_test[:num_pics], s=40, label='Qz test',cmap=discrete_cmap(10, base_cmap='Vega10'))
    plt.colorbar()
    plt.scatter(embedding[num_pics:, 0], embedding[num_pics:, 1],
                            color='navy', s=10, marker='*',label='Pz')

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
    plt.legend(loc='upper left')
    plt.text(0.47, 1., 'UMAP latents', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### The loss curves
    ax = plt.subplot(gs[1, 2])
    total_num = len(losses)
    x_step = max(int(total_num / 100), 1)
    x = np.arange(1, len(losses) + 1, x_step)

    y = np.log(losses[::x_step])
    plt.plot(x, y, linewidth=3, color='black', label='loss')

    l = np.array(losses_rec)
    y = np.log(l[::x_step])
    plt.plot(x, y, linewidth=2, color='red', label='rec')

    if len(kl_gau)>0:
        y = np.log(np.abs(losses_match[::x_step]))
        plt.plot(x, y, linewidth=2, color='blue', label='|match loss|)')

        y = np.log(kl_gau[::x_step])
        plt.plot(x, y, linewidth=2, color='blue', linestyle=':', label='cont KL')

        y = np.log(kl_dis[::x_step])
        plt.plot(x, y, linewidth=2, color='blue', linestyle='--', label='disc KL')

    else:
        l = np.array(losses_match)
        y = np.log(opts['lambda']*np.abs(l[::x_step]))
        plt.plot(x, y, linewidth=2, color='blue', label='|MMD|')

        # l = np.array(losses_VAE)
        # y = np.log(l[::x_step])
        # plt.plot(x, y, linewidth=2, color='red', linestyle=(0, (5, 10)), label='VAE rec')

    plt.grid(axis='y')
    plt.legend(loc='lower left')
    plt.text(0.47, 1., 'Loss curves', ha="center", va="bottom",
                                size=20, transform=ax.transAxes)

    ### Saving plots and data
    # Plot
    plots_dir = 'train_plots'
    save_path = os.path.join(work_dir,plots_dir)
    utils.create_dir(save_path)
    fig.savefig(utils.o_gfile((save_path, filename), 'wb'),
                dpi=dpi, format='png')
    plt.close()

    # data
    data_dir = 'train_data'
    save_path = os.path.join(work_dir,data_dir)
    utils.create_dir(save_path)
    name = filename[:-4]
    if len(kl_gau)>0:
        np.savez(os.path.join(save_path,name),
                    test_data=sample_test,
                    rec_test=rec_test,
                    probs_test=probs_test,
                    probs_train=probs_train,
                    loss=np.array(losses[::x_step]),
                    loss_rec=np.array(losses_rec[::x_step]),
                    loss_rec_test=np.array(loss_rec_test),
                    loss_match=np.array(losses_match[::x_step]),
                    kl_cont=np.array(kl_gau[::x_step]),
                    kl_disc=np.array(kl_dis[::x_step]),
                    fid=np.array(blurr_vals),
                    acc=np.array(accuracies))
    else:
        np.savez(os.path.join(save_path,name),
                    test_data=sample_test,
                    rec_test=rec_test,
                    probs_test=probs_test,
                    probs_train=probs_train,
                    loss=np.array(losses[::x_step]),
                    loss_rec=np.array(losses_rec[::x_step]),
                    loss_rec_test=np.array(loss_rec_test),
                    loss_vae=np.array(losses_vae[::x_step]),
                    loss_match=np.array(np.array(losses_match[::x_step])),
                    fid=np.array(blurr_vals),
                    acc=np.array(accuracies))

def save_vizu(opts, data_train, data_test,              # images
                    label_test,                         # labels
                    rec_train, rec_test,                # reconstructions
                    enc_mw_test,                        # mixweights
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

    if opts['input_normalize_sym']:
        data_train = data_train / 2. + 0.5
        data_test = data_test / 2. + 0.5
        rec_train = rec_train / 2. + 0.5
        rec_test = rec_test / 2. + 0.5
        interpolation = interpolation / 2. + 0.5
        samples = samples / 2. + 0.5
        prior_interpolation = prior_interpolation / 2. + 0.5

    images = []

    ### Reconstruction plots
    for pair in [(data_train, rec_train),
                 (data_test, rec_test)]:
        # Arrange pics and reconstructions in a proper way
        sample, recon = pair
        num_pics = np.shape(sample)[0]
        num_cols = 20
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
    num_pics = np.shape(enc_mw_test)[0]
    for i in range(10):
        probs = [enc_mw_test[k] for k in range(num_pics) if label_test[k]==i]
        probs = np.mean(np.stack(probs,axis=0),axis=0)
        mean_probs.append(probs)
    mean_probs = np.stack(mean_probs,axis=0)
    # entropy
    #entropies = calculate_row_entropy(mean_probs)
    #cluster_to_digit = relabelling_mask_from_entropy(mean_probs, entropies)
    cluster_to_digit = relabelling_mask_from_probs(mean_probs)
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
    if opts['zdim']==2:
        embedding = np.concatenate((encoded,samples_prior),axis=0)
        #embedding = np.concatenate((encoded,enc_mean,sample_prior),axis=0)
    else:
        embedding = umap.UMAP(n_neighbors=5,
                                min_dist=0.3,
                                metric='correlation').fit_transform(np.concatenate((encoded[:num_pics],samples_prior),axis=0))
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
                enc_mw_test=enc_mw_test,
                lmbda=np.array(opts['lambda']))

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, color_list, N)
