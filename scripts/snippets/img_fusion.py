

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path_source = Path.cwd()/'results'
knee = 'fusion'  # 'fusion'
for protocol in ["lowSNR", "highSNR"]:
    path_img_1 = path_source/ 'RealExpDecayDataset'/str(knee)/protocol/'knee_core'/('box-plot_rel_'+protocol+'_tau.png')
    path_img_2 = path_source/'RealExpDecayDataset'/str(knee)/protocol/'tissue'/('box-plot_rel_'+protocol+'_tau.png')

    img = mpimg.imread(path_img_1)
    height, width = img.shape[:-1]
    hw_ratio = height/width
    wh_ratio = width/height
    ncols = 2
    nrows = 1
    figsize = (3*ncols*max(wh_ratio, 1)*(1), 3*nrows*(max(hw_ratio, 1)*(1) ))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=figsize,
                            gridspec_kw={ 'wspace':0,  'hspace':0, 'left':0, 'right':1 , 'bottom': 0, 'top':1},
                            subplot_kw={'aspect':'equal'}
                            )
    for axis in ax:
        axis.axis('off')


    ax[0].imshow(mpimg.imread(path_img_1))
    ax[1].imshow(mpimg.imread(path_img_2))


    fig.savefig(path_img_1.parents[1]/('box-plot_rel_'+protocol+'_tau.png'), dpi=300)
    fig.savefig(path_img_1.parents[1]/('box-plot_rel_'+protocol+'_tau.tiff'), dpi=300)



# ----------------------------- Distribution -------------------------------
# path_img_1 = Path.cwd()/Path('results/ExpDecayData/ensemble/box-plot_noise_tau.png')
# path_img_2 = Path.cwd()/Path('results/ExpDecayData/ensemble/box-plot_noise-rel_tau.png')

# path_img_1 = Path.cwd()/Path('results/ExpDecayData/ensemble/box-plot_snr_tau.png')
# path_img_2 = Path.cwd()/Path('results/ExpDecayData/ensemble/box-plot_snr-rel_tau.png')


# img = mpimg.imread(path_img_1)
# height, width = img.shape[:-1]
# hw_ratio = height/width
# wh_ratio = width/height
# ncols = 2
# nrows = 1
# figsize = (3*ncols*max(wh_ratio, 1)*(1), 3*nrows*(max(hw_ratio, 1)*(1) ))
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=figsize,
#                         gridspec_kw={ 'wspace':0,  'hspace':0, 'left':0, 'right':1 , 'bottom': 0, 'top':1},
#                         subplot_kw={'aspect':'equal'}
#                         )
# for axis in ax:
#     axis.axis('off')


# ax[0].imshow(mpimg.imread(path_img_1))
# ax[1].imshow(mpimg.imread(path_img_2))


# fig.savefig('dist.png', dpi=300)


for knee in [1, 2, 3, 4, 5, 6, 7]:
    for crop_name in ["knee_core", "tissue"]:
        path_img_1 = path_source / Path('RealExpDecayDataset/' + str(knee) + '/highSNR/' + crop_name + '/visual_highSNR.png')
        path_img_2 = path_source / Path('RealExpDecayDataset/' + str(knee) + '/lowSNR/' + crop_name + '/visual_lowSNR.png')

        img = mpimg.imread(path_img_1)
        height, width = img.shape[:-1]
        hw_ratio = height / width
        wh_ratio = width / height
        ncols = 1
        nrows = 2
        figsize = (3 * ncols * max(wh_ratio, 1) * (1), 3 * nrows * (max(hw_ratio, 1) * (1)))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=figsize,
                               gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'bottom': 0, 'top': 1},
                               subplot_kw={'aspect': 'auto'}
                               )
        for axis in ax:
            axis.axis('off')

        ax[0].imshow(mpimg.imread(path_img_1))
        ax[0].set_aspect('auto')
        ax[1].imshow(mpimg.imread(path_img_2))
        ax[1].set_aspect('auto')

        # fig.tight_layout()
        fig.savefig(path_img_1.parents[2] / ('visual_' + crop_name + '.png'), dpi=300)

    path_img_1 = path_source / Path('RealExpDecayDataset/' + str(knee) + '/visual_knee_core.png')
    path_img_2 = path_source / Path('RealExpDecayDataset/' + str(knee) + '/visual_tissue.png')

    img = mpimg.imread(path_img_1)
    height, width = img.shape[:-1]
    hw_ratio = height / width
    wh_ratio = width / height
    ncols = 1
    nrows = 2
    figsize = (3 * ncols * max(wh_ratio, 1) * (1), 3 * nrows * (max(hw_ratio, 1) * (1)))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=figsize,
                           gridspec_kw={'wspace': 0, 'hspace': 0, 'left': 0, 'right': 1, 'bottom': 0, 'top': 1},
                           subplot_kw={'aspect': 'auto'})
    for axis in ax:
        axis.axis('off')

    ax[0].imshow(mpimg.imread(path_img_1))
    ax[0].set_aspect('auto')
    ax[1].imshow(mpimg.imread(path_img_2))
    ax[1].set_aspect('auto')
    fig.savefig(path_img_1.parent / ('visual.png'), dpi=300)
    fig.savefig(path_img_1.parent / ('visual.tiff'), dpi=300)
