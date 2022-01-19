

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from fitting.data import RealExpDecayDataset
import SimpleITK as sitk


def draw_patches(axis, focus):
    y, height = focus[0].start, focus[0].stop - focus[0].start
    x, width = focus[1].start, focus[1].stop - focus[1].start
    # rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='b', facecolor='none')  # xy, width, height
    # axis.add_patch(rect)


focuses = {1: (slice(0, 170), slice(180, 350)),  # y,y+height , x, x+width,
           2: (slice(50, 220), slice(165, 335)),
           3: (slice(50, 220), slice(165, 335)),
           4: (slice(70, 240), slice(165, 335)),
           5: (slice(50, 220), slice(165, 335)),
           6: (slice(30, 220), slice(120, 310)),
           7: (slice(0, 220), slice(130, 350)),
           }

clips = {1: (5, 70),
         2: (5, 110),
         3: (5, 110),
         4: (5, 110),
         5: (5, 110),
         6: (5, 110),
         7: (5, 110),
         }


for knee in [1, 2, 3, 4, 5, 6, 7]:
    for crop_name in ["tissue", "knee_core"]:
        for protocol in ["highSNR", "lowSNR"]:  # ,
            # crop_name = "tissue" # "knee_core", tissue
            # protocol =  "lowSNR"  # lowSNR, highSNR

            use_focus = crop_name == "tissue"
            focus = focuses[knee] if use_focus else slice(None)
            vmin, vmax = clips[knee]
            set_title = protocol == "highSNR"  # crop_name == "knee_core"
            para_n = 1  # tau
            fontdict = {'fontsize': 12, 'fontweight': 'bold'}

            path_run = Path.cwd() / 'runs/2021_03_10_202304'
            ds = RealExpDecayDataset(knee=knee, protocol=protocol, crop=None, noise_est_method='background')
            crop_mask = ds.get_crop(crop_name, ds.path_root)[:, :, 0]
            crop_mask_knee = ds.get_crop("knee_core", ds.path_root)[:, :, 0]

            path_out = Path.cwd() / 'results'
            result_subdir = ds.__class__.__name__.split('Module')[0]
            path_out = path_out / result_subdir / str(ds.knee) / ds.protocol / crop_name
            path_out.mkdir(parents=True, exist_ok=True)

            height, width = ds.image[:, :, 0].shape
            hw_ratio = height / width
            wh_ratio = width / height
            ncols = 5
            nrows = 1
            top_offset = 0.1 if set_title else 0.01
            bottom_offset = 0  # 0.005
            left_offset = 0.015
            right_offset = 0.04
            figsize = (3 * ncols * max(wh_ratio, 1) / (1 - (left_offset + right_offset)),
                       3 * nrows * max(hw_ratio, 1) / (1 - (top_offset + bottom_offset)))
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, squeeze=True, figsize=figsize, gridspec_kw={
                    'wspace': 0, 'hspace': 0, 'left': 0 + left_offset, 'right': 1 - right_offset, 'bottom': 0 + bottom_offset, 'top': 1 - top_offset}, subplot_kw={
                    'aspect': 'equal'})

            ax_iter = iter(ax)
            for axis in ax:
                axis.axis('off')

            # ---------- Load Density Image ---------------
            axis = next(ax_iter)
            perecentile = np.percentile(ds.image, [0.5, 99.5])
            image = np.clip(ds.image, *perecentile)
            image = image[:, :, 2]  # 0: TE=10ms, 1:TE=20ms, 2:TE=30ms, ... 9:TE=100ms
            image[~crop_mask_knee] = 0
            axis.imshow(image[focus], cmap='gray', interpolation='none')
            axis.set_title('Morphologic', fontdict=fontdict) if set_title else None
            axis.text(-0.02,
                      0.5,
                      ' '.join([protocol[0:protocol.index('S')],
                                protocol[protocol.index('S'):]]),
                      transform=axis.transAxes,
                      fontdict=fontdict,
                      rotation='vertical',
                      horizontalalignment='right',
                      verticalalignment='center')
            draw_patches(axis, focuses[knee]) if not use_focus else None

            # ---------- Load LSE-Image ---------------
            paras = ds.load_paras(method="lse")
            para_map = paras[:, :, para_n].reshape(ds.image.shape[0:-1])
            # paras = ds.comp_lse_paras(method="lse")
            # para_map = paras[:, para_n].reshape(ds.image.shape[0:-1])
            para_map[~crop_mask] = np.nan  # mark as "bad"
            axis = next(ax_iter)
            axis.imshow(image[focus], cmap='gray', interpolation='none')
            axis.imshow(para_map[focus], vmin=vmin, vmax=vmax, cmap='jet', interpolation='none')
            axis.set_title('LSE', fontdict=fontdict) if set_title else None
            draw_patches(axis, focuses[knee]) if not use_focus else None

            # ---------- Load Offset LSE Image ---------------
            paras = ds.load_paras(method="lse-unbiased")
            para_map = paras[:, :, para_n].reshape(ds.image.shape[0:-1])
            # paras = ds.comp_lse_paras(method="lse-unbiased")
            # para_map = paras[:, para_n].reshape(ds.image.shape[0:-1])
            para_map[~crop_mask] = np.nan  # mark as "bad"
            axis = next(ax_iter)
            axis.imshow(image[focus], cmap='gray', interpolation='none')
            axis.imshow(para_map[focus], vmin=vmin, vmax=vmax, cmap='jet', interpolation='none')
            axis.set_title('OLSE', fontdict=fontdict) if set_title else None
            draw_patches(axis, focuses[knee]) if not use_focus else None

            # ---------- Load Noise Corrected Image ---------------
            paras = ds.load_paras(method="lse-noise_corrected")
            para_map = paras[:, :, para_n].reshape(ds.image.shape[0:-1])
            # paras = ds.comp_lse_paras(signalpoints=ds.signalpoints_filtered, method="lse")
            # para_map = paras[:, para_n].reshape(ds.image.shape[0:-1])
            para_map[~crop_mask] = np.nan  # mark as "bad"
            axis = next(ax_iter)
            axis.imshow(image[focus], cmap='gray', interpolation='none')
            axis.imshow(para_map[focus], vmin=vmin, vmax=vmax, cmap='jet', interpolation='none')
            axis.set_title('NCLSE', fontdict=fontdict) if set_title else None
            draw_patches(axis, focuses[knee]) if not use_focus else None

            # ---------- Load NN-Image ---------------
            paras = ds.load_paras(method="nn")
            para_map = paras[:, :, para_n].reshape(ds.image.shape[0:-1])
            # paras = ds.comp_nn_paras(model=path_run)
            # para_map = paras[:, para_n].reshape(ds.image.shape[0:-1])
            para_map[~crop_mask] = np.nan  # mark as "bad"
            axis = next(ax_iter)
            axis.imshow(image[focus], cmap='gray', interpolation='none')
            ax_img = axis.imshow(para_map[focus], vmin=vmin, vmax=vmax, cmap='jet', interpolation='none')
            axis.set_title('NN', fontdict=fontdict) if set_title else None
            draw_patches(axis, focuses[knee]) if not use_focus else None

            # ---------- Load Noise Filtered Image---------------
            # paras = ds.load_paras(method="lse", noise_suppressed=True)
            # para_map = paras[:, :, para_n].reshape(ds.image.shape[0:-1])
            # # paras = ds.comp_lse_paras(signalpoints=ds.signalpoints_filtered, method="lse")
            # # para_map = paras[:, para_n].reshape(ds.image.shape[0:-1])
            # para_map[~crop_mask] = np.nan # mark as "bad"
            # axis = next(ax_iter)
            # axis.imshow(image[focus], cmap='gray', interpolation='none')
            # axis.imshow(para_map[focus], vmin=vmin, vmax=vmax, cmap='jet', interpolation='none')
            # axis.set_title('Low SNR (NFLSE estimation)', fontdict=fontdict) if set_title else None
            # draw_patches(axis, focuses[knee]) if not use_focus else None

            # ------------ Add Colorbar ------------
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(ax_img, cax=cax)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(r'$\mathbf{T2}$', **fontdict)
            axis.set_aspect('auto')

            # --------------- Save ----------
            # fig.tight_layout()
            fig.savefig(path_out / ('visual_' + ds.protocol + '.png'), dpi=300)
