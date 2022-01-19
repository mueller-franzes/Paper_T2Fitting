

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sns
import numpy as np
import json
import torch

from fitting.data import ExpDecayDataModule
from fitting.visual import fontdict
import scipy.stats
import pandas as pd

logger = logging.getLogger()


if __name__ == "__main__":
    # ------------- Settings -------------------
    path_run = Path.cwd() / 'runs/2021_03_10_202304'
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Create Data with different SNR  ----------------------
    snrs = [5, 10, 15, 20, 25, 30]
    dms = [ExpDecayDataModule(samples_test=2**19,
                              sigma_dist=None,  # disable
                              snr_dist=[round(snr), round(snr)])
           for snr in snrs]
    params = dms[0].params_dict()

    # --------------- Create Output Directory -----------
    path_out = Path.cwd() / 'results'
    result_subdir = dms[0].__class__.__name__.split('Module')[0]
    path_out = path_out / result_subdir / 'ensemble'
    path_out.mkdir(parents=True, exist_ok=True)

    # -------- Iterate over every dataset/SNR and evaluate model --------
    results_dict = {etype: {para_key: {} for para_key in params.keys()} for etype in ['abs', 'rel']}
    data_abs = []
    data_rel = []
    failed = {snr: {} for snr in snrs}
    for n_dm, dm in enumerate(dms):

        # ----------- Setup Data ----------------
        dm.setup(stage='test', mkdir=False)
        ds = dm.ds_test
        dl = dm.test_dataloader()

        snr = ds.snr_dist[0]  # [SNR_min, SNR_max], here equal
        logger.info("SNR {}".format(snr))

        # -------------- Compute parameter predictions -----------
        params_pred_nn = ds.comp_nn_paras(path_run)
        params_pred_lse, failed[snr]['LSE'] = ds.comp_lse_paras("lse")
        # params_pred_rlse, failed[snr]['RLSE'] = ds.comp_lse_paras("lse-robust")
        params_pred_olse, failed[snr]['ULSE'] = ds.comp_lse_paras("lse-unbiased")
        params_pred_nclse, failed[snr]['NCLSE'] = ds.comp_lse_paras("lse-noise_corrected")
        params_true = torch.cat([batch["parameters"] for batch in dl]).numpy()

        for para_n, (para_key, para_name) in enumerate(params.items()):
            if para_n == 0:  # Skip evaluation of Parameter A
                continue
            logger.info("Parameter {}".format(para_name))

            # ----------------- Compute Errors -------------
            error_dict = {
                'LSE': params_pred_lse[:, para_n] - params_true[:, para_n],
                # 'RLSE': params_pred_rlse[:, para_n] - params_true[:, para_n],
                'OLSE': params_pred_olse[:, para_n] - params_true[:, para_n],
                'NCLSE': params_pred_nclse[:, para_n] - params_true[:, para_n],
                'NN': params_pred_nn[:, para_n] - params_true[:, para_n]
            }
            # If A = 0 then tau can't be estimated:
            if para_key == 'tau':
                mask = params_true[:, 0] != 0
                logger.info("Tau estimation not possible in {} cases".format(np.sum(~mask)))
                for method, error in error_dict.items():
                    error_dict[method] = error[mask]
            else:
                mask = slice(None)

            # ------------- Error Statistics ---------------
            for etype in ['abs', 'rel']:
                div = 1 if etype == 'abs' else params_true[mask, para_n] / 100
                type_error_dict = {method: np.abs(error) / div for method, error in error_dict.items()}

                med_NN = np.median(type_error_dict['NN'])

                for method, abs_error in type_error_dict.items():
                    m = np.percentile(abs_error, q=[50, 2.5, 97.5])

                    results_dict[etype][para_key][method + '_mean_' + str(n_dm)] = np.mean(abs_error)
                    results_dict[etype][para_key][method + '_std_' + str(n_dm)] = np.std(abs_error, ddof=1)
                    results_dict[etype][para_key][method + '_median_' + str(n_dm)] = m[0]
                    results_dict[etype][para_key][method + '_q0_' + str(n_dm)] = m[1]
                    results_dict[etype][para_key][method + '_q1_' + str(n_dm)] = m[2]

                    results_dict[etype][para_key]['SNR_' + str(n_dm)] = snr

                    if method != "NN":
                        p_mea = scipy.stats.wilcoxon(type_error_dict['NN'], abs_error, alternative='less')[1]
                        p_med = scipy.stats.median_test(type_error_dict['NN'], abs_error)[1]
                        p_nor = scipy.stats.kstest((abs_error - np.mean(abs_error)) / np.std(abs_error), 'norm')[1]
                        direc = '<' if med_NN < m[0] else '>'
                        results_dict[etype][para_key][method + '_wilcoxon_' +
                                                      str(n_dm)] = '<0.001' if p_mea < 0.001 else '{:.3f}'.format(p_mea)
                        results_dict[etype][para_key][method + '_mood_' +
                                                      str(n_dm)] = '<0.001' + direc if p_med < 0.001 else '{:.3f}{}'.format(p_med, direc)
                        results_dict[etype][para_key][method + '_ks_' +
                                                      str(n_dm)] = '<0.001' if p_nor < 0.001 else '{:.3f}'.format(p_nor)

            # ----------- Store Error --------------
            for method, error in error_dict.items():
                data_method = pd.DataFrame({'SNR': snr, 'Error': error, 'Method': method, 'Parameter': para_key})
                data_abs.append(data_method)

            # ----------- Store rel. Error --------------
            for method, error in error_dict.items():
                data_method = pd.DataFrame(
                    {'SNR': snr, 'Error': error / params_true[mask, para_n] * 100, 'Method': method, 'Parameter': para_key})
                data_rel.append(data_method)

    data_abs = pd.concat(data_abs, ignore_index=True)
    data_rel = pd.concat(data_rel, ignore_index=True)

    with open(path_out / ('failed-fittings_snr.json'), 'w') as f:
        json.dump(failed, f)

    for para_n, (para_key, para_name) in enumerate(params.items()):
        if para_n == 0:  # Skip evaluation of Parameter A
            continue
        for etype, data in zip(['abs', 'rel'], [data_abs, data_rel]):
            unit = '[%]' if etype == 'rel' else ('[ms]' if para_key == "tau" else "")

            # ------------------ Box Plot ---------------------
            fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
            sns_c = (0.23921568627450981, 0.23921568627450981, 0.23921568627450981, 1.0)
            sns.boxplot(x='SNR', y='Error', hue='Method', data=data[data['Parameter'] == para_key],
                        showmeans=False, ax=axis, showfliers=False, width=0.6,
                        meanprops={"marker": "D", "markerfacecolor": sns_c, "markeredgecolor": sns_c, "markersize": 4})
            axis.axhline(ls='--', c='k')
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.legend(loc='best', shadow=False)
            axis.set_ylabel(r"Relative " + para_name + "-Quantification Error " + unit, fontdict=fontdict)
            axis.set_xlabel(r"SNR", fontdict=fontdict)
            fig.tight_layout()
            fig.savefig(path_out / ("box-plot_snr-" + etype + "_" + para_key + ".png"), dpi=300)
            fig.savefig(path_out / ("box-plot_snr-" + etype + "_" + para_key + ".tiff"), dpi=300)

            # ------------------ Violin Plot ---------------------
            fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
            sns_c = (0.23921568627450981, 0.23921568627450981, 0.23921568627450981, 1.0)
            sns.violinplot(x='SNR',
                           y='Error',
                           hue='Method',
                           data=data[data['Parameter'] == para_key],
                           cut=0,
                           scale='count',
                           scale_hue=False,
                           inner='box')
            axis.legend(loc='best', shadow=False)
            axis.set_ylabel(r"Relative " + para_name + "-Quantification Error " + unit, fontdict=fontdict)
            axis.set_xlabel(r"SNR", fontdict=fontdict)
            fig.tight_layout()
            fig.savefig(path_out / ("violin-plot_snr-" + etype + "_" + para_key + ".png"), dpi=300)

            # ------------- Write box-plot statistics to file -------
            stats = {}
            for x in snrs:  # data.x.unique()
                mask = (data['SNR'] == x) & (data['Parameter'] == para_key)
                dfgb = data[mask][['Method', "Error"]].groupby('Method')
                error_list = dfgb['Error'].apply(list).tolist()
                box_stats = boxplot_stats(error_list)
                stats[x] = {method: {} for method in dfgb.groups.keys()}
                for method, box_stat in zip(dfgb.groups.keys(), box_stats):
                    stats[x][method]['whishi'] = box_stat['whishi']
                    stats[x][method]['whislo'] = box_stat['whislo']
                    stats[x][method]['iqr'] = box_stat['iqr']
                    stats[x][method]['q1'] = box_stat['q1']
                    stats[x][method]['q3'] = box_stat['q3']
                    stats[x][method]['mean'] = box_stat['mean']
                    stats[x][method]['med'] = box_stat['med']
            with open(path_out / ('box_snr-' + etype + '-stats_' + para_key + '.json'), 'w') as f:
                json.dump(stats, f)

            # ----------------- Write data as table -----------------------------------
            table_str = ''.join(['\tSNR={{SNR_{n}:.1f}}'.format(n=str(n)) for n in range(len(dms))])
            for method in type_error_dict.keys():
                table_str += '\n' + method
                for n in range(len(dms)):
                    table_str += '\t{{{method}_median_{n}:.0f}} [{{{method}_q0_{n}:.0f}}, {{{method}_q1_{n}:.0f}}]'.format(
                        method=method, n=str(n))

            table_str += '\n\n'
            for method in type_error_dict.keys():
                if method == "NN":
                    continue
                table_str += '\nSig-' + method
                for n in range(len(dms)):
                    table_str += '\t{{{method}_mood_{n}}}'.format(method=method, n=str(n))

            with open(path_out / ('table_snr-' + etype + '_' + para_key + '.txt'), 'w') as f:
                f.write(table_str.format(**results_dict[etype][para_key]))
