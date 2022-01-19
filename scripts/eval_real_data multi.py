

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd
import seaborn as sns
import json


from fitting.data import RealExpDecayDataset
from fitting.visual import fontdict
import scipy.stats


def dict_extend(input_dict, data, *keys):
    if len(keys) == 1:
        if keys[0] in input_dict:
            input_dict[keys[0]].extend(data)
        else:
            input_dict[keys[0]] = data  # must be a list

    for key in keys:
        if key in input_dict:
            return dict_extend(input_dict[key], data, *keys[1:])
        else:
            input_dict[key] = {}
            return dict_extend(input_dict[key], data, *keys[1:])


logger = logging.getLogger()

if __name__ == "__main__":
    # ------------- Settings -------------------
    path_run = Path.cwd() / 'runs/2021_03_10_202304'
    for protocol in ["lowSNR", "highSNR"]:
        for crop in ['tissue', 'knee_core']:
            knees = [1, 2, 3, 4, 5, 6, 7]
            crop_dict = {'knee_core': "Entire Joint", "tissue": "Patellofemoral Cartilage"}
            dms, n_dm = [None], 0  # Just for compatibility
            params = RealExpDecayDataset.params_dict()

            # --------------- Create Output Directory -----------
            path_base_dir = Path.cwd() / 'results' 
            result_subdir = "RealExpDecayDataset"
            path_out = path_base_dir / result_subdir / 'fusion' / protocol / crop
            path_out.mkdir(parents=True, exist_ok=True)

            path_results = path_base_dir / result_subdir

            # ---------------- Load data -------------------
            error_dicts = {}
            true_dicts = {}
            snr_dicts = {}
            for knee in knees:
                path_knee = path_results / str(knee) / protocol / crop

                with open(path_knee / 'error_dict.json') as f:
                    error_dict = json.load(f)

                for para_n, error_para in error_dict.items():
                    for method, errors in error_para.items():
                        dict_extend(error_dicts, errors, para_n, method)

                with open(path_knee / 'true_dict.json') as f:
                    true_dict = json.load(f)

                for para_n, gts in true_dict.items():
                    dict_extend(true_dicts, gts, para_n)

                with open(path_knee / 'snr_dict.json') as f:
                    snr_dict = json.load(f)

                for snr_mod, snr_val in snr_dict.items():
                    dict_extend(snr_dicts, snr_val, snr_mod)

            # ------------- Error Statistics ---------------
            results_dict = {etype: {para_key: {} for para_key in params.keys()} for etype in ['abs', 'rel']}
            snr = np.round(np.median(snr_dicts['SNR']))
            data_abs = []
            data_rel = []
            for para_n in error_dicts.keys():
                para_key = list(params.keys())[int(para_n)]
                error_dict = error_dicts[para_n]
                ref = np.asarray(true_dicts[para_n])

                for etype in ['abs', 'rel']:
                    div = 1 if etype == 'abs' else ref / 100
                    type_error_dict = {method: np.abs(error) / div for method, error in error_dict.items()}
                    med_NN = np.median(type_error_dict['NN'])

                    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                    ax = iter(ax.flatten())

                    for method, abs_error in type_error_dict.items():
                        m = np.percentile(abs_error, q=[50, 2.5, 97.5])

                        results_dict[etype][para_key][method + '_mean_' + str(n_dm)] = np.mean(abs_error)
                        results_dict[etype][para_key][method + '_std_' + str(n_dm)] = np.std(abs_error, ddof=1)
                        results_dict[etype][para_key][method + '_median_' + str(n_dm)] = m[0]
                        results_dict[etype][para_key][method + '_q0_' + str(n_dm)] = m[1]
                        results_dict[etype][para_key][method + '_q1_' + str(n_dm)] = m[2]

                        results_dict[etype][para_key]['SNR_' + str(n_dm)] = snr

                        # Plot error distribution
                        axis = next(ax)
                        sns.histplot(abs_error, binrange=np.percentile(abs_error, q=[2.5, 97.5]), ax=axis)
                        axis.set_title(etype + " T2 quantification error of " + method)
                        axis.axvline(np.mean(abs_error), c='r', label="Mean {:.1f}".format(np.mean(abs_error)))
                        axis.axvline(np.median(abs_error), c='g', label="Median {:.1f}".format(np.median(abs_error)))
                        axis.legend()

                        if method != "NN":
                            p_mea = scipy.stats.wilcoxon(type_error_dict['NN'], abs_error, alternative='less')[1]
                            p_med = scipy.stats.median_test(type_error_dict['NN'], abs_error)[1]
                            direc = '<' if med_NN < m[0] else '>'
                            results_dict[etype][para_key][method + '_wilcoxon_' +
                                                        str(n_dm)] = '<0.001' if p_mea < 0.001 else '{:.3f}'.format(p_mea)
                            results_dict[etype][para_key][method + '_mood_' + str(n_dm)] = '{:.5f}{}'.format(p_med, direc)
                    fig.savefig(path_out / ('error_dist_' + etype + '.pdf'))

                # ----------- Store Error --------------
                for method, error in error_dict.items():
                    data_method = pd.DataFrame({'SNR': snr, 'Error': error, 'Method': method, 'Parameter': para_key})
                    data_abs.append(data_method)

                # ----------- Store rel. Error --------------
                for method, error in error_dict.items():
                    error = np.asarray(error)
                    ref_val = ref
                    mask2 = ref_val != 0
                    logger.info("Relative error estimation not possible in {} cases as reference is zero".format(np.sum(~mask2)))
                    data_method = pd.DataFrame({'SNR': snr,
                                                'Error': error[mask2] / ref_val[mask2] * 100,
                                                'Method': method,
                                                'Parameter': para_key})
                    data_rel.append(data_method)

            data_abs = pd.concat(data_abs, ignore_index=True)
            data_rel = pd.concat(data_rel, ignore_index=True)

            # ----- SNR -----------
            snr_a = np.asarray(snr_dicts['SNR'])
            fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))

            sns.histplot(snr_a[:, 0], bins='auto', binrange=np.percentile(snr_a[:, 0], q=(0.1, 99.9)), stat="density", ax=ax[0])
            mean_snr, std_snr = np.mean(snr_a[:, 0]), np.std(snr_a[:, 0])
            median_snr = np.median(snr_a[:, 0])
            ax[0].set_title('TE start in ' + crop)
            ax[0].axvline(mean_snr, color='r', linestyle='--', label=f"Mean {mean_snr:.2f} ± {std_snr:.2f}")
            ax[0].axvline(median_snr, color='g', linestyle='-', label=f"Median {median_snr:.2f}")
            ax[0].legend()

            sns.histplot(snr_a[:, -1], bins='auto', binrange=np.percentile(snr_a[:, -1], q=(0.1, 99.9)), stat="density", ax=ax[1])
            mean_snr, std_snr = np.mean(snr_a[:, -1]), np.std(snr_a[:, -1])
            median_snr = np.median(snr_a[:, -1])
            ax[1].set_title('TE end in ' +  crop)
            ax[1].axvline(mean_snr, color='r', linestyle='--', label=f"Mean {mean_snr:.2f} ± {std_snr:.2f}")
            ax[1].axvline(median_snr, color='g', linestyle='-', label=f"Median {median_snr:.2f}")
            ax[1].legend()

            fig.savefig(path_out / ("snr_" + protocol + ".png"), dpi=300)
            logger.info("SNR at TE=10: {:.0f} ±  {:.0f}, IQR [{:.0f}; {:.0f}]".format(
                np.mean(snr_a[:, 0]), np.std(snr_a[:, 0]), *np.percentile(snr_a[:, 0], q=(25, 75))))

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
                    axis.set_title(crop_dict[crop], fontdict=fontdict)
                    axis.set_ylabel(r"Relative " + para_name + "-Quantification Error " + unit, fontdict=fontdict)
                    axis.set_xlabel(None)
                    axis.set_xticks([])
                    fig.tight_layout()
                    fig.savefig(path_out / ("box-plot_" + etype + "_" + protocol + "_" + para_key + ".png"), dpi=300)

                    # ------------- Write box-plot statistics to file -------
                    stats = {}
                    for x in [snr]:  # data.x.unique()
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
                    with open(path_out / ('box-plot_' + etype + '-stats_' + para_key + '.json'), 'w') as f:
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
