

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import numpy as np
import pandas as pd
import seaborn as sns
import json

import torch
from torch.utils.data.dataloader import DataLoader


from fitting.data import RealExpDecayDataset
from fitting.visual import fontdict
from fitting.utils import AdvJsonEncoder, cohen_w
import scipy.stats


logger = logging.getLogger()


def save_update(path_file, new_dict):
    if path_file.is_file():
        with open(path_file) as f:
            data = json.load(f)
    else:
        data = {}

    data.update(new_dict)

    with open(path_file, 'w') as f:
        json.dump(data, f, cls=AdvJsonEncoder)


if __name__ == "__main__":
    # ------------- Settings -------------------
    path_run = Path.cwd() / 'runs/2021_03_10_202304'
    for knee in [1, 2, 3, 4, 5, 6, 7]:
        for crop in ['tissue', 'knee_core']:
            # protocol="lowSNR"
            protocol = "highSNR"

            # stream_handler = logging.StreamHandler(sys.stdout)
            # stream_handler.setLevel(logging.DEBUG)
            # logger.addHandler(stream_handler)

            # ---------------- Load Data  ----------------------
            crop_dict = {'knee_core': "Entire Joint", "tissue": "Patellofemoral Cartilage"}
            dms, n_dm = [None], 0  # Just for compatibility
            ds = RealExpDecayDataset(knee=knee, protocol=protocol, crop=crop, norm=False, noise_est_method="vst-hmfg")
            samples = len(ds)
            print(samples)
            params = ds.params_dict()
            dl = DataLoader(ds, batch_size=len(ds), num_workers=0, shuffle=False)

            # --------------- Create Output Directory -----------
            path_out = Path.cwd() / 'results'
            result_subdir = ds.__class__.__name__.split('Module')[0]
            path_out = path_out / result_subdir / str(ds.knee) / ds.protocol / ds.crop_name
            path_out.mkdir(parents=True, exist_ok=True)

            # ---------------- Remove deprecated results -----------
            path_error_dict = path_out / 'error_dict.json'
            path_true_dict = path_out / 'true_dict.json'
            path_error_dict.unlink(missing_ok=True)
            path_true_dict.unlink(missing_ok=True)

            # --------------- Load Data ---------------
            timepoints = torch.cat([batch["timepoints"] for batch in dl]).numpy()  # [Samples, TimePoints]
            signalpoints = torch.cat([batch["signalpoints"] for batch in dl]).numpy()  # [Samples, TimePoints]
            signalpoints_ref = torch.cat([batch["signalpoints_ref"] for batch in dl]).numpy()  # [Samples, TimePoints]
            signal_scale = torch.cat([batch["signal_scale"] for batch in dl]).numpy()

            # -------------- Noise Stats -----------
            snr = ds.signalpoints / ds.noise  # [Voxels, Echo Time]
            snr_ref = ds.signalpoints_ref / ds.noise_ref

            with open(path_out / 'snr_dict.json', 'w') as f:
                json.dump({'SNR': snr, 'SNR_Ref': snr_ref}, f, cls=AdvJsonEncoder)

            np.set_printoptions(precision=0)
            snr = np.round(np.median(snr), 0)

            # ------ Option 1: Calculate parameters and measure calculation time 
            # time_diffs = {'NN':[], 'LSE': [], 'OLSE':[], 'NCLSE':[]}

            # ********* NN ************
            # device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # for n in range(100):
            #     start_time = time.time()
            #     params_pred_nn = ds.comp_nn_paras(path_run, device=device)
            #     # time_diffs['NN'].append(time.time() - start_time)
            #     time_diffs['NN'].append(ds.time_comp_nn)

            # # ******** LSE ************
            # params_true =  ds.comp_lse_paras(signalpoints=signalpoints_ref) # ground truth
            # for n in range(100):
            #     start_time = time.time()
            #     params_pred_lse = ds.comp_lse_paras()
            #     time_diffs['LSE'].append(time.time() - start_time)

            #     start_time = time.time()
            #     params_pred_olse = ds.comp_lse_paras(method="lse-unbiased")
            #     time_diffs['OLSE'].append(time.time() - start_time)

            #     start_time = time.time()
            #     ds.filter_noise() # Just here to estimate time for noise estimation
            #     params_pred_nclse = ds.comp_lse_paras(method="lse-noise_corrected")
            #     time_diffs['NCLSE'].append(time.time() - start_time)

            # # ----------- Save exectution time to file ---------
            # with open(path_out/'execution_time.json', 'w') as f:
            #     json.dump(time_diffs, f)

            # ------------ Print Time --------
            path_time_table = path_out / 'execution_time.json'
            if path_time_table.is_file():
                with open(path_time_table, 'r') as f:
                    time_diffs = json.load(f)

                fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
                ax = iter(ax.flatten())

                fval, p = scipy.stats.friedmanchisquare(*list(time_diffs.values()))
                print("Friedman test p-Value {:.4f}".format(p))
                for method, times in time_diffs.items():
                    stat, p_norm = scipy.stats.normaltest(times)
                    print("{} took {:.0f}s ±{:.0f}s ".format(method, np.mean(times), np.std(times, ddof=1)))
                    print("Check Normal Distribution p={:.3f}".format(p_norm))
                    if method != "NN":
                        stats, p_dif = scipy.stats.wilcoxon(time_diffs['NN'], times, alternative='less')
                        print("NN vs {} Wilcoxon test p={:.4f}".format(method, p_dif))

                    axis = next(ax)
                    sns.histplot(times, ax=axis)
                    axis.set_title("Computational time of " + method)
                fig.savefig(path_out / 'comp_time.png', dpi=300)

            # ------ Option 2: Load precalculated parameters
            params_pred_lse = ds.load_paras(method="lse")
            params_pred_olse = ds.load_paras(method="lse-unbiased")
            params_pred_nclse = ds.load_paras(method="lse-noise_corrected")
            params_pred_nn = ds.load_paras(method="nn")
            params_true = ds.load_paras(method="lse", protocol="highSNR")

            # ----------------- Evaluate Errors -------------
            data_abs = []
            data_rel = []
            results_dict = {etype: {para_key: {} for para_key in params.keys()} for etype in ['abs', 'rel']}
            for para_n, (para_key, para_name) in enumerate(params.items()):
                if para_n == 0:  # skip Parameter A
                    continue

                # ----------------- Compute Errors -------------
                error_dict = {
                    'LSE': params_pred_lse[:, para_n] - params_true[:, para_n],
                    # 'RLSE': params_pred_rlse[:, para_n] - params_true[:, para_n],
                    'OLSE': params_pred_olse[:, para_n] - params_true[:, para_n],
                    'NCLSE': params_pred_nclse[:, para_n] - params_true[:, para_n],
                    'NN': params_pred_nn[:, para_n] - params_true[:, para_n],
                    # 'NFLSE': np.abs(errors_nflse),
                }
                # If A = 0 then tau can't be estimated:
                if para_key == 'tau':
                    mask = params_true[:, 0] != 0
                    logger.info("Tau estimation not possible in {} cases".format(np.sum(~mask)))
                    for method, error in error_dict.items():
                        error_dict[method] = error[mask]
                else:
                    mask = slice(None)

                save_update(path_error_dict, {para_n: error_dict})
                save_update(path_true_dict, {para_n: params_true[mask, para_n]})

                # ------------- Error Statistics ---------------

                for etype in ['abs', 'rel']:
                    div = 1 if etype == 'abs' else params_true[mask, para_n] / 100
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

                        axis = next(ax)
                        sns.histplot(np.clip(abs_error, *np.percentile(abs_error, q=[2.5, 97.5])), bins=100, ax=axis)
                        axis.set_title(etype + " T2 quantification error of " + method)
                        axis.axvline(np.mean(abs_error), c='r', label="Mean {:.1f}".format(np.mean(abs_error)))
                        axis.axvline(np.median(abs_error), c='g', label="Median {:.1f}".format(np.median(abs_error)))
                        axis.legend()

                        if method != "NN":
                            p_mea = scipy.stats.wilcoxon(type_error_dict['NN'], abs_error, alternative='less')[1]
                            chi2, p_med, med, tbl = scipy.stats.median_test(type_error_dict['NN'], abs_error)

                            if etype == 'rel' and method == "LSE":
                                effect_size_w = cohen_w(tbl)
                                x1, x2 = type_error_dict['NN'], abs_error
                                effect_size_d = np.abs((np.mean(x1) - np.mean(x2)) /
                                                       np.sqrt((np.var(x1, ddof=1) + np.var(x2, ddof=1)) / 2))
                            direc = '<' if med_NN < m[0] else '>'
                            results_dict[etype][para_key][method + '_wilcoxon_' +
                                                          str(n_dm)] = '<0.001' if p_mea < 0.001 else '{:.3f}'.format(p_mea)
                            results_dict[etype][para_key][method + '_mood_' +
                                                          str(n_dm)] = '{:.4f} {}'.format(p_med, direc)
                    fig.savefig(path_out / ('error_dist_' + etype + '.pdf'))

                # ----------- Store Error --------------
                for method, error in error_dict.items():
                    data_method = pd.DataFrame({'SNR': snr, 'Error': error, 'Method': method, 'Parameter': para_key})
                    data_abs.append(data_method)

                # ----------- Store rel. Error --------------
                for method, error in error_dict.items():
                    ref_val = params_true[mask, para_n]
                    mask2 = ref_val != 0
                    logger.info(
                        "Relative error estimation not possible in {} cases as reference is zero".format(
                            np.sum(
                                ~mask2)))
                    data_method = pd.DataFrame({'SNR': snr,
                                                'Error': error[mask2] / ref_val[mask2] * 100,
                                                'Method': method,
                                                'Parameter': para_key})
                    data_rel.append(data_method)

            data_abs = pd.concat(data_abs, ignore_index=True)
            data_rel = pd.concat(data_rel, ignore_index=True)


            for para_n, (para_key, para_name) in enumerate(params.items()):
                if para_n == 0:  # Skip evaluation of Parameter A
                    continue
                for etype, data in zip(['abs', 'rel'], [data_abs, data_rel]):
                    unit = '[%]' if etype == 'rel' else ('[ms]' if para_key == "tau" else "")

                    # ------------------ Box Plot ---------------------
                    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))
                    sns_c = (0.23921568627450981, 0.23921568627450981, 0.23921568627450981, 1.0)
                    sns.boxplot(x='SNR',
                                y='Error',
                                hue='Method',
                                data=data[data['Parameter'] == para_key],
                                showmeans=False,
                                ax=axis,
                                showfliers=False,
                                width=0.6,
                                meanprops={"marker": "D",
                                           "markerfacecolor": sns_c,
                                           "markeredgecolor": sns_c,
                                           "markersize": 4})

                    axis.axhline(ls='--', c='k')
                    axis.spines['right'].set_visible(False)
                    axis.spines['top'].set_visible(False)
                    axis.legend(loc='best', shadow=False)
                    axis.set_title(crop_dict[ds.crop_name], fontdict=fontdict)
                    axis.set_ylabel(r"Relative " + para_name + "-Quantification Error " + unit, fontdict=fontdict)
                    axis.set_xlabel(None)
                    axis.set_xticks([])
                    fig.tight_layout()
                    fig.savefig(path_out / ("box-plot_" + etype + "_" + ds.protocol + "_" + para_key + ".png"), dpi=300)

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



               
