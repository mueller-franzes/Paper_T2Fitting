from pathlib import Path

import numpy as np
import torch
import scipy.stats
from scipy.optimize import curve_fit
import scipy.special as sc



from fitting.utils import uniform_exp
from fitting.data.data_module_base import BaseDataModule
from fitting.noise import noise_corrected_exp


from multiprocessing import Pool, get_context, cpu_count


class ExpDecayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: int,
        num_points_bounds=[5, 15],  # IntUniform() [low, high]
        timepoints_start_dist=(5, 15),  # Uniform()
        timepoints_step_dist=(2, 15),  # Uniform()
        initial_signal_dist=None,  # (0, 2000),  # Amplite Distribution Parameters Uniform()
        relaxation_time_dist=(0.66, 5, 70),  # Relaxationtime Distribution Parameters LogNorm()
        snr_dist=(2, 30),   # Signal-Noise-Ratio(SNR) Distribution Parameters Uniform()
        sigma_dist=[0, 300],  # [0,30],#[10, 500], #[0, 1000],#[0, 25], # None, # [0, 30]
        seed: int = 0,
        ram: bool = False
    ):

        self.samples = samples
        self.num_points_bounds = num_points_bounds
        self.timepoints_start_dist = timepoints_start_dist
        self.timepoints_step_dist = timepoints_step_dist
        self.initial_signal_dist = initial_signal_dist
        self.relaxation_time_dist = relaxation_time_dist
        self.snr_dist = snr_dist
        self.sigma_dist = sigma_dist
        self.seed = seed
        self.ram = ram
        self.lse_bounds = [[0, 5], [2500, 500]]

        np.random.seed(self.seed)

        if self.ram:
            print("Start Data Creation")
            processes = min(cpu_count(), self.samples)
            with Pool(processes=processes) as pool:
                data = pool.map(self._sample, range(self.samples), chunksize=self.samples // processes)
            self.data = data
            print("Data loaded in RAM")

    def __getitem__(self, item: int):
        if self.ram:
            return self.data[item]
        else:
            return self._sample(item)

    def _sample(self, item):
        np.random.seed(self.seed + item)

        # ----- Sample signal parameters ----
        # initial_signal = np.random.uniform(*self.initial_signal_dist)
        initial_signal = uniform_exp.rvs()
        initial_signal = np.floor(initial_signal)  # WORKAROUND: Sampling exactly 0.0 will prob. never happen
        relaxation_time = scipy.stats.lognorm.rvs(*self.relaxation_time_dist)
        # relaxation_time = np.random.uniform(15, 120)

        # ---- Sample timepoints at which signal is evaluated ---
        num_points = np.random.randint(self.num_points_bounds[0], self.num_points_bounds[1] + 1)
        timepoints_start = np.random.uniform(*self.timepoints_start_dist)
        timepoints_step = np.random.uniform(*self.timepoints_step_dist)

        timepoints = timepoints_start + timepoints_step * np.arange(num_points)

        # ---- Sample noise level ----
        if self.sigma_dist is not None:
            sigma = np.random.uniform(*self.sigma_dist)
        else:
            snr = np.random.uniform(*self.snr_dist)
            # sigma = initial_signal / snr # WARNING: if initial_signal = 0  => sigma = 0
            # sigma_bg = np.random.uniform(2, 25)
            # sigma_signal = initial_signal / snr
            # sigma = max(sigma_bg, sigma_signal)
            #
            if initial_signal == 0:  # background
                sigma = np.random.uniform(10, 25)
            else:
                sigma = initial_signal / snr  # [1/25 ... 1/2 ... 2000/25 ..... ]

        signalpoints = self.noisy_signal_func(sigma=sigma,
                                              x=timepoints,
                                              initial_signal=initial_signal,
                                              relaxation_time=relaxation_time,
                                              )

        # ---- Scale signal ----
        signal_scale = np.array(1)
        # signal_scale =  signalpoints[0] # np.max(signalpoints)
        # signalpoints = signalpoints / signal_scale
        # initial_signal = initial_signal / signal_scale

        # -------------- Unify shape of signalpoints and timepoints (otherwise no batch learning) --------
        max_num_points = np.max(self.num_points_bounds)
        signalpoints_unify = np.ones(max_num_points) * -1
        signalpoints_unify[0:num_points] = signalpoints
        timepoints_unify = np.ones(max_num_points) * -1
        timepoints_unify[0:num_points] = timepoints

        return {'signalpoints': signalpoints_unify.astype(np.float32),
                "timepoints": timepoints_unify.astype(np.float32),
                'parameters': np.asarray([initial_signal, relaxation_time]).astype(np.float32),
                'signal_scale': signal_scale.astype(np.float32),
                'sigma': np.float32(sigma),
                }

    def __len__(self):
        return self.samples

    def model_func(self, x, initial_signal, relaxation_time):
        if isinstance(x, torch.Tensor):
            return torch.where(x >= 0, self.signal_func(x, initial_signal, relaxation_time),
                               torch.tensor(-1.0, device=x.device))
        else:
            return np.where(x >= 0, self.signal_func(x, initial_signal, relaxation_time), -1.0)

    def model_func_bias(self, x, initial_signal, relaxation_time, bias):
        return np.where(x >= 0, (self.signal_func(x, initial_signal, relaxation_time).T + bias).T, -1.0)

    def jac_model_func(self, x, initial_signal, relaxation_time):
        partial_s0 = np.exp(-x/relaxation_time)
        partial_t2 = initial_signal*x/(relaxation_time**2) * np.exp(-x/relaxation_time)
        return np.asarray([partial_s0, partial_t2]).T
    
    def jac_model_func_bias(self, x, initial_signal, relaxation_time, bias):
        partial_s0 = np.exp(-x/relaxation_time)
        partial_t2 = initial_signal*x/(relaxation_time**2) * np.exp(-x/relaxation_time)
        partial_c = np.full(len(x), 1.0, dtype=x.dtype) 
        return np.asarray([partial_s0, partial_t2, partial_c]).T

    def jac_model_func_cor(self, x, initial_signal, relaxation_time, sigma):
        eps = 1e-8
        initial_signal = initial_signal + eps
        sigma = sigma + eps

        alpha = (initial_signal / (2 * sigma))**2
        expo = np.exp(-x / relaxation_time)
        # print(alpha)
        if alpha < 8:
            partial_s = np.sqrt(np.pi * alpha / 2) * np.exp(-alpha) * (sc.i0(alpha) + sc.i1(alpha))
        else:
            partial_s = 1 - 1 / (8 * alpha)

        partial_t2 = partial_s * initial_signal * x / relaxation_time**2 * expo
        partial_s0 = partial_s * expo
        return np.asarray([partial_s0, partial_t2]).T

    @property
    def model_func_str(self):
        return r"${:.2} \cdot exp(-t/{:.2})$"
        # return r"$S_0 \cdot exp(-t/T2)$"

    @classmethod
    def params_str(cls):
        return r"$S_0={:.2f}\; T2={:.2f}$"
        # return r"T2={:.2f}$"

    @classmethod
    def params_dict(cls):
        return {'a': r'$\mathbf{S_0}$', 'tau': r'$\mathbf{T2}$'}
        # return {'tau': r'$T2$'}

    def params_bounds(self):
        return list(zip(self.initial_signal_range, self.relaxation_time_range))

    @classmethod
    def params_amplitude_scale(cls, params, scale):
        params[:, 0] = params[:, 0] * scale
        return params

    @classmethod
    def signal_func(cls, x, initial_signal, relaxation_time):
        if isinstance(x, torch.Tensor):
            return (initial_signal * torch.exp(-x.T / relaxation_time)).T
        else:
            x = np.asarray(x)
            return (initial_signal * np.exp(-x.T / relaxation_time)).T

    @classmethod
    def noisy_signal_func(cls, sigma=0, **kwargs):
        """Add noise to signal function

        Args:
            sigma (float): std of noise (complex noise added to signal - riccian noise), can be 0 -> noise-free

        Returns:
            array of floats: the (noisy) signal values
        """

        # Signal
        signal_real = cls.signal_func(**kwargs)
        n_singalpoints = len(signal_real)

        #  Noise
        noise_real = np.random.normal(scale=sigma, size=n_singalpoints)  # real part of noise
        noise_imag = np.random.normal(scale=sigma, size=n_singalpoints)  # imaginary part of noise
        noise = np.sqrt(noise_real**2 + noise_imag**2)  # Amplitude of noise (Absolute Value)

        # Add error
        signal = np.sqrt((signal_real + noise_real)**2 + (0 + noise_imag)**2)   # Amplitude of signal

        return signal

    def comp_lse_paras(self, config=None, **kwargs):
        self._lse_fitting_config = config
        self._lse_fitting_kwargs = kwargs
        samples = list(range(self.samples))
        processes = min(cpu_count(), len(samples))
        print("Using {} Processor".format(processes))
        with get_context('fork').Pool(processes=processes) as pool:
            params_failed = pool.map(self._lse_fitting, samples, chunksize=len(samples) // processes)
        params, failed = map(list, zip(*params_failed))  # [Singalpoints, Parameters]
        return np.asarray(params), sum(failed)

    def _lse_fitting(self, n_sample):
        data = self._sample(n_sample)
        timepoints = data['timepoints']
        signalpoints = data['signalpoints']
        sigma = data['sigma']
        mask = timepoints >= 0
        try:
            if self._lse_fitting_config =="lse":
                opt_para = curve_fit(self.model_func, timepoints[mask], signalpoints[mask],
                                     bounds=self.lse_bounds, jac=self.jac_model_func, method='trf', loss='linear',
                                     sigma=None, f_scale=1, max_nfev=None,  p0=[250, 50])
            elif self._lse_fitting_config == "lse-robust":
                opt_para = curve_fit(self.model_func, timepoints[mask], signalpoints[mask],
                                     bounds=self.lse_bounds, jac="3-point", method='trf', loss='huber',
                                     sigma=None, f_scale=10, max_nfev=None, p0=[250, 50])
            elif self._lse_fitting_config == "lse-unbiased":
                opt_para = curve_fit(self.model_func_bias, timepoints[mask], signalpoints[mask],
                                     bounds=[[0, 5, 0], [2500, 500, 2500]], jac=self.jac_model_func_bias, method='trf', loss='linear',
                                     sigma=None, f_scale=1, max_nfev=None, p0=[250, 50, 0])
                opt_para = [opt_para[0][0:2], opt_para[1][0:2]] if not self._lse_fitting_kwargs.get(
                    'all_variables', False) else opt_para
            elif self._lse_fitting_config == "lse-noise_corrected":
                def func_wrapper(x, initial_signal, relaxation_time):
                    return noise_corrected_exp(self.model_func(x, initial_signal, relaxation_time), sigma)

                def jac_wrapper(x, initial_signal, relaxation_time):
                    return self.jac_model_func_cor(x, initial_signal, relaxation_time, sigma)
                opt_para = curve_fit(func_wrapper, timepoints[mask], signalpoints[mask],
                                     bounds=self.lse_bounds, jac=jac_wrapper, method='trf', loss='linear',
                                     sigma=None, f_scale=1, max_nfev=None, p0=[250, 50])
            elif self._lse_fitting_config == "custom":
                opt_para = curve_fit(self.model_func, timepoints[mask], signalpoints[mask],
                                     bounds=self.lse_bounds, **self._lse_fitting_kwargs)
            else:
                raise ValueError()
            return opt_para[0], 0
        except RuntimeError:
            # return [float('nan')] * len(self.params_dict()) # WARNING: NaN might be interpreted as 0
            if self._lse_fitting_config == "lse-unbiased":
                return [0, 5], 1 if not self._lse_fitting_kwargs.get('all_variables', False) else [0, 5, 0], 1
            else:
                return self.lse_bounds[0], 1  # set (arbitrary) to lower bound

    def comp_nn_paras(self, model, dl=None, dict_key='params', skip=False, device=None):
        if dl is None:
            from torch.utils.data.dataloader import DataLoader
            dl = DataLoader(self, batch_size=4096, num_workers=cpu_count(), shuffle=False)

        if isinstance(model, str) or isinstance(model, Path):
            from fitting.models import FCN
            if Path(model).suffix == '.ckpt':
                model_nn = FCN.load_from_checkpoint(str(model))
            else:
                model_nn = FCN.load_best_checkpoint(str(model))
            model_nn.model_func = self.model_func
        else:
            model_nn = model
        model_nn.eval()
        model_nn.to(device) if device is not None else False
        if skip:
            params_pred_nns = torch.cat([model_nn(batch["signalpoints"][:, 1:], batch["timepoints"][:, 1:])[dict_key]
                                         for batch in dl]).detach().numpy()
        else:
            params_pred_nns = torch.cat([model_nn(batch["signalpoints"], batch["timepoints"])[dict_key]
                                         for batch in dl]).detach().numpy()
        return params_pred_nns


    def load_signalpoints(self, paras, timepoints_array, method='lse'):
        A = np.take(paras, 0, axis=-1).flatten()
        tau = np.take(paras, 1, axis=-1).flatten()
        if method == "lse-unbiased":
            offset = np.take(paras, 2, axis=-1).flatten()
            return self.model_func_bias(timepoints_array, A.flatten(), tau.flatten(), offset.flatten())
        else:
            return self.model_func(timepoints_array, A.flatten(), tau.flatten())

class ExpDecayDataModule(BaseDataModule):
    Dataset = ExpDecayDataset



