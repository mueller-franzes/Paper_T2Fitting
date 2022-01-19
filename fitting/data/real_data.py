
from pathlib import Path
import numpy as np
import torch
from pydicom import dcmread
from multiprocessing import Pool, get_context, cpu_count
import SimpleITK as sitk
from scipy.optimize import curve_fit
import scipy.special as sc

import fitting.noise as ns
import time


class RealExpDecayDataset():
    path_root_default = {1: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie1/DICOM'),
                         2: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie2/DICOM'),
                         3: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie3/DICOM'),
                         4: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie4/DICOM'),
                         5: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie5/DICOM'),
                         6: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie6/DICOM'),
                         7: Path('/home/gustav/Documents/datasets/KnieT2Messung/Knie7/DICOM')}

    protocol_names = {'lowSNR': {1: 'T2 map_2mm_NSA1_Sense3',
                                 2: 'T2map_highSNR_SENSE3_NSA1',
                                 3: 'T2map_lowSNR_SENSE3_NSA1',
                                 4: 'T2map_lowSNR_SENSE3_NSA1',
                                 5: 'T2map_lowSNR_SENSE3_NSA1',
                                 6: 'T2map_lowSNR_SENSE3_NSA1',
                                 7: 'T2map_lowSNR_SENSE3_NSA1'},
                      'highSNR': {1: 'T2 map_2mm_NSA5',
                                  2: 'T2map_highSNR_SENSE1_NSA4',
                                  3: 'T2map_highSNR_SENSE1_NSA4',
                                  4: 'T2map_highSNR_SENSE1_NSA4',
                                  5: 'T2map_highSNR_SENSE1_NSA4',
                                  6: 'T2map_highSNR_SENSE1_NSA4',
                                  7: 'T2map_highSNR_SENSE1_NSA4'}
                      }

    def __init__(self, knee, path_root=None, protocol='lowSNR', norm=False, crop=None,
                 noise_est_method='vst-hmfg'):  # vst-hmfg
        self.path_root = Path(path_root) if path_root is not None else self.path_root_default[knee]

        if protocol not in ['lowSNR', 'highSNR']:
            raise AttributeError("Unknown protocol {}".format(protocol))

       
        self.crop_name = crop
        if isinstance(crop, str):
            crop = self.get_crop(crop, self.path_root)
        elif crop is None:
            crop = slice(None)

        keep_image = isinstance(crop, tuple) or isinstance(crop, slice)

        # Read desire measurement
        image, timepoints = RealExpDecayDataset.read_dicom(
            self.path_root, self.protocol_names[protocol][knee])  # [x,y, timpoints]
        self.raw_image = image
        signalpoints, signal_scale = self._img2singal(image, crop, norm)
        self.timepoints = timepoints
        self.timepoints_array = np.repeat(timepoints[np.newaxis], len(signalpoints), axis=0)
        if keep_image:
            self.image = self._apply_crop(image, crop)
            self.image_normed = signalpoints.reshape(self.image.shape)
        self.signalpoints = signalpoints
        self.signal_scale = signal_scale

        # Read reference measurement
        image_ref, timepoints_ref = RealExpDecayDataset.read_dicom(
            self.path_root, self.protocol_names['highSNR'][knee])
        self.raw_image_ref = image_ref
        signalpoints_ref, signal_scale_ref = self._img2singal(image_ref, crop, norm)
        assert all(timepoints_ref == timepoints), "Timepoints between measurement and reference measurement differ!"
        if keep_image:
            self.image_ref = self._apply_crop(image_ref, crop)
            self.image_ref_normed = signalpoints_ref.reshape(self.image_ref.shape)
        self.signalpoints_ref = signalpoints_ref
        self.signal_scale_ref = signal_scale_ref

        # Estimate Noise
        noise, img_filtered = self.filter_noise(image, method=noise_est_method)
        # noise_filtered, _ = self.filter_noise(img_filtered, method=noise_est_method)
        self.noise = self._apply_crop(noise, crop)
        # self.noise_filtered = self._apply_crop(noise_filtered, crop)
        self.noisepoints, _ = self._img2singal(noise, crop, norm)
        # self.img_filtered = self._apply_crop(img_filtered, crop)
        # self.signalpoints_filtered, _ = self._img2singal(img_filtered, crop, norm)

        noise_ref, img_ref_filtered = self.filter_noise(image_ref, method=noise_est_method)
        self.noise_ref = self._apply_crop(noise_ref, crop)
        self.noisepoints_ref, _ = self._img2singal(noise_ref, crop, norm)
        # self.img_ref_filtered = self._apply_crop(img_ref_filtered, crop)

        # Determine noise level name
        self.snr_name = "high SNR" if protocol == "highSNR" else "low SNR"
        self.knee = knee
        self.protocol = protocol
        self.norm = norm
        self.crop = crop
        self.lse_bounds = [[0, 5], [2500, 500]]

    def filter_noise(self, image=None, method='vst-hmfg'):
        image = self.raw_image if image is None else image
        img_cor = np.empty(image.shape)
        sigma_maps = np.empty(image.shape)
        for time_slice_i in range(image.shape[-1]):
            img_slice = image[:, :, time_slice_i]

            # Estimate of Noise
            if method == "vst-hmfg":
                # A_0, sigma_0 = ns.expectation_maximization(img_slice, lm_method='average', lm_kernel=7)
                A_0, sigma_0 = ns.expectation_maximization(img_slice)
                # A_0, sigma_0 = ns.loc_moment(img_slice), ns.hmf_g(img_slice)

                # crop = self.get_crop('background', self.path_root)
                # bg = self._apply_crop(img_slice, crop)
                # sigma_0 = np.full(img_slice.shape, np.sum(bg) / (bg.size) * np.sqrt(2 / np.pi))
                # A_0 = image[:,:,0] 
                
                
                # snr_0 = A_0/ns.lpf(sigma_0, 2)  # Code: 
                # snr_0 =  A_0 / sigma_0 # Paper
                snr_0 = A_0/ns.lpf(sigma_0)

 
                # A_0_rice, sigma_0_rice = ns.expectation_maximization(img_slice, lm_method='average', lm_kernel=3)
                # snr_0_rice = A_0_rice/ns.lpf(sigma_0_rice , 1.2)
                # sigma_0 = ns.hmf_r(img_slice, snr_0_rice , lm_method='value', lm_value=A_0_rice)
                sigma_0 = ns.hmf_r(img_slice, snr_0)


                img_0 = ns.vst(img_slice, sigma_0, snr_0)
                sigma = ns.hmf_g(img_0)

                print(f"Slice {time_slice_i} Init: {np.mean(sigma_0):.2f} After: {np.mean(sigma):.2f}")

                # print(f"Slice {time_slice_i} VST+HMF_G: {np.mean(sigma)} ")
            elif method == "background":
                crop = self.get_crop('background', self.path_root)
                bg = self._apply_crop(img_slice, crop)
                img_0 = img_slice
                sigma = np.sum(bg) / (bg.size) * np.sqrt(2 / np.pi)
                sigma = np.ones(img_slice.shape) * sigma

            # Correction
            img_cor_slice = ns.lmmse(img_0, sigma)

            # img_cor_slice = img_0
            # sigma_n = sigma
            # for _ in range(10):
            #     img_cor_slice = ns.lmmse(img_cor_slice, sigma_n)
            #     bg = self._apply_crop(img_cor_slice, crop)
            #     sigma_n = np.sum(bg) / (bg.size) * np.sqrt(2 / np.pi)
            #     sigma_n = np.ones(img_cor_slice.shape) * sigma_n
            img_cor[:, :, time_slice_i] = img_cor_slice
            sigma_maps[:, :, time_slice_i] = sigma

        return sigma_maps, img_cor

    def _img2singal(self, image, crop=None, norm=True):
        # Apply crop
        if crop is not None:
            image = self._apply_crop(image, crop)

        # Transform image (2D) into measurement series (1D)
        signalpoints = image.reshape(-1, image.shape[-1]).astype(np.float32)  # [pixels, timepoints]

        # Scale each series such that maximum is 1
        if norm:
            signal_scale = signalpoints[:, 0]  # np.max(signalpoints, axis=-1).astype(np.float32)
            signalpoints = self.norm_signalpoints(signalpoints, signal_scale)
        else:
            signal_scale = np.ones(signalpoints.shape[0])

        return signalpoints, signal_scale

    @classmethod
    def _apply_crop(cls, image, crop):
        if isinstance(crop, tuple) or isinstance(crop, slice):
            return image[crop]
        elif isinstance(crop, np.ndarray):
            if image.ndim == 2:
                return image[crop[:, :, 0]]
            else:
                return np.stack([image[:, :, sli][crop[:, :, sli]] for sli in range(image.shape[-1])], axis=-1)

    def __len__(self):
        return len(self.signalpoints)

    def __getitem__(self, idx):
        return {"timepoints": self.timepoints_array[idx],  # Time Echo (TE)
                'signalpoints': self.signalpoints[idx],
                'signalpoints_ref': self.signalpoints_ref[idx],
                "signal_scale": self.signal_scale[idx],
                "signal_scale_ref": self.signal_scale_ref[idx]}

    def model_func(self, x, initial_signal, relaxation_time):
        return self.signal_func(x, initial_signal, relaxation_time)

    def model_func_bias(self, x, initial_signal, relaxation_time, bias):
        return (self.signal_func(x, initial_signal, relaxation_time).T + bias).T

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
        """ Jacobian for Noise Corrected Exponential (NCEXP) / Free Induction Decay """
        eps = 1e-8
        initial_signal = initial_signal + eps
        sigma = sigma + eps

        alpha = (initial_signal / (2 * sigma))**2
        expo = np.exp(-x / relaxation_time)

        def calc_partial_s(alpha):
            if alpha < 8:
                partial_s = np.sqrt(np.pi * alpha / 2) * np.exp(-alpha) * (sc.i0(alpha) + sc.i1(alpha))
            else:
                partial_s = 1 - 1 / (8 * alpha)
            return partial_s

        calc_partial_s = np.vectorize(calc_partial_s)
        partial_s = calc_partial_s(alpha)

        partial_t2 = partial_s * initial_signal * x / relaxation_time**2 * expo
        partial_s0 = partial_s * expo
        return np.asarray([partial_s0, partial_t2]).T

    @property
    def model_func_str(self):
        return r"$S_0 \cdot exp(-t/T2)$"

    @classmethod
    def params_str(cls):
        return r"$S_0={:.2f}\; T2={:.2f}$"

    @classmethod
    def params_dict(cls, method="lse"):
        if method=="lse-unbiased":
            return {'a': r'$\mathbf{S_0}$', 'tau': r'$\mathbf{T2}$', 'offset':r'$\mathbf{o}$'}
        else:
            return {'a': r'$\mathbf{S_0}$', 'tau': r'$\mathbf{T2}$'}

    @classmethod
    def params_amplitude_scale(cls, params, scale):
        params[:, 0] = params[:, 0] * scale
        return params

    @classmethod
    def params_amplitude_replace(cls, params, new=1):
        if params.ndim == 1:
            params[0] = new
        else:
            params[:, 0] = new
        return params

    @classmethod
    def signal_func(cls, x, initial_signal, relaxation_time):
        if isinstance(x, torch.Tensor):
            return (initial_signal * torch.exp(-x.T / relaxation_time)).T
        else:
            x = np.asarray(x)
            return (initial_signal * np.exp(-x.T / relaxation_time)).T

    @staticmethod
    def read_dicom(path_dir, desired_protocol_name):
        slices = []
        for f in path_dir.rglob('IM_*'):
            if not f.is_file():
                continue
            dicom_obj = dcmread(f)

            protocol_name = getattr(dicom_obj, "ProtocolName", None)
            if protocol_name != desired_protocol_name:
                continue
            if not hasattr(dicom_obj, "EchoTime"):
                continue
            slices.append(dicom_obj)

        # Sort Slices
        slices = sorted(slices, key=lambda s: s.EchoTime)
        # slices = slices[0:5]  # Use only first x Echos
        slices = slices[0:7] 
        # slices = slices[0:10] 

        print("""Protocol {},
              Slices {},
              Echo Time {}ms,
              Repetition Time {}ms
              Number of Averages {}
              Acquisition Time {}s""".format(
            slices[0].ProtocolName,
            len(slices),
            slices[0].EchoTime,
            slices[0].RepetitionTime,
            slices[0].NumberOfAverages,
            slices[0].AcquisitionDuration
        ))

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(slices):
            img3d[:, :, i] = s.pixel_array

        return img3d, np.asarray([sli.EchoTime for sli in slices], dtype=np.float32)

    def load_paras(self, protocol='dataset', norm='dataset', crop='dataset', method="lse", noise_suppressed=False):
        path_image = Path.cwd() / 'results/images/' / str(self.knee)
        if protocol == "dataset":
            protocol = self.protocol

        if norm == "dataset":
            norm = self.norm

        if crop == "dataset":
            crop = self.crop

        if noise_suppressed:
            noise_mode = "_noise_cancel"
        else:
            noise_mode = ""

        file_tau = method + '_' + protocol + '_tau_' + str(norm) + noise_mode + '.nii'
        file_A = method + '_' + protocol + '_a_' + str(norm) + noise_mode + '.nii'
        file_o = method + '_' + protocol + '_offset_' + str(norm) + noise_mode + '.nii' 

        tau = sitk.ReadImage(str(path_image / file_tau), sitk.sitkFloat32)
        tau_array = sitk.GetArrayFromImage(tau)
        tau_array = self._apply_crop(tau_array, crop=crop)

        A = sitk.ReadImage(str(path_image / file_A), sitk.sitkFloat32)
        A_array = sitk.GetArrayFromImage(A)
        A_array = self._apply_crop(A_array, crop=crop)

        if method == "lse-unbiased":
            o = sitk.ReadImage(str(path_image / file_o), sitk.sitkFloat32)
            o_array = sitk.GetArrayFromImage(o)
            o_array = self._apply_crop(o_array, crop=crop)

        if method == "lse-unbiased":
            return np.stack([A_array, tau_array, o_array], axis=-1)
        else:
            return np.stack([A_array, tau_array], axis=-1)

    def load_signalpoints(self, timepoints_array=None, protocol='dataset', norm='dataset', crop='dataset', method="lse"):
        paras = self.load_paras(protocol=protocol, norm=norm, crop=crop, method=method)
        A = np.take(paras, 0, axis=-1).flatten()
        tau = np.take(paras, 1, axis=-1).flatten()
        timepoints_array = self.timepoints_array if timepoints_array is None else timepoints_array
        if method == "lse-unbiased":
            offset = np.take(paras, 2, axis=-1).flatten()
            return self.model_func_bias(timepoints_array, A.flatten(), tau.flatten(), offset.flatten())
        else:
            return self.model_func(timepoints_array, A.flatten(), tau.flatten())

    def comp_lse_paras(self, signalpoints=None, method="lse", noise=None, skip=False, **kwargs):
        self._lse_fitting_method = method
        self._lse_fitting_kwargs = kwargs
        self._lse_fitting_skip = skip
        signalpoints = self.signalpoints if signalpoints is None else signalpoints
        noise = self.noisepoints if noise is None else noise

        signalpoints = signalpoints.astype(np.float64)
        noise = noise.astype(np.float64)
        if skip:
            signalpoints = signalpoints[:, 1:]
        processes = min(cpu_count(), len(signalpoints))
        print("Using {} Processor".format(processes))
        with get_context().Pool(processes=processes) as pool:
            cor_vec = pool.starmap(self._lse_fitting, zip(signalpoints, noise),
                                   chunksize=len(signalpoints) // processes)
        return np.asarray(cor_vec)  # [Singalpoints, Parameters]

    def _lse_fitting(self, signalpoint_series, noise_series):
        if self._lse_fitting_skip:
            timepoints = self.timepoints[1:]
        else:
            timepoints = self.timepoints

        try:
            if self._lse_fitting_method == "lse":
                opt_para = curve_fit(self.model_func, timepoints, signalpoint_series,
                                     bounds=self.lse_bounds, jac=self.jac_model_func, method='trf', loss='linear',
                                     p0=[250, 50], sigma=None, f_scale=1, max_nfev=None)
            # elif self._lse_fitting_method == "lse-robust":
            #     opt_para = curve_fit(self.model_func, timepoints, signalpoint_series,
            #                          bounds=self.lse_bounds, jac="3-point", method='trf', loss='huber',
            #                          p0=[250, 50], sigma=None, f_scale=10, max_nfev=None)
            elif self._lse_fitting_method == "lse-unbiased":
                opt_para = curve_fit(self.model_func_bias, timepoints, signalpoint_series,
                                     method='trf', # jac="3-point", 
                                     loss='linear', bounds=[[0, 5, 0], [2500, 500, 2500]], 
                                     jac=self.jac_model_func_bias,
                                     p0=[250, 50, 0], sigma=None, f_scale=1, max_nfev=None
                                     )
                opt_para = [opt_para[0][0:2], opt_para[1][0:2]] if not self._lse_fitting_kwargs.get(
                    'all_variables', False) else opt_para
            elif self._lse_fitting_method == "lse-noise_corrected":
                sigma = np.median(noise_series)# noise_series[-1] # Compute Median along temporal axis not spatial !

                def func_wrapper(x, initial_signal, relaxation_time):
                    return ns.noise_corrected_exp(self.model_func(x, initial_signal, relaxation_time), sigma)

                def jac_wrapper(x, initial_signal, relaxation_time):
                    return self.jac_model_func_cor(x, initial_signal, relaxation_time, sigma)

                opt_para = curve_fit(func_wrapper, timepoints, signalpoint_series,
                                     bounds=self.lse_bounds, jac=jac_wrapper, method='trf', loss='linear',
                                     p0=[250, 50], sigma=None, f_scale=1, max_nfev=None)
            return opt_para[0]

            # return opt_para.x
        except RuntimeError:
            # return [float('nan')] * len(self.params_dict()) # NaN will be set to 0
            # by SimpleITK - not correct for tau > 0
            if self._lse_fitting_method == "lse-unbiased":
                return [0, 5] if not self._lse_fitting_kwargs.get('all_variables', False) else [0, 5, 0]
            else:
                return self.lse_bounds[0]  # set (arbitrary) to lower bound

    def comp_nn_paras(self, model, dl=None, dict_key='params', protocol="dataset", skip=False, device='cpu'):
        if protocol == "dataset":
            ds = self
        else:
            ds = RealExpDecayDataset(
                path_root=self.path_root,
                protocol=protocol,
                norm=self.norm,
                crop=self.crop_name)
        if dl is None:
            from torch.utils.data.dataloader import DataLoader
            dl = DataLoader(ds, batch_size=len(self), num_workers=cpu_count(), shuffle=False)

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
        model_nn.to(device)
        if skip:
            params_pred_nns = torch.cat([model_nn(batch["signalpoints"][:, 1:].to(device), batch["timepoints"][:, 1:].to(device))[
                                        dict_key] for batch in dl]).cpu().detach().numpy()
        else:
            start_time = time.time()
            params_pred_nns = torch.cat([model_nn(batch["signalpoints"].to(device), batch["timepoints"].to(device))[
                                        dict_key] for batch in dl]).cpu().detach().numpy()
            self.time_comp_nn = time.time() - start_time
        return params_pred_nns

    @classmethod
    def norm_signalpoints(cls, signalpoints, signal_scale):
        where = (signal_scale != 0) & ~np.isnan(signal_scale) & ~np.isinf(signal_scale)
        return np.divide(signalpoints.T, signal_scale, out=signalpoints.T, where=where).T

    @classmethod
    def get_crop(cls, crop, path_root=None, knee=1):
        path_root = cls.path_root_default[knee] if path_root is None else path_root
        if crop == "tissue":
            tissue_mask = sitk.ReadImage(str(path_root.parent / 'seg_tissue.nii.gz'))
            tissue_mask = np.where(sitk.GetArrayFromImage(tissue_mask) == 1, True, False)
            crop = np.repeat(np.expand_dims(tissue_mask[0, :, :], -1), tissue_mask.shape[0], axis=-1)
        elif crop == "background":
            tissue_mask = sitk.ReadImage(str(path_root.parent / 'seg_background.nii.gz'))
            tissue_mask = np.where(sitk.GetArrayFromImage(tissue_mask) == 2, True, False)
            crop = np.repeat(np.expand_dims(tissue_mask[0, :, :], -1), tissue_mask.shape[0], axis=-1)
        elif crop == "knee":
            tissue_mask = sitk.ReadImage(str(path_root.parent / 'seg_knee.nii.gz'))
            tissue_mask = np.where(sitk.GetArrayFromImage(tissue_mask) > 0, True, False)
            crop = np.repeat(np.expand_dims(tissue_mask[0, :, :], -1), tissue_mask.shape[0], axis=-1)
        elif crop == "knee_core":
            tissue_mask = sitk.ReadImage(str(path_root.parent / 'seg_knee.nii.gz'))
            tissue_mask = np.where(sitk.GetArrayFromImage(tissue_mask) == 1, True, False)  # [timepoints, x,y]
            crop = np.repeat(np.expand_dims(tissue_mask[0, :, :], -1), tissue_mask.shape[0], axis=-1)
        else:
            raise ValueError("Unknown value for crop {}".format(crop))
        return crop







if __name__ == "__main__":
    import numpy as np
    from scipy.optimize import curve_fit
    import SimpleITK as sitk
    from fitting.models import FCN
    import torch
    from torch.utils.data.dataloader import DataLoader

    # ds = RealExpDecayDataset(3, protocol='highSNR', crop=None, norm=False, noise_est_method="background")
    # v = ds.jac_model_func_bias(ds.signalpoints[10000], 100, 50, 2)
    # print(v)
    # v = ds.jac_model_func_cor(ds.signalpoints[10000], 100, 50, 2)
    # print(v)

    for knee in [1,2,3,4,5,6,7]: #[1,2,3,4,5,6,7]: # 1,2,3,4,5,6,7
        for protocol in ['lowSNR',  'highSNR' ]: # 'lowSNR',  'highSNR'
            path_out = Path.cwd() / 'results/images' / str(knee)
            path_out.mkdir(parents=True, exist_ok=True)

            # ---------- Save Image ---------------
            # ds = RealExpDecayDataset(knee, protocol=protocol, crop=None)
            # image_nii = sitk.GetImageFromArray(np.moveaxis(ds.image, -1, 0))
            # sitk.WriteImage(image_nii, str(path_out / (ds.protocol + '.nii')))

            
            # ---------- Compute LSE-Image ---------------
            ds = RealExpDecayDataset(knee, protocol=protocol, crop=None, norm=False, noise_est_method="vst-hmfg")
            # method = "lse-unbiased" # lse or lse-robust, lse-unbiased, lse-noise_corrected
            # for method in ["lse", "lse-unbiased", "lse-noise_corrected"]:
            for method in ["lse-noise_corrected"]:
                lse_paras = ds.comp_lse_paras(method=method, all_variables=True)
                for para_n, para_name in enumerate(ds.params_dict(method=method).keys()):
                    image_array = lse_paras[:, para_n].reshape(ds.image.shape[0:-1])
                    image_nii = sitk.GetImageFromArray(image_array)
                    sitk.WriteImage(image_nii, str(path_out / (method + '_' + ds.protocol +
                                                            '_' + para_name + '_' + str(ds.norm) + '.nii')))


            # ---------- Compute NN-Image ---------------
            # ds = RealExpDecayDataset(knee, protocol=protocol, crop=None, norm=False)
            # dl = DataLoader(ds, batch_size=512 * 512 * 10, num_workers=0, shuffle=False)
            # path_run = Path.cwd() / 'runs/2021_03_10_202304'  # runs/2021_03_10_202304 , runs/2021_05_12_203207
            # model_nn = FCN.load_best_checkpoint(str(path_run))
            # model_nn.eval()
            # model_nn.model_func = ds.model_func
            # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # # model_nn.to(device)
    

            # params_pred_nns = torch.cat([model_nn(batch["signalpoints"], batch["timepoints"])['params']
            #                             for batch in dl]).detach().numpy()

            # for para_n, para_name in enumerate(ds.params_dict().keys()):
            #     image_array = params_pred_nns[:, para_n].reshape(ds.image.shape[0:-1])
            #     image_nii = sitk.GetImageFromArray(image_array)
            #     sitk.WriteImage(image_nii, str(path_out / ("nn_" + ds.protocol +
            #                                             '_' + para_name + '_' + str(ds.norm) + '.nii')))






   
