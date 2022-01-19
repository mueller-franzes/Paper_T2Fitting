from pathlib import Path

import json
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()

    def forward(self, x):
        return F.layer_norm(x, x.shape[1:])


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            norm_op=nn.Identity,
            drop_op=nn.Identity,
            acti_op=nn.LeakyReLU,
            num_hid=0,
            skipcon="add"):
        super(DenseBlock, self).__init__()
        self.skipcon = skipcon
        self.norm_op = norm_op
        acti_kwargs = {'negative_slope': 0.1} if isinstance(acti_op, nn.LeakyReLU) else {}
        hid_ch = max(in_ch, out_ch) * 2

        if "GroupNorm" in str(norm_op.__name__):
            norm_hid_kwargs = {'num_groups': max(hid_ch // 4, 1), 'num_channels': hid_ch}
            norm_out_kwargs = {'num_groups': max(out_ch // 4, 1), 'num_channels': out_ch}
        elif "BatchNorm" in str(norm_op.__name__):
            norm_hid_kwargs = {'num_features': hid_ch}
            norm_out_kwargs = {'num_features': out_ch}
        else:
            norm_hid_kwargs = {}
            norm_out_kwargs = {}

        if num_hid == 0:
            self.denseblock = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=True),
                norm_op(**norm_out_kwargs),
                drop_op(p=0.2),
                acti_op(**acti_kwargs))
        elif num_hid == 1:
            self.denseblock = nn.Sequential(
                nn.Linear(in_ch, hid_ch, bias=True),
                norm_op(**norm_hid_kwargs),
                acti_op(),
                nn.Linear(hid_ch, out_ch, bias=True),
                norm_op(**norm_out_kwargs),
                acti_op(**acti_kwargs))

        if skipcon == "cat":
            self.skipblock = nn.Sequential(
                nn.Linear(in_ch + out_ch, out_ch, bias=True),
                acti_op(**acti_kwargs)
            )
        elif skipcon == "add":
            self.skipblock = nn.Sequential(
                acti_op(**acti_kwargs)
            )

    def forward(self, x):
        x2 = self.denseblock(x)

        if self.skipcon == "cat":
            x3 = torch.cat([x, x2], dim=1)
            return self.skipblock(x3)
        elif self.skipcon == "add" and x2.shape == x.shape:
            x3 = x + x2
            return self.skipblock(x3)
        else:
            return x2




class FCN(pl.LightningModule):
    """Fully Connected NeuralNet"""

    def __init__(self, n_signalpoints, n_params):
        super().__init__()
        self.save_hyperparameters()

        self.build_model()
        self.model_func = None  # WORKAROUND: If given as argument Lightning freezes in save_hparams_to_yaml()

        self.noise_losses = {"Huber": torch.nn.SmoothL1Loss(beta=1)}
        self.param_losses = {"Huber": torch.nn.SmoothL1Loss(beta=1)}

        self.param_metrics = {
            "MSE": torch.nn.MSELoss(),
            "L1": torch.nn.L1Loss(),
            "Huber": torch.nn.SmoothL1Loss(
                beta=1)}
        self.signal_metrics = {"L1": torch.nn.L1Loss()}


        self.step_val = 0
        self.step_train = 0
        self.step_test = 0

    def build_model(self):
        n_blocks = 2
        i_0 = 512
        b = 1

     
        inp_nodes = self.hparams["n_signalpoints"] * 2
        out_noes = self.hparams["n_params"] + 1  # + self.hparams["n_signalpoints"] * 1

        encoder_0 = DenseBlock(inp_nodes, i_0, drop_op=nn.Identity, acti_op=nn.LeakyReLU)
        self.encoders = nn.ModuleList([encoder_0] + [DenseBlock(i_0 * b**e, i_0 * b**(e + 1))
                                                     for e in range(0, n_blocks)])

        decoder_n = DenseBlock(
            i_0,
            out_noes,
            drop_op=nn.Identity,
            norm_op=nn.Identity,
            acti_op=nn.ReLU)  # acti_op=nn.LeakyReLU,
        self.decoders = nn.ModuleList([DenseBlock(i_0 * b**e, i_0 * b**(e - 1))
                                       for e in range(n_blocks, 0, -1)] + [decoder_n])

 

    def forward(self, signalpoints, timepoints):
        batch_size, num_point = signalpoints.shape

        num_point_norm = 15
        singalpoints_norm = -1 * torch.ones((batch_size, num_point_norm), device=signalpoints.device)
        timepoints_norm = -1 * torch.ones((batch_size, num_point_norm), device=timepoints.device)

        singalpoints_norm[:, 0:num_point] = signalpoints
        timepoints_norm[:, 0:num_point] = timepoints
        singalpoints_norm = singalpoints_norm / 500
        timepoints_norm = timepoints_norm / 120  # 240/2

        x = torch.cat([singalpoints_norm, timepoints_norm], dim=1)
        x_down = [x]
        for n, (encoder, decoder) in enumerate(zip(self.encoders, reversed(self.decoders))):
            x_down.append(encoder(x_down[-1]))

        x_temp = x_down.pop()
        for n, (decoder, x_skip) in enumerate(zip(self.decoders, reversed(x_down))):
            x_temp = decoder(x_temp)
    
        n_par = self.hparams["n_params"]
        n_sig = self.hparams["n_signalpoints"]

        return {'params': x_temp[:, 0:n_par], 'sigma': x_temp[:, n_par:n_par + 1]}  # x_temp[:, n_par:n_par+1]




    def _step(self, batch: dict, batch_idx: int, prefix: str, step: int):
        timepoints, signalpoints, params_true, sigma_true = batch["timepoints"], batch["signalpoints"], batch["parameters"], batch['sigma']

        # [Batch, Parameters], True, if and only if parameters (A, tau) can be estimated from signal, e.g. tau can't be estimated if A=0 as 0 = 0 *exp(-t/tau)
        estimateable_A = torch.ones(params_true.shape[0], device=params_true.device, dtype=bool)
        estimateable_tau = params_true[:, 0] > 0
        estimateable = torch.stack([estimateable_A, estimateable_tau], dim=-1)

        # Add Dim
        sigma_true = sigma_true[:, None]  # [Batch, 1]

        results = self(signalpoints, timepoints)
        params_pred = results.get('params')
        sigma_pred = results.get('sigma')
  

        # ------------ Compute losses ---------------
        losses_params = {
            loss_name + "_para:" + str(para_name):
            loss_func(para_pred[mask], para_true[mask])
            for loss_name, loss_func in self.param_losses.items()
            for para_name, (para_pred, para_true, mask) in enumerate(zip(params_pred.T, params_true.T, estimateable.T))
        }
 
        loss_noise = {
            loss_name + "_sigma:": loss_func(sigma_pred, sigma_true)
            for loss_name, loss_func in self.noise_losses.items()
        }

        loss = torch.stack(
            list(losses_params.values()) +
            list(loss_noise.values()) 
        ).sum()  # mean()

        # ---------- Compute metrics -------------
        metrics_params = {
            metric_name + "_para:" + str(para_name): metric_func(para_pred, para_true)
            for metric_name, metric_func in self.param_metrics.items()
            for para_name, (para_pred, para_true) in enumerate(zip(params_pred.T, params_true.T))
        }

        metrics_signal = {
            metric_name + "_signal-" + true_key: metric_func(
                self.model_func(timepoints, *params_pred.T),
                true_val
            )
            for metric_name, metric_func in self.signal_metrics.items()
            for true_key, true_val in {
                # 'est': signalpoints - noise_pred,
                'func': self.model_func(timepoints, *params_true.T)
            }.items()
        }

        results = metrics_params
        results.update(**metrics_signal)
        results.update(**losses_params)
        results.update(**loss_noise)



        results.update({'loss': loss})

        # logging
        for k, v in results.items():
            self.logger.experiment.add_scalar("{}/step_{}".format(prefix, k), v, step)

        self.log(prefix + "/loss", loss)

        return results

    def training_step(self, batch: dict, batch_idx: int):
        self.step_train = self.global_step
        return self._step(batch, batch_idx, "Train", self.step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self.step_val += 1
        return self._step(batch, batch_idx, "Val", self.step_val)

    def test_step(self, batch: dict, batch_idx: int):
        self.step_test += 1
        return self._step(batch, batch_idx, "Test", self.step_test)

    def _epoch_end(self, outputs: list, prefix: str) -> dict:
        """
        Template function for epoch end.
        Args:
            outputs: the outputs (from '*_step' function) from each iteration
            prefix: the prefix for logging ('Train', 'Val', 'Test')
        Returns:
            dict: Mean over all epochs
        """

        # Aggregate
        results = {metric_name: [] for metric_name in outputs[0].keys()}
        for out_iter in outputs:
            for metric_name, metric_val in out_iter.items():
                results[metric_name].append(metric_val)

        # Mean
        for metric_name, metric_values in results.items():
            results[metric_name] = torch.mean(torch.as_tensor(metric_values))

        # Logging
        for k, v in results.items():
            self.logger.experiment.add_scalar("{}/epoch_{}".format(prefix, k), v, self.current_epoch)

        return results

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "Train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "Val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "Test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-3)
        # return torch.optim.SGD(self.parameters(), 1e-5, momentum=0.1, nesterov=True)

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_path': best_model_path}, f)

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_best_checkpoint = json.load(f)['best_model_path']
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)
