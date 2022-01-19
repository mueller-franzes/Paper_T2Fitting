from pathlib import Path
import json
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from multiprocessing import cpu_count

from fitting.utils import AdvJsonEncoder




class BaseDataModule(pl.LightningDataModule):
    Dataset = torch.utils.data.Dataset

    def __init__(self,
                 path_checkpoint_dir: str = None,
                 samples_train: int = 2**26,  # 2**26,
                 samples_val: int = 2**23,  # 2**23,
                 samples_test: int = 2**14,
                 batch_size: int = 1024,  # 2**12
                 num_workers: int = cpu_count(),
                 **kwargs):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.path_checkpoint_dir = path_checkpoint_dir

        self.samples_train = samples_train
        self.samples_val = samples_val
        self.samples_test = samples_test

        self.seed_train = 0
        self.seed_val = samples_train
        self.seed_test = samples_train + samples_val

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.kwargs = kwargs

    def save(self):
        with open(Path(self.path_checkpoint_dir) / 'data_conf.json', 'w') as f:
            json.dump(self.hyperparameters, f, cls=AdvJsonEncoder)

    @classmethod
    def params_dict(cls):
        return cls.Dataset.params_dict()

    @classmethod
    def load(cls, path_checkpoint_dir):
        with open(Path(path_checkpoint_dir / 'data_conf.json'), 'r') as f:
            hyperparameters = json.load(f)
        return cls(**hyperparameters)

    def setup(self, stage=None, mkdir=True):
        if stage == 'fit' or stage is None:
            if mkdir:
                Path(self.path_checkpoint_dir).mkdir(parents=True, exist_ok=True)

            params = self.kwargs.copy()
            params.update({'samples': self.samples_train, 'seed': self.seed_train})
            self.ds_train = self.Dataset(**params)

            params = self.kwargs.copy()
            params.update({'samples': self.samples_val, 'seed': self.seed_val})
            self.ds_val = self.Dataset(**params)

        if stage == 'test' or stage is None:
            params = self.kwargs.copy()
            params.update({'samples': self.samples_test, 'seed': self.seed_test})
            self.ds_test = self.Dataset(**params)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def dataloader(self, mode):
        if mode == "train":
            return self.train_dataloader()
        elif mode == "val":
            return self.val_dataloader()
        elif mode == "test":
            return self.test_dataloader()
        else:
            raise ValueError("Unknown mode {}".format(mode))


class FileDataModule(BaseDataModule):
    Dataset = torch.utils.data.Dataset

    def __init__(self,
                 path_checkpoint_dir: str = None,
                 batch_size: int = 4096,
                 num_workers: int = 0,
                 **kwargs):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.path_checkpoint_dir = path_checkpoint_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.kwargs = kwargs

    def setup(self, stage=None, mkdir=True):
        if stage == 'fit' or stage is None:
            if mkdir:
                Path(self.path_checkpoint_dir).mkdir(parents=True, exist_ok=True)

            params = self.kwargs.copy()
            params.update({'mode': 'train'})
            self.ds_train = self.Dataset(**params)

            params = self.kwargs.copy()
            params.update({'mode': 'val'})
            self.ds_val = self.Dataset(**params)

        if stage == 'test' or stage is None:
            params = self.kwargs.copy()
            params.update({'mode': 'test'})
            self.ds_test = self.Dataset(**params)
