import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from typing import Optional
from copy import deepcopy

from .Datasets import WrappedDataset, ABCDataset


def worker_init_fn(_):
    """
    For multi-process dataloading
    
    https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    """
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train: ABCDataset, 
                 val: Optional[ABCDataset] = None, 
                 test: Optional[ABCDataset]=None, 
                 predict: Optional[ABCDataset]=None, 
                 batch_size: Optional[int]=None, 
                 wrap=False, 
                 num_workers=None, 
                 use_worker_init_fn=False,
                 shuffle_train_loader=True, 
                 shuffle_val_loader=False, 
                 shuffle_test_loader=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.shuffle_train_loader = shuffle_train_loader
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_loader = shuffle_val_loader
        
        self.datasets = {'fit': train, 'validate': val, 'test': test, 'predict': predict}

    def setup(self, stage: str):
        if self.wrap:
            for k in self.datasets:
                if self.datasets[k] is not None:
                    self.datasets[k] = WrappedDataset(self.datasets[k])

        if stage == 'fit':
            if self.datasets['fit'] is None:
                raise ValueError("no train dataset provided")
            elif self.datasets['validate'] is None:
                val = deepcopy(self.datasets['fit'])
                val.change_stage(train=False)
                self.datasets['validate'] = val
        
        if stage == 'test':
            if self.datasets['test'] is None:
                raise ValueError("no test dataset provided")
        
        if stage == 'predict':
            if self.datasets['predict'] is None:
                raise ValueError("no predict dataset provided")

    def train_dataloader(self):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        bs = self.batch_size if self.batch_size is not None else len(self.datasets["fit"])
        return DataLoader(self.datasets["fit"], batch_size=bs,
                          num_workers=self.num_workers, shuffle=self.shuffle_train_loader,
                          worker_init_fn=init_fn)

    def val_dataloader(self):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        bs = self.batch_size if self.batch_size is not None else len(self.datasets["validate"])
        return DataLoader(self.datasets["validate"],
                          batch_size=bs,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=self.shuffle_val_loader)

    def test_dataloader(self):
        init_fn = worker_init_fn if self.use_worker_init_fn else None 
        bs = self.batch_size if self.batch_size is not None else len(self.datasets["test"])
        return DataLoader(self.datasets["test"], batch_size=bs,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=self.shuffle_test_loader)

    def predict_dataloader(self):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        bs = self.batch_size if self.batch_size is not None else len(self.datasets["predict"])
        return DataLoader(self.datasets["predict"], batch_size=bs,
                          num_workers=self.num_workers, worker_init_fn=init_fn)
