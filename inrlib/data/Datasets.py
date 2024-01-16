import os, gdown
from os import path
from abc import ABC, abstractmethod
import numpy as np
from numpy import random
from typing import List, Literal
import torch
from torch import nn
from torch.utils.data import Dataset
from phantominator import ct_shepp_logan, ct_modified_shepp_logan_params_3d

from inrlib.utils.imaging import resize, subsampling_mask, get_coordinates
from inrlib.utils import make_complex
from . import ABCDataset


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
   
####################################################################################################
# @title ATLAS Data Gen
def get_atlas_dataset_3D(num_grid_search_samples, test_samples, RES=96):
    total_samples = num_grid_search_samples + test_samples

    id = '1SLejANPHTA_eSJhIjCk9WGeFsKSEMZMx'
    filename = 'atlas_3d.npz'
    if not os.path.exists(filename):
        gdown.download(id=id, output=filename, quiet=False)
    data = np.load(filename)['data']/255.0

    scan_ids = [0, 1, 4, 7, 9, 11, 14, 16, 18, 20, 23, 24, 28]
    samples = resize(data[scan_ids, ...], (len(scan_ids), RES, RES, RES))
    new_samples = random.permutation(samples)

    out = {
        "data_grid_search": np.array(new_samples[:num_grid_search_samples, :, :]),
        "data_test": np.array(new_samples[num_grid_search_samples:, :, :]),
    }
    return out


# @title Shepp Data Gen
def get_shepp_dataset_3D(num_grid_search_samples, test_samples, RES=96):
    total_samples = num_grid_search_samples + test_samples

    ct_params = np.array(ct_modified_shepp_logan_params_3d())

    shepps = []
    for i in range(total_samples):
        i_ct_params = ct_params + random.normal(size=ct_params.shape)/20.0
        shepps.append(np.clip(ct_shepp_logan((RES,RES,RES), E=i_ct_params, zlims=(-0.25,0.25)), 0.0, 1.0))

    samples = np.stack(shepps, axis=0)

    out = {
        "data_grid_search":np.array(samples[:num_grid_search_samples,:,:]),
        "data_test":np.array(samples[num_grid_search_samples:,:,:]),
    }
    return out


class MRI3DDataset(ABCDataset):
    """
    3D MRI dataset based on https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
    """
    def __init__(self, 
          RES=96, 
          sparsity=0.12, 
          train=True, 
          shepp_or_atlas='shepp'):

        self.RES = RES
        self.stage = train

        assert shepp_or_atlas in ['shepp', 'atlas']
        fn = get_atlas_dataset_3D if shepp_or_atlas == 'atlas' else get_shepp_dataset_3D
        
        dataset = fn(1, 1, RES)
        images = dataset["data_grid_search"] if train else dataset["data_test"]
        self.image = images[0, ...]

        # self.y_train = fft(self.image)  # (RES, RES, RES)
        self.y_data = self.image

        self.mask = subsampling_mask(images[0].shape, int(np.prod(self.image.shape)*sparsity))
        
        self.input_shape = self.image.shape
        self.x_train, self.x_test = get_coordinates(self.input_shape)  # (RES, RES, RES, 3)
        
        self.x_data = self.x_train if train else self.x_test
        self.x_data = self.x_data.reshape(-1,self.x_data.shape[-1])
        self.y_data = self.y_data.reshape(-1, 1)

    def change_stage(self, train: bool): 
        self.stage = train
        self.x_data = self.x_train if train else self.x_test
        self.x_data = self.x_data.reshape(-1,self.x_data.shape[-1])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
  
        x = torch.tensor(x).float() # real coords
        y = torch.tensor(y if y is np.ndarray else np.array(y)).float() # real or complex data

        return {'x': x, 'y': y, 'idx': idx}

