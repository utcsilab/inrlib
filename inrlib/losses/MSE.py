import numpy as np
import torch
from torch import nn
from ..losses import ABCLoss
from ..utils.numeric import make_complex
from typing import Mapping, Tuple


def mse(x: torch.Tensor, dim: Tuple[int,...]=(0,1)) -> torch.Tensor:
	N = np.prod([x.shape[d] for d in dim])
	return torch.sum(torch.abs(x)**2) / N


class MSELoss(nn.Module): 
    def __init__(self, dim: Tuple[int,...]=(0,1), **kwargs):
        super().__init__()
        self.dim = dim
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return mse(pred - target, dim=self.dim)
    

class NRMSE:
    def __init__(self, dim: Tuple[int, ...] = (0,1)):
        self.dim = dim
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
            return np.sqrt(np.mean(np.square(np.abs(pred-target)), axis=self.dim)) / np.sqrt(np.mean(np.square(np.abs(target)), axis=self.dim))


class NRMSELoss(nn.Module):
    def __init__(self, dim: Tuple[int,...]=(0,1)):
        super().__init__()
        self.dim = dim
        self.mse = MSELoss(dim=dim)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred,target)) / torch.sqrt(mse(target, dim=self.dim))
