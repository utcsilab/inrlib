import numpy as np
import torch
from torch import nn
from ..losses import ABCLoss
from ..utils.numeric import make_complex
from typing import Mapping

MSELoss = nn.MSELoss

class NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred,target)) / torch.sqrt(torch.mean(torch.square(target)))


## COMPLEX

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
       return (0.5*(pred - target)**2).mean(dtype=torch.complex64) 
   

class ComplexNRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = ComplexMSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred,target)) / torch.sqrt(torch.mean(torch.square(target)))


class Real2ComplexMSELoss(ABCLoss):
    def set_params(self, *args, **kwargs):
        '''
        Set additional optimization parameters for loss, i.e., clean reconstruction
        '''
        return
    
    def reconstruct_params(self, image_shape, outputs, **other) -> Mapping[str, np.ndarray]:
        '''
        If added additional parameters, reconstruct them here 
        '''
        return {**other}
    
    def prepare_input(self,  x: torch.Tensor, y: torch.Tensor, **other) -> Mapping[str, torch.Tensor]:
        xi = x.clone()
        yi = make_complex(y)
        yi = torch.view_as_real(yi)
        return {'x': xi, 'y': yi, **other}
    
    def prepare_output(self,  y_hat: torch.Tensor, y: torch.Tensor, **other) -> Mapping[str, torch.Tensor]:
        return {'pred': y_hat, 'target': y, **other}
