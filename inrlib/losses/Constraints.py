import torch
import torch.nn as nn

from ..utils.imaging import make_complex

class ComplexRealConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.real if x.is_complex() else x[..., 0]
        return x.flatten()
   
    
class ComplexImaginaryConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.imag if x.is_complex() else x[..., -1]
        return x.flatten()


class ComplexMagnitudeConstraint(nn.Module):
    def __init__(self, magnitude: float = 1.0):
        super().__init__()
        self.magnitude = magnitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
            
        mag = compl.abs()
        return self.magnitude - mag.flatten()
        