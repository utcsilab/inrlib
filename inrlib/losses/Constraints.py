import torch
import torch.nn as nn

from ..utils.numeric import make_complex
from ..Transforms import FFTTransform, IFFTTransform


class ComplexRealConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.real if x.is_complex() else x[..., 0]
        return xi.flatten()
   
    
class ComplexImaginaryConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.imag if x.is_complex() else x[..., 1]
        return xi.flatten()


class ComplexMagnitudeConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
            
        mag = compl.abs()
        return mag.flatten()
    
    
class ComplexPhaseConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
        
        phase = compl.angle()
        return phase.flatten()
     

class ComplexMagnitudeValueConstraint(nn.Module):
    def __init__(self, magnitude: float = 1.0):
        super().__init__()
        self.magnitude = magnitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
            
        mag = compl.abs()
        return self.magnitude - mag.flatten()


class ComplexPhaseValueConstraint(nn.Module):
    def __init__(self, phase: float = 0.0):
        super().__init__()
        self.phase = phase
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
        
        phase = compl.angle()
        return self.phase - phase.flatten()
