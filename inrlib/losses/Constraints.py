import torch
import torch.nn as nn

from ..utils.imaging import reshape
from ..utils.numeric import make_complex
from ..Transforms import FFTTransform, IFFTTransform


class ComplexRealConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.real if x.is_complex() else x[..., 0]
        return xi
   
    
class ComplexImaginaryConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xi = x.imag if x.is_complex() else x[..., 1]
        return xi


class ComplexMagnitudeConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
            
        mag = compl.abs()
        return mag 
    
    
class ComplexPhaseConstraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
        
        phase = compl.angle()
        return phase 
     

class ComplexMagnitudeValueConstraint(nn.Module):
    def __init__(self, magnitude: float = 1.0):
        super().__init__()
        self.magnitude = magnitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
            
        mag = compl.abs()
        return self.magnitude - mag 


class ComplexPhaseValueConstraint(nn.Module):
    def __init__(self, phase: float = 0.0):
        super().__init__()
        self.phase = phase
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
        
        phase = compl.angle()
        return self.phase - phase 


class ComplexCasoratiConstraint(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compl = make_complex(x)
        
        return reshape(compl, (-1,compl.shape[-1]))