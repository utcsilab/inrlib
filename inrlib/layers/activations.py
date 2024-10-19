import numpy as np
import torch
import torch.nn as nn

from ..utils.numeric import make_complex, make_real


class Sine(nn.Module):
	def __init__(self, w0: float = 1.):
		super().__init__()
		self.w0 = w0
  
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.sin(self.w0 * x)


class modReLU(nn.Module):
    def __init__(self, b: float = -1.):
        super().__init__()
        assert b < 0, "b must be negative"
        self.b = b
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = make_complex(x)
        xmag = torch.abs(xc)
        term = xmag + self.b
        return torch.where(term >= 0, xc/xmag * term, torch.tensor(0.))


class Cardioid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = make_complex(x)
        return 0.5 * (1 + torch.cos(torch.angle(xc))) * xc


class zReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = make_complex(x)
        return torch.where(0 <= torch.angle(xc) <= np.pi/2, xc, torch.tensor(0.))