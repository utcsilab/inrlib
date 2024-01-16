import torch
from torch import nn
import numpy as np
from numpy import random


class GaussianPosEnc(nn.Module):
  """
  Gaussian positional encoding for 3D input points.

  Fourier-NeRF notebook: https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
  """

  def __init__(
      self,
      d_input: int,
      embed_sz: int = 256,
      scale: float = 2.,
  ):
    super().__init__()
    self.embed_sz =  embed_sz
    
    bvals = random.normal(size=(self.embed_sz, d_input)) * scale
    avals = np.ones(bvals.shape[0]) 

    self.avals = nn.Parameter(torch.from_numpy(avals).float(), requires_grad=False)
    self.bvals = nn.Parameter(torch.from_numpy(bvals).float(), requires_grad=False)

    pi = torch.tensor(np.pi)
    self.encoder = lambda x: torch.cat([self.avals * torch.sin((2 * pi * x) @ self.bvals.T), 
                                            self.avals * torch.cos((2 * pi * x) @ self.bvals.T)], dim=-1)
        
    self.d_output = self.avals.shape[0] + self.bvals.shape[0]

  def forward(
      self,
      x
  ) -> torch.Tensor:
    r"""
    Apply positional encoding to input.
    """
    try: 
      return self.encoder(x)
    except RuntimeError:
      print("RuntimeError: check that d_input is consistent with the input dim of the data / network!")
      raise


class NeRFPosEnc(nn.Module):
  """
  OG Fourier positional encoder for input points.
  NeRF in PyTorch: https: // towardsdatascience.com/its-nerf-from -nothing-build-a-vanilla-nerf-with -pytorch-7846e4c45666
  """

  def __init__(
      self,
      d_input: int,
      n_freqs: int,
      log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.embed_sz =  1 + 2 * self.n_freqs
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.embed_sz)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.embed_sz)

    # Alternate sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    self.d_output = len(self.embed_fns) * self.d_input 

  def forward(
      self,
      x
  ) -> torch.Tensor:
    r"""
    Apply positional encoding to input.
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class FourierPosEnc(nn.Module):
    """
    Yet another Fourier positional encoding for input points.
    
    Fourier-NeRF notebook: https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
    """
    def __init__(self, 
                 d_input: int,
                 n_freqs: int,
                 embed_sz: int=None, 
                 ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.embed_sz = embed_sz if embed_sz is not None else 1 + 2 * n_freqs
        
        bvals = 2.**np.linspace(0,n_freqs,self.embed_sz) - 1.  # log scale
        bvals = np.stack([bvals]+[np.zeros_like(bvals)]*(d_input-1), -1)
        bvals = np.concatenate([bvals] + [np.roll(bvals, i, axis=-1) for i in range(1,d_input)], 0) 
        avals = np.ones((bvals.shape[0])) 

        self.avals = nn.Parameter(torch.from_numpy(avals).float(), requires_grad=False)
        self.bvals = nn.Parameter(torch.from_numpy(bvals).float(), requires_grad=False)

        self.encoder = lambda x: torch.cat([self.avals * torch.sin((2 * np.pi * x) @ self.bvals.T), 
                                            self.avals * torch.cos((2 * np.pi * x) @ self.bvals.T)], dim=-1)
        
        self.d_output = self.avals.shape[0] + self.bvals.shape[0]

    def forward(self, x):
        return self.encoder(x)

