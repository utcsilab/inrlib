import numpy as np
from typing import Union, List, Literal, Tuple
import torch

import ptwt, pywt # pytorch wavelets, pywavelets

from .utils.imaging import fft, ifft
from .utils.numeric import make_complex, make_real

from abc import ABC, abstractmethod

class ABCTransform(ABC):
    def __init__(self, **plotting_kwargs):
        self.plotting_kwargs = plotting_kwargs
    def __call__(self) -> Union[torch.Tensor, np.ndarray]:
        pass


class FFTTransform(ABCTransform):
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return fft(x).numpy()
    
    
class IFFTTransform(ABCTransform):
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return ifft(x).numpy()


class MagnitudeTransform(ABCTransform):
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        compl = make_complex(x) # returned as tensor
        mag = compl.abs()
        return mag.numpy()


class PhaseTransform(ABCTransform): 
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        compl = make_complex(x) # returned as tensor
        phase = compl.angle()
        return phase.numpy()


class WaveletTransform(ABCTransform):
    def __init__(self, wavelet='db1', mode='symmetric', dims: Tuple[int, ...]=(0,1), level: int=1, **kwargs):
        
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.mode = mode
        self.dims = dims
        self.level = level
    
    def __call__(self, x: Union[torch.Tensor, np.ndarray]):
        compl = make_real(x) # ptwt expects float input
        coefs = ptwt.wavedec2(compl, self.wavelet, mode=self.mode, axes=self.dims, level=self.level)
        return coefs
  

class InverseWaveletTransform(ABCTransform):
    def __init__(self, wavelet='db1', mode='symmetric', dims: Tuple[int, ...]=(0,1), **kwargs):
        super().__init__(**kwargs)

        self.wavelet = wavelet
        self.mode = mode
        self.dims = dims
    
    def __call__(self, coefs: torch.Tensor):
        return ptwt.waverec2(coefs, self.wavelet, axes=self.dims)


if __name__ == '__main__':
    wavelet_transform = WaveletTransform(wavelet='db1', mode='symmetric', dims=(0, 1), level=4)
    inverse_wavelet_transform = InverseWaveletTransform(wavelet='db1', mode='symmetric', dims=(0, 1))

    # Input tensor (replace with actual data)
    x = torch.randn(32, 32, 3, 2)  # Example 2D image with shape (batch_size, channels, height, width)

    # Perform forward wavelet transform
    coefs = wavelet_transform(x)

    # Perform inverse wavelet transform
    x_reconstructed = inverse_wavelet_transform(coefs)

    # Check reconstruction error
    reconstruction_error = torch.norm(x - x_reconstructed)
    print(f'Reconstruction error: {reconstruction_error.item()}')