import numpy as np
from typing import Union, List, Literal
import torch

from .utils.imaging import fft, ifft
from .utils.numeric import make_complex 

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
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError("WaveletTransform not implemented yet")
