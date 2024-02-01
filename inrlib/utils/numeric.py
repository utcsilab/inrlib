import numpy as np
from numpy import random
import torch

from typing import Union, Optional, Tuple, Callable, Any, Sequence, List


def make_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        is_numpy = isinstance(x, np.ndarray)
        return torch.from_numpy(x.copy()) if is_numpy else x.clone()


def make_complex(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    compl = make_tensor(x) 
    if not torch.is_complex(compl): 
        compl = torch.view_as_complex(compl) if compl.shape[-1] == 2 else torch.complex(compl, torch.zeros_like(compl))
    return compl


def make_real(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    real = make_tensor(x)
    return torch.view_as_real(real) if torch.is_complex(real) else real
