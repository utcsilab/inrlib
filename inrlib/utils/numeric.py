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
    return torch.view_as_real(real).float() if torch.is_complex(real) else real


### 2-channel complex arithmetic

def zmul(x1i: Union[np.ndarray, torch.Tensor], x2i: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    ''' complex-valued multiplication '''
    x1 = make_complex(x1i)
    x2 = make_complex(x2i)
    
    return x1 * x2


def zconj(xi: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    ''' complex-valued conjugate '''
    x = make_complex(xi)
    return torch.conj(x)


def zabs(xi: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    ''' complex-valued magnitude '''
    x = make_complex(xi)
    return torch.abs(x)


def zangle(xi: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    ''' complex-valued angle '''
    x = make_complex(xi)
    return torch.angle(x)


def zdot(x1i: Union[np.ndarray, torch.Tensor], x2i: Union[np.ndarray, torch.Tensor], dim=-1) -> torch.Tensor: 
    """Finds the complex-valued dot product of two complex-valued tensors.
    """
    x1 = make_complex(x1i)
    x2 = make_complex(x2i)

    return torch.sum(x1 * torch.conj(x2), dim=dim)


# def l2ball_proj_batch(x, eps):
#     """ Performs a batch projection onto the L2 ball.

#     Args:
#         x (Tensor): The tensor to be projected.
#         eps (Tensor): A tensor containing epsilon values for each dimension of the L2 ball.

#     Returns:
#         The projection of x onto the L2 ball.
#     """

#     #print('l2ball_proj_batch')
#     reshape = (-1,) + (1,) * (len(x.shape) - 1)
#     x = x.contiguous()
#     q1 = torch.real(zdot_single_batch(x)).sqrt()
#     #print(eps,q1)
#     q1_clamp = torch.min(q1, eps)

#     z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
#     #q2 = torch.real(zdot_single_batch(z)).sqrt()
#     #print(eps,q1,q2)
#     return z