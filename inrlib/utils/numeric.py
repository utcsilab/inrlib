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


### 2-channel complex arithmetic
'''
@author Jon Tamir

Defines complex-valued arithmetic for ndarrays, where the real and imaginary
channels are stored in the last dimension
'''

def c2r(z):
    ''' Convert from complex to 2-channel real '''
    assert type(z) is np.ndarray, 'Must be numpy.ndarray'
    return np.stack((z.real, z.imag), axis=-1)

def r2c(x):
    ''' Convert from 2-channel real to complex '''
    assert type(x) is np.ndarray, 'Must be numpy.ndarray'
    return x[...,0] + 1j *  x[...,1]

def zmul(x1, x2):
    ''' complex-valued multiplication '''
    xr = x1[...,0] * x2[...,0] -  x1[...,1] * x2[...,1]
    xi = x1[...,0] * x2[...,1] +  x1[...,1] * x2[...,0]
    if type(x1) is np.ndarray:
        return np.stack((xr, xi), axis=-1)
    elif type(x1) is torch.Tensor:
        return torch.stack((xr, xi), dim=-1)
    else:   
        return xr, xi

def zconj(x):
    ''' complex-valued conjugate '''
    if type(x) is np.ndarray:
        return np.stack((x[...,0], -x[...,1]), axis=-1)
    elif type(x) is torch.Tensor:
        return torch.stack((x[...,0], -x[...,1]), dim=-1)
    else:   
        return x[...,0], -x[...,1]

def zabs(x):
    ''' complex-valued magnitude '''
    if type(x) is np.ndarray:
        return np.sqrt(zmul(x, zconj(x)))[...,0]
    elif type(x) is torch.Tensor:
        return torch.sqrt(zmul(x, zconj(x)))
    else:   
        return -1.

### OPT
"""Vector operations for use in calculating conjugate gradient descent."""

def dot(x1, x2):
    """Finds the dot product of two vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2.
    """

    return torch.sum(x1*x2)

def dot_single(x):
    """Finds the dot product of a vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x.
    """

    return dot(x, x)

def dot_batch(x1, x2):
    """Finds the dot product of two multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)


def dot_single_batch(x):
    """Finds the dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return dot_batch(x, x)

def zdot(x1, x2):
    """Finds the complex-valued dot product of two complex-valued vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2, defined as sum(conj(x1) * x2)
    """

    return torch.sum(torch.conj(x1)*x2)

def zdot_single(x):
    """Finds the complex-valued dot product of a complex-valued vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x., defined as sum(conj(x) * x)
    """

    return zdot(x, x)

def zdot_batch(x1, x2):
    """Finds the complex-valued dot product of two complex-valued multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)


def zdot_single_batch(x):
    """Finds the complex-valued dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return zdot_batch(x, x)

def l2ball_proj_batch(x, eps):
    """ Performs a batch projection onto the L2 ball.

    Args:
        x (Tensor): The tensor to be projected.
        eps (Tensor): A tensor containing epsilon values for each dimension of the L2 ball.

    Returns:
        The projection of x onto the L2 ball.
    """

    #print('l2ball_proj_batch')
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    x = x.contiguous()
    q1 = torch.real(zdot_single_batch(x)).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = torch.real(zdot_single_batch(z)).sqrt()
    #print(eps,q1,q2)
    return z