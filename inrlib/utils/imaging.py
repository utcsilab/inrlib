import numpy as np
from numpy import random
import torch

from typing import Union, Optional, Tuple, Callable, Any, Sequence, List

from .numeric import make_complex


def reshape(arr: Union[np.ndarray, torch.Tensor], shape: Tuple[int, ...]) -> Union[np.ndarray, torch.Tensor]:
    """
    Reshape an array to a given shape
    """
    if isinstance(arr, np.ndarray):
        out = arr.copy()
    elif isinstance(arr, torch.Tensor):
        out = arr.clone()
    
    return out.reshape(*shape) if tuple(out.shape) != shape else out


# @title NP Area Resize Code
# from https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
def reflect_breaks(size: int) -> np.ndarray:
    """Calculate cell boundaries with reflecting boundary conditions."""
    result = np.concatenate([[0], 0.5 + np.arange(size - 1), [size - 1]])
    assert len(result) == size + 1
    return result


def interval_overlap(first_breaks: np.ndarray,
                                            second_breaks: np.ndarray) -> np.ndarray:
    """Return the overlap distance between all pairs of intervals.

    Args:
        first_breaks: breaks between entries in the first set of intervals, with
            shape (N+1,). Must be a non-decreasing sequence.
        second_breaks: breaks between entries in the second set of intervals, with
            shape (M+1,). Must be a non-decreasing sequence.

    Returns:
        Array with shape (N, M) giving the size of the overlapping region between
        each pair of intervals.
    """
    first_upper = first_breaks[1:]
    second_upper = second_breaks[1:]
    upper = np.minimum(first_upper[:, np.newaxis], second_upper[np.newaxis, :])

    first_lower = first_breaks[:-1]
    second_lower = second_breaks[:-1]
    lower = np.maximum(first_lower[:, np.newaxis], second_lower[np.newaxis, :])

    return np.maximum(upper - lower, 0)


def resize_weights(
                old_size: int, new_size: int, reflect: bool = False) -> np.ndarray:
    """Create a weight matrix for resizing with the local mean along an axis.

    Args:
        old_size: old size.
        new_size: new size.
        reflect: whether or not there are reflecting boundary conditions.

    Returns:
        NumPy array with shape (new_size, old_size). Rows sum to 1.
    """
    if not reflect:
        old_breaks = np.linspace(0, old_size, num=old_size + 1)
        new_breaks = np.linspace(0, old_size, num=new_size + 1)
    else:
        old_breaks = reflect_breaks(old_size)
        new_breaks = (old_size - 1) / (new_size - 1) * reflect_breaks(new_size)

    weights = interval_overlap(new_breaks, old_breaks)
    weights /= np.sum(weights, axis=1, keepdims=True)
    assert weights.shape == (new_size, old_size)
    return weights


def resize(array: np.ndarray,
                     shape: list[int],
                     reflect_axes: list[int] = ()) -> np.ndarray:
    """Resize an array with the local mean / bilinear scaling.

    Works for both upsampling and downsampling in a fashion equivalent to
    block_mean and zoom, but allows for resizing by non-integer multiples. Prefer
    block_mean and zoom when possible, as this implementation is probably slower.

    Args:
        array: array to resize.
        shape: shape of the resized array.
        reflect_axes: iterable of axis numbers with reflecting boundary conditions,
            mirrored over the center of the first and last cell.

    Returns:
        Array resized to shape.

    Raises:
        ValueError: if any values in reflect_axes fall outside the interval
            [-array.ndim, array.ndim).
    """
    reflect_axes_set = set()
    for axis in reflect_axes:
        if not -array.ndim <= axis < array.ndim:
            raise ValueError('invalid axis: {}'.format(axis))
        reflect_axes_set.add(axis % array.ndim)

    output = array
    for axis, (old_size, new_size) in enumerate(zip(array.shape, shape)):
        reflect = axis in reflect_axes_set
        weights = resize_weights(old_size, new_size, reflect=reflect)
        product = np.tensordot(output, weights, [[axis], [-1]])
        output = np.moveaxis(product, -1, axis)
    return output
## end of NP Area Resize Code

def compute_posenc_vals(mres, embed_sz):
        bvals = 2.**np.linspace(0, mres, embed_sz//3) - 1.
        bvals = np.stack([bvals, np.zeros_like(bvals), np.zeros_like(bvals)], -1)
        bvals = np.concatenate(
                [bvals, np.roll(bvals, 1, axis=-1), np.roll(bvals, 2, axis=-1)], 0)
        avals = np.ones((bvals.shape[0]))
        return avals, bvals


### RECON
def get_coordinates(shape: tuple, indexing='ij'):
    """
    Continuous coordinate system for a given shape

    For a cubic resolution, return coordinates of the form (x,y,z) in [0,1]^3
    """
    train_RES = [np.linspace(0, 1, res+1)[:-1] for res in shape]
    x_train = np.stack(np.meshgrid(*train_RES,indexing=indexing), axis=-1)
    
    test_RES = [np.linspace(0, 1, res+1)[:-1] + 1./(2*res) for res in shape]
    x_test = np.stack(np.meshgrid(*test_RES,indexing=indexing), axis=-1)
    return x_train, x_test


def get_inds(coords: np.ndarray, shape: Union[Tuple[int], List[int], np.ndarray]) -> np.ndarray:
        """
        Compute indices for an array of continuous values in [0,1] given 
        """
        assert len(shape) == coords.shape[-1], f"Number of axes {len(shape)} and coordinate dimension {coords.shape[-1]} mismatch"

        inds = np.zeros_like(coords)
        for axis, l in enumerate(shape):
                inds[..., axis] = coords[...,axis] * (l+1)
        
        return inds.astype(np.int32)

class BasicImageReconstructor: 
    def __call__(self, x_coords: np.ndarray, y_hat: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        # get discrete indices from continuous [0,1] coordinates and reconstruct image from shuffled outputs
        inds = get_inds(x_coords,image_shape)
        yu = np.zeros((*image_shape, y_hat.shape[-1]))
        yu[tuple(inds.T)] = y_hat

        self.recon_image = yu.squeeze()
        return self.recon_image

####

def subsampling_mask(shape, nsamp):
    """
    Generate a random multivariate Gaussian mask for subsampling
    """
    mean = np.array(shape)//2
    cov = np.eye(len(shape)) * (2*shape[0])
    samps = random.multivariate_normal(mean, cov, size=(1,nsamp))[0,...].astype(np.int32)
    samps = samps.clip(0, shape[0]-1)
    mask = np.zeros(shape)
    mask[samps] = 1.   # prev used deprecated index_update and index from jax.ops
    mask = np.fft.fftshift(mask).astype(np.complex64)  # fftshift does not compute fft, just shifts the spectrum
    return mask


def fft(x: Union[np.ndarray, torch.Tensor], dims=Optional[List[int]]=None) -> torch.Tensor:
    compl = make_complex(x)
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(compl), norm='ortho', dim=dims))

def ifft(x: Union[np.ndarray, torch.Tensor], dims=Optional[List[int]]=None) -> torch.Tensor:
    compl = make_complex(x)
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(compl), norm='ortho', dim=dims))

class NRMSE:
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
            return np.sqrt(np.mean(np.square(pred-target))) / np.sqrt(np.mean(np.square(target)))
