from inr.data.DataModules import *
from inr.data.Datasets import *
from inr.models.MLP import *
from inr.losses.MSE import *
from inr.losses.Regularizers import *
from inr.losses.Constraints import * 
from inr.utils import *
from inr.ImageLoggers import *
from inr.Transforms import *

import numpy as np

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


def get_coordinates(RES=96):
	"""
	For a cubic resolution, return coordinates of the form (x,y,z) in [0,1]^3
	"""
	x1 = np.linspace(0, 1, RES+1)[:-1] # use full image resolution 
	x_train = np.stack(np.meshgrid(x1,x1,x1), axis=-1)
	x_test = x_train
	return x_train, x_test


def mri_mask(shape, nsamp):
	"""
	Generate a random multivariate Gaussian mask for MRI subsampling in k-space
	"""
	mean = np.array(shape)//2
	cov = np.eye(len(shape)) * (2*shape[0])
	samps = random.multivariate_normal(mean, cov, size=(1,nsamp))[0,...].astype(np.int32)
	samps = samps.clip(0, shape[0]-1)
	mask = np.zeros(shape)
	mask[samps] = 1.   # prev used deprecated index_update and index from jax.ops
	mask = np.fft.fftshift(mask).astype(np.complex64)  # fftshift does not compute fft, just shifts the spectrum
	return mask


def fft3D(x):
	return np.fft.fft(np.fft.fft(np.fft.fft(x, axis=0), axis=1), axis=2)
