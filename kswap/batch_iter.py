import numpy as np

from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from typing import Iterable, Callable, Union, Sequence
from dpipe.batch_iter.sources import sample
from dpipe.itertools import pam, squeeze_first
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap
from dpipe.batch_iter import unpack_args
from dpipe.im.patch import sample_box_center_uniformly

SPATIAL_DIMS = (-3, -2, -1)


def get_random_slice(*arrays, interval: int = 1):
    slc = np.random.randint(arrays[0].shape[-1] // interval) * interval
    return tuple(array[..., slc] for array in arrays)

def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2

def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper

def slicewise_naive_swap(predict):
    def wrapper(*arrays, **kwargs):
        return np.stack(lmap(unpack_args(predict, **kwargs), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper

def extract_patch(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    if len(inputs) == 3:
        x, y, center = inputs

        x_patch_size = np.array(x_patch_size)
        y_patch_size = np.array(y_patch_size)
        x_spatial_box = get_centered_box(center, x_patch_size)
        y_spatial_box = get_centered_box(center, y_patch_size)

        x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
        y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
        return x_patch, y_patch

    elif len(inputs) == 2:
        x, center = inputs
        x_patch_size = np.array(x_patch_size)
        x_spatial_box = get_centered_box(center, x_patch_size)
        x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
        return x_patch

    else:
        raise ValueError