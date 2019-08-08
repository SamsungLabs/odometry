import os
from functools import partial

import numpy as np
import scipy
import PIL

import torch
import torch.nn.functional as F


def get_channels_count(preprocess_mode):
    if preprocess_mode == 'rgb':
        return 3
    if preprocess_mode == 'rgba':
        return 4
    if preprocess_mode == 'grayscale':
        return 1
    if preprocess_mode == 'flow_xy':
        return 2
    if preprocess_mode == 'flow_xy_nan':
        return 3
    if preprocess_mode == 'depth':
        return 1
    if preprocess_mode == 'disparity':
        return 1
    if preprocess_mode == 'motion_maps':
        return 7
    if preprocess_mode == 'motion_maps_xy':
        return 4
    if preprocess_mode == 'motion_maps_z':
        return 2


def fill_with_median(x):
    assert x.ndim == 2
    mask = np.isnan(x)
    if mask.all():
        return np.zeros_like(x)
    return np.where(mask, np.nanmedian(x), x)


def fill_with_zeros(x):
    assert x.ndim == 2
    mask = np.isnan(x)
    return np.where(mask, 0, x)


def fill_with_random(x, nan, mean=None, std=None):
    mask = [x == nan] if nan else np.isnan(x)

    if mask.all():
        return np.zeros_like(x)

    if not mask.any():
        return x

    if mean is not None and std is not None:
        mean, std = np.nanmean(x), np.nanstd(x)

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x = np.where(mask, np.random.randn(*x.shape) * std + mean, x)
    x = np.clip(x, a_min=x_min, a_max=x_max)
    return x


def interpolate(x):
    mask = np.isnan(x)
    if not mask.any():
        return x

    height, width = x.shape
    hh, ww = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = (hh, ww)
    grid_valid = (hh[~mask], ww[~mask])
    x_valid = x[~mask]
    x = scipy.interpolate.griddata(grid_valid, x_valid.ravel(), grid)
    return x


def fill_flow(method='random'):
    if method == 'random':
        return partial(fill_with_random, nan=np.nan, mean=0, std=1)
    if method == 'interpolate':
        return interpolate
    if method == 'median':
        return fill_with_median
    else:
        return fill_with_zeros


def fill_depth(method='random'):
    if method == 'random':
        return partial(fill_with_random, nan=0)
    if method == 'interpolate':
        return interpolate
    if method == 'median':
        return fill_with_median
    else:
        return fill_with_zeros


def resize_image(image_arr, target_size, data_format, mode):
    if image_arr.shape[:-1] == target_size:
        return image_arr

    if data_format == 'channels_last':
        image_arr = image_arr.transpose(2, 0, 1)

    image_arr = F.interpolate(torch.Tensor(image_arr).unsqueeze_(0),
                              target_size,
                              mode=mode).numpy()[0]

    if data_format == 'channels_last':
        image_arr = image_arr.transpose(1, 2, 0)
    return image_arr


def load_pil_image(fpath, mode=None):
    image = PIL.Image.open(fpath)
    image.load()

    if mode and image.mode != mode:
        image = image.convert(mode)

    image_arr = np.asarray(image, dtype='float32')

    if hasattr(image, 'close'):
        image.close()

    return image_arr
