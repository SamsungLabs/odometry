from functools import partial

import numpy as np
import scipy
import PIL
import cv2

import torch
import torch.nn.functional as F


def warp2d(image, flow):
    assert (image[...,0].shape == flow[...,0].shape), 'shape mismatch'
    width, height = image.shape[1], image.shape[0]
    flow_absolute = flow.copy()
    flow_absolute[..., 0] *= width
    flow_absolute[..., 1] *= height
    x_indexes, y_indexes = np.meshgrid(np.arange(width), np.arange(height))
    x_warped_indexes = np.round(x_indexes + flow_absolute[..., 0]).astype(int)
    y_warped_indexes = np.round(y_indexes + flow_absolute[..., 1]).astype(int)
    x_warped_indexes = np.clip(x_warped_indexes, a_min=0, a_max=width - 1)
    y_warped_indexes = np.clip(y_warped_indexes, a_min=0, a_max=height - 1)
    warped_image = np.zeros_like(image)
    warped_image[y_warped_indexes, x_warped_indexes] = image[y_indexes, x_indexes]
    return warped_image


def resize_image(image, target_size):
    return cv2.resize(image, target_size, cv2.INTER_LINEAR)


def save_image(image, image_filepath):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_filepath, image)


def load_image(image_filepath, target_size=None):
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        image = resize_image(image, target_size)
    return image


def resize_image_arr(image_arr, target_size, data_format, mode):
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


def load_image_arr(fpath, mode=None):
    image = PIL.Image.open(fpath)
    image.load()

    if mode and image.mode != mode:
        image = image.convert(mode)

    image_arr = np.asarray(image, dtype='float32')

    if hasattr(image, 'close'):
        image.close()

    return image_arr


def convert_hwc_to_chw(image_arr):
    return image_arr.transpose((2, 0, 1))


def convert_chw_to_hwc(image_arr):
    return image_arr.transpose((1, 2, 0))


def get_channels_num(preprocess_mode):
    if preprocess_mode == 'rgb':
        return 3
    elif preprocess_mode == 'rgba':
        return 4
    elif preprocess_mode == 'grayscale':
        return 1
    elif preprocess_mode == 'flow_xy':
        return 2
    elif preprocess_mode == 'flow_xy_nan':
        return 3
    elif preprocess_mode == 'depth':
        return 1
    elif preprocess_mode == 'disparity':
        return 1
    elif preprocess_mode == 'motion_maps':
        return 7
    elif preprocess_mode == 'motion_maps_xy':
        return 4
    elif preprocess_mode == 'motion_maps_z':
        return 2
    else:
        raise ValueError(f'Unknown preprocess mode: {preprocess_mode}')


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


def fill_with_random(x, nan_value, mean=None, std=None):
    mask = [x == nan_value] if nan_value else np.isnan(x)

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


def fill_with_interpolation(x):
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


def get_fill_fn(method='random', **kwargs):
    if method == 'random':
        return partial(fill_with_random, **kwargs)
    if method == 'interpolate':
        return fill_with_interpolation
    if method == 'median':
        return fill_with_median
    else:
        return fill_with_zeros
