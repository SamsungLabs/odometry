import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import PIL
import keras_preprocessing.image as keras_img
from keras_preprocessing.image import ImageDataGenerator
import psutil


def _get_number_of_channels(preprocess_mode):
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
    if preprocess_mode == 'seven_flow':
        return 7
    if preprocess_mode == 'xyt_xyr_dof_flow':
        return 4
    if preprocess_mode == 'zt_yr_dof_flow':
        return 2


def _fill_flow_with_median(flow_plane):
    assert flow_plane.ndim == 2
    if np.isnan(flow_plane).all():
        return np.zeros_like(flow_plane)
    return np.where(np.isnan(flow_plane), np.nanmedian(flow_plane), flow_plane)


def _fill_flow_with_zeros(flow_plane):
    assert flow_plane.ndim == 2
    return np.where(np.isnan(flow_plane), 0, flow_plane)


def _fill_flow_with_random(flow_plane):
    assert flow_plane.ndim == 2
    return np.where(np.isnan(flow_plane), np.random.normal(size=flow_plane.shape), flow_plane)


def _fill_depth_with_random(depth_plane):
    valid_values = depth_plane[depth_plane > 0]
    mean, std = np.mean(valid_values), np.std(valid_values)
    min_depth, max_depth = np.min(valid_values), np.max(valid_values)
    depth_plane = np.where((depth_plane == 0), np.random.randn(*depth_plane.shape) * std + mean, depth_plane)
    depth_plane = np.clip(depth_plane, a_min=min_depth, a_max=max_depth)
    return depth_plane


def _interpolate(img_arr):
    from scipy import interpolate

    height, width = img_arr.shape
    hh, ww = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = (hh, ww)

    mask = np.isnan(img_arr)
    if not np.any(mask):
        return img_arr

    grid_valid = (hh[~mask], ww[~mask])
    img_arr_valid = img_arr[~mask]
    img_arr = interpolate.griddata(grid_valid, img_arr_valid.ravel(), grid)
    return img_arr


def _get_fill_flow_function(flow_fill_method):
    if flow_fill_method == 'random':
        return _fill_flow_with_random
    if flow_fill_method == 'interpolate':
        return _interpolate
    if flow_fill_method == 'median':
        return _fill_flow_with_median
    
    return _fill_flow_with_zeros


def _resize(img_arr, target_size, data_format, mode):
    if img_arr.shape[:-1] == target_size:
        return img_arr

    if data_format == 'channels_last':
        img_arr = img_arr.transpose(2, 0, 1)

    img_arr = F.interpolate(torch.Tensor(img_arr).unsqueeze_(0),
                            target_size,
                            mode=mode).numpy()[0]

    if data_format == 'channels_last':
        img_arr = img_arr.transpose(1, 2, 0)
    return img_arr


class ExtendedDataFrameIterator(keras_img.iterator.BatchFromFilesMixin, keras_img.Iterator):
    def __init__(self,
                 dataframe,
                 directory,
                 image_data_generator,
                 x_col="x_col",
                 y_col="y_col",
                 image_columns=None,
                 target_size=(256, 256),
                 load_modes='rgb',
                 preprocess_modes=None,
                 batch_size=128,
                 shuffle=True,
                 seed=42,
                 data_format='channels_last',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32',
                 flow_multiplicator=(1,1),
                 flow_fill_method='random',
                 depth_multiplicator=1./5000., 
                 cached_imgs=None,
                 filter_invalid=True,
                 max_memory_consumption=0.8):
        super().set_processing_attrs(image_data_generator,
                                     target_size,
                                     'rgb',
                                     data_format,
                                     save_to_dir=False,
                                     save_prefix='',
                                     save_format='png',
                                     subset=subset,
                                     interpolation=interpolation)
        if image_columns is None:
            image_columns = []

        assert isinstance(image_columns, list)

        self.df = dataframe
        self.df[image_columns] = self.df[image_columns].astype(str)
        self.directory = directory
        self.dtype = dtype
        self.samples = len(self.df)
        self.df_images = self.df[image_columns]

        if isinstance(x_col, str):
            self.x_cols = [x_col]
        else:
            self.x_cols = x_col
        if isinstance(y_col, str):
            self.y_cols = [y_col]
        else:
            self.y_cols = y_col

        assert set(image_columns) <= (set(self.x_cols) | set(self.y_cols))
        assert (set(self.x_cols) | set(self.y_cols)) <= set(self.df.columns)

        self.image_cols = image_columns

        if isinstance(load_modes, str):
            self.load_modes = dict((image_column, load_modes) for image_column in self.image_cols)
        else:  # load_mode is iterable
            self.load_modes = dict(zip(self.image_cols, load_modes))

        assert len(self.load_modes) == len(self.image_cols)

        if isinstance(preprocess_modes, str) or preprocess_modes is None:
            self.preprocess_modes = dict((image_column, preprocess_modes) for image_column in self.image_cols)
        else:
            self.preprocess_modes = dict(zip(self.image_cols, preprocess_modes))

        assert len(self.preprocess_modes) == len(self.image_cols)

        channels_counts_dict = dict((image_column, _get_number_of_channels(preprocess_mode))
                                    for image_column, preprocess_mode in self.preprocess_modes.items())
        self.channels_counts = [channels_counts_dict[image_column] 
                                for image_column in self.x_cols if image_column in self.image_cols]
        self.image_shapes = dict((image_column, self.target_size + (num_channels,))
                                 for image_column, num_channels in channels_counts_dict.items())

        self.flow_multiplicator = flow_multiplicator
        self._fill_flow = _get_fill_flow_function(flow_fill_method)

        self.depth_multiplicator = depth_multiplicator
        self._fill_depth = _fill_depth_with_random

        self.set_cache(cached_imgs)
        self.max_memory_consumption = max_memory_consumption
        self.stop_caching = self._check_stop_caching()

        self.filter_invalid = filter_invalid

        super(ExtendedDataFrameIterator, self).__init__(
            self.samples,
            batch_size,
            shuffle,
            seed)

    def _check_stop_caching(self):
        return psutil.virtual_memory().percent / 100 > self.max_memory_consumption

    def set_cache(self, cached_imgs):
        if cached_imgs is not None:
            assert isinstance(cached_imgs, dict)
            if len(cached_imgs) == 0:
                print('Set empty cache')
        else:
            print('No cache')
        self.cached_imgs = cached_imgs

    def _load_img(self, fpath, load_mode):
        if fpath.endswith('.npy'):
            img_arr = np.load(fpath)
        else:
            img = PIL.Image.open(fpath)
            img.load()

            if load_mode == 'grayscale':
                if img.mode != 'L':
                    img = img.convert('L')
            elif load_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif load_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')

            img_arr = np.asarray(img, dtype="float32")

            if hasattr(img, 'close'):
                img.close()

        if len(img_arr.shape) == 2:
            img_arr = np.expand_dims(img_arr, -1)

        if load_mode == 'seven_flow':
            img_arr = img_arr.transpose(1, 2, 0)

        if load_mode == 'zt_yr_dof_flow':
            img_arr = img_arr[[2,5],:,:].transpose(1, 2, 0)

        if load_mode == 'xyt_xyr_dof_flow':
            img_arr = img_arr[[0,1,4,5],:,:].transpose(1, 2, 0)

        img_arr = _resize(img_arr, self.target_size, self.data_format, mode=self.interpolation)
        return img_arr

    def _preprocess_img(self, img_arr, load_mode, preprocess_mode):
        if load_mode == 'depth':
            img_arr *= self.depth_multiplicator

        if load_mode == 'disparity':
            img_arr /= self.depth_multiplicator

        if load_mode in ('flow_xy', 'flow_xy_nan'):
            img_arr[:, :, 0] *= self.flow_multiplicator[0]
            img_arr[:, :, 1] *= self.flow_multiplicator[1]

        if load_mode in ('depth', 'disparity'):
            if (img_arr == 0).all():
                print('invalid depth')
                if self.filter_invalid:
                    return None

                max_depth = 100.
                if load_mode == 'depth':
                    img_arr = np.ones_like(img_arr) * max_depth
                else:
                    img_arr = np.ones_like(img_arr) * (1. / max_depth)

            elif (img_arr == 0).any():
                img_arr = self._fill_depth(img_arr)

        if load_mode == preprocess_mode:
            return img_arr

        if load_mode == 'flow_xy' and preprocess_mode == 'flow_xy_nan':
            isnan = (np.isnan(img_arr[:, :, 0]) | np.isnan(img_arr[:, :, 1])).astype(self.dtype)
            if isnan.any():
                img_arr[:, :, 0] = self._fill_flow(img_arr[:, :, 0])
                img_arr[:, :, 1] = self._fill_flow(img_arr[:, :, 1])

            img_arr = np.concatenate([img_arr, np.expand_dims(isnan, -1)], axis=-1)

        elif load_mode == 'flow_xy_nan' and preprocess_mode == 'flow_xy':
            img_arr = img_arr[:, :, :2]

        elif (load_mode == 'depth' and preprocess_mode == 'disparity' or
              load_mode == 'disparity' and preprocess_mode == 'depth'):
            img_arr = 1.0 / img_arr

        else:
            print('Can not perform casting from {} to {}'.format(load_mode, preprocess_mode))
            raise NotImplemented

        return img_arr

    def _get_preprocessed_img(self, fname, load_mode, preprocess_mode):
        fpath = os.path.join(self.directory, fname)
        if (self.cached_imgs is not None) and (fpath in self.cached_imgs):
            img_arr = self.cached_imgs[fpath]
        else:
            img_arr = self._load_img(fpath, load_mode)
            img_arr = self._preprocess_img(img_arr, load_mode, preprocess_mode)

            if (self.cached_imgs is not None) and \
                    (not self.stop_caching) and \
                    (len(self.cached_imgs) % 1000 == 0):
                self.stop_caching = self._check_stop_caching()

            if (self.cached_imgs is not None) and (not self.stop_caching):
                self.cached_imgs[fpath] = img_arr

        if img_arr is None:
            return None

        return img_arr.copy()

    def _init_batch(self, columns, index_array):
        batch = []

        for col in columns:
            if col in self.image_cols:
                batch.append(
                    np.zeros((len(index_array),) + self.image_shapes[col], dtype=self.dtype))
            else:
                batch.append(self.df[col].values[index_array])

        return batch

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = self._init_batch(self.x_cols, index_array)
        batch_y = self._init_batch(self.y_cols, index_array)

        # build batch of image data
        valid_samples = np.ones(len(index_array)).astype(bool)
        for index_in_batch, df_row_index in enumerate(index_array):
            seed = np.random.randint(1000000)
            for col, fname in self.df_images.iloc[df_row_index].iteritems():
                params = self.image_data_generator.get_random_transform(self.image_shapes[col], seed)
                img_arr = self._get_preprocessed_img(fname, self.load_modes[col], self.preprocess_modes[col])
                if img_arr is None:
                    valid_samples[index_in_batch] = False
                    continue
                img_arr = self.image_data_generator.apply_transform(img_arr, params)
                img_arr = self.image_data_generator.standardize(img_arr)
                if col in self.x_cols:
                    batch_x[self.x_cols.index(col)][index_in_batch] = img_arr
                if col in self.y_cols:
                    batch_y[self.y_cols.index(col)][index_in_batch] = img_arr

        batch_x = [features[valid_samples] for features in batch_x]
        batch_y = [target[valid_samples] for target in batch_y]

        if np.sum(valid_samples) < 0.5 * len(index_array):
            print('Batch is too small: {} samples'.format(np.sum(valid_samples)))
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self._get_batches_of_transformed_samples(index_array)
