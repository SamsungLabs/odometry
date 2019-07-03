import os

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import PIL
import keras_preprocessing.image as keras_image
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


def _interpolate(image_arr):
    from scipy import interpolate

    height, width = image_arr.shape
    hh, ww = np.meshgrid(np.arange(0, width), np.arange(0, height))
    grid = (hh, ww)

    mask = np.isnan(image_arr)
    if not np.any(mask):
        return image_arr

    grid_valid = (hh[~mask], ww[~mask])
    image_arr_valid = image_arr[~mask]
    image_arr = interpolate.griddata(grid_valid, image_arr_valid.ravel(), grid)
    return image_arr


def _get_fill_flow_function(flow_fill_method):
    if flow_fill_method == 'random':
        return _fill_flow_with_random
    if flow_fill_method == 'interpolate':
        return _interpolate
    if flow_fill_method == 'median':
        return _fill_flow_with_median
    
    return _fill_flow_with_zeros


def _resize(image_arr, target_size, data_format, mode):
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


class ExtendedDataFrameIterator(keras_image.iterator.BatchFromFilesMixin, keras_image.Iterator):
    def __init__(self,
                 dataframe,
                 directory,
                 image_data_generator,
                 x_col='x_col',
                 y_col='y_col',
                 image_col=None,
                 target_size=(256, 256),
                 load_mode='rgb',
                 preprocess_mode=None,
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
                 cached_images=None,
                 filter_invalid=True,
                 max_memory_consumption=0.8,
                 return_confidences=False):
        super().set_processing_attrs(image_data_generator,
                                     target_size,
                                     'rgb',
                                     data_format,
                                     save_to_dir=False,
                                     save_prefix='',
                                     save_format='png',
                                     subset=subset,
                                     interpolation=interpolation)
        if image_col is None:
            image_col = []

        assert isinstance(image_col, list)

        self.df = dataframe
        self.df[image_col] = self.df[image_col].astype(str)
        self.directory = directory
        self.dtype = dtype
        self.samples = len(self.df)
        self.df_images = self.df[image_col]

        if isinstance(x_col, str):
            self.x_cols = [x_col]
        else:
            self.x_cols = x_col
        if isinstance(y_col, str):
            self.y_cols = [y_col]
        else:
            self.y_cols = y_col

        self.image_cols = image_col

        assert set(self.image_cols) <= (set(self.x_cols) | set(self.y_cols))
        assert (set(self.x_cols) | set(self.y_cols)) <= set(self.df.columns)

        if isinstance(load_mode, str):
            self.load_mode = dict((col, load_mode) for col in self.image_cols)
        else:  # load_mode is iterable
            self.load_mode = dict(zip(self.image_cols, load_mode))

        assert len(self.load_mode) == len(self.image_cols)

        if isinstance(preprocess_mode, str) or preprocess_mode is None:
            self.preprocess_mode = dict((col, preprocess_mode) for col in self.image_cols)
        else:
            self.preprocess_mode = dict(zip(self.image_cols, preprocess_mode))

        assert len(self.preprocess_mode) == len(self.image_cols)

        channels_counts_dict = dict((col, _get_number_of_channels(preprocess_mode))
                                    for col, preprocess_mode in self.preprocess_mode.items())
        self.channels_counts = [channels_counts_dict[col]
                                for col in self.x_cols if col in self.image_cols]
        self.image_shapes = dict((col, self.target_size + (num_channels,))
                                 for col, num_channels in channels_counts_dict.items())
        self.input_shapes = []
        for col in self.x_cols:
            if col in self.image_cols:
                self.input_shapes.append(self.image_shapes[col])
            else:
                self.input_shapes.append((1,))

        self.flow_multiplicator = flow_multiplicator
        self._fill_flow = _get_fill_flow_function(flow_fill_method)

        self.depth_multiplicator = depth_multiplicator
        self._fill_depth = _fill_depth_with_random

        self.set_cache(cached_images)
        self.max_memory_consumption = max_memory_consumption
        self.stop_caching = self._check_stop_caching()

        self.filter_invalid = filter_invalid

        self.return_confidences = return_confidences

        super(ExtendedDataFrameIterator, self).__init__(
            self.samples,
            batch_size,
            shuffle,
            seed)

    def _check_stop_caching(self):
        return psutil.virtual_memory().percent / 100 > self.max_memory_consumption

    def set_cache(self, cached_images):
        if cached_images is not None:
            assert isinstance(cached_images, dict)
            if len(cached_images) == 0:
                print('Set empty cache')
        else:
            print('No cache')
        self.cached_images = cached_images

    def _load_image(self, fpath, load_mode):
        if os.path.islink(fpath):
            fpath = os.readlink(fpath)

        if fpath.endswith('.npy'):
            image_arr = np.load(fpath)
        else:
            image = PIL.Image.open(fpath)
            image.load()

            if load_mode == 'grayscale':
                if image.mode != 'L':
                    image = image.convert('L')
            elif load_mode == 'rgba':
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
            elif load_mode == 'rgb':
                if image.mode != 'RGB':
                    image = image.convert('RGB')

            image_arr = np.asarray(image, dtype='float32')

            if hasattr(image, 'close'):
                image.close()

        if len(image_arr.shape) == 2:
            image_arr = np.expand_dims(image_arr, -1)

        if load_mode == 'seven_flow':
            image_arr = image_arr.transpose(1, 2, 0)

        if load_mode == 'zt_yr_dof_flow':
            image_arr = image_arr[[2,5],:,:].transpose(1, 2, 0)

        if load_mode == 'xyt_xyr_dof_flow':
            image_arr = image_arr[[0,1,4,5],:,:].transpose(1, 2, 0)

        image_arr = _resize(image_arr, self.target_size, self.data_format, mode=self.interpolation)
        return image_arr

    def _preprocess_image(self, image_arr, load_mode, preprocess_mode):
        if load_mode == 'depth':
            image_arr *= self.depth_multiplicator

        if load_mode == 'disparity':
            image_arr /= self.depth_multiplicator

        if load_mode in ('flow_xy', 'flow_xy_nan'):
            image_arr[:, :, 0] *= self.flow_multiplicator[0]
            image_arr[:, :, 1] *= self.flow_multiplicator[1]

        if load_mode in ('depth', 'disparity'):
            if (image_arr == 0).all():
                if self.filter_invalid:
                    return None

                max_depth = 100.
                if load_mode == 'depth':
                    image_arr = np.ones_like(image_arr) * max_depth
                else:
                    image_arr = np.ones_like(image_arr) * (1. / max_depth)

            elif (image_arr == 0).any():
                image_arr = self._fill_depth(image_arr)

        if load_mode == preprocess_mode:
            return image_arr

        if load_mode == 'flow_xy' and preprocess_mode == 'flow_xy_nan':
            isnan = (np.isnan(image_arr[:, :, 0]) | np.isnan(image_arr[:, :, 1])).astype(self.dtype)
            if isnan.any():
                image_arr[:, :, 0] = self._fill_flow(image_arr[:, :, 0])
                image_arr[:, :, 1] = self._fill_flow(image_arr[:, :, 1])

            image_arr = np.concatenate([image_arr, np.expand_dims(isnan, -1)], axis=-1)

        elif load_mode == 'flow_xy_nan' and preprocess_mode == 'flow_xy':
            image_arr = image_arr[:, :, :2]

        elif (load_mode == 'depth' and preprocess_mode == 'disparity' or
              load_mode == 'disparity' and preprocess_mode == 'depth'):
            image_arr = 1.0 / image_arr

        else:
            print(f'Can not perform casting from {load_mode} to {preprocess_mode}')
            raise NotImplemented

        return image_arr

    def _get_preprocessed_image(self, fname, load_mode, preprocess_mode):
        fpath = os.path.join(self.directory, fname)
        if (self.cached_images is not None) and (fpath in self.cached_images):
            image_arr = self.cached_images[fpath]
        else:
            image_arr = self._load_image(fpath, load_mode)
            image_arr = self._preprocess_image(image_arr, load_mode, preprocess_mode)

            if image_arr is not None:
                if (self.cached_images is not None) and \
                        (not self.stop_caching) and \
                        (len(self.cached_images) % 1000 == 0):
                    self.stop_caching = self._check_stop_caching()
                if (self.cached_images is not None) and (not self.stop_caching):
                    self.cached_images[fpath] = image_arr

        if image_arr is None:
            return None

        return image_arr.copy()

    def _init_batch(self, cols, index_array, return_confidences=False):
        batch = []

        for col in cols:
            if col in self.image_cols:
                batch.append(
                    np.zeros((len(index_array),) + self.image_shapes[col], dtype=self.dtype))
            else:
                values = self.df[col].values[index_array]
                if return_confidences:
                    values = np.stack((values, np.ones_like(values)), axis=1)
                batch.append(values)

        return batch

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = self._init_batch(self.x_cols, index_array)
        batch_y = self._init_batch(self.y_cols, index_array, return_confidences=self.return_confidences)

        # build batch of image data
        valid_samples = np.ones(len(index_array)).astype(bool)
        for index_in_batch, df_row_index in enumerate(index_array):
            seed = np.random.randint(1000000)
            for col, fname in self.df_images.iloc[df_row_index].iteritems():
                params = self.image_data_generator.get_random_transform(self.image_shapes[col], seed)
                image_arr = self._get_preprocessed_image(fname, self.load_mode[col], self.preprocess_mode[col])
                if image_arr is None:
                    valid_samples[index_in_batch] = False
                    continue
                image_arr = self.image_data_generator.apply_transform(image_arr, params)
                image_arr = self.image_data_generator.standardize(image_arr)
                if col in self.x_cols:
                    batch_x[self.x_cols.index(col)][index_in_batch] = image_arr
                if col in self.y_cols:
                    batch_y[self.y_cols.index(col)][index_in_batch] = image_arr

        batch_x = [features[valid_samples] for features in batch_x]
        batch_y = [target[valid_samples] for target in batch_y]

        if np.sum(valid_samples) < 0.5 * len(index_array):
            print('Batch is too small: {} samples'.format(np.sum(valid_samples)))
        return batch_x, batch_y

    def next(self):
        '''For python 2.x.
        # Returns
            The next batch.
        '''
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self._get_batches_of_transformed_samples(index_array)
