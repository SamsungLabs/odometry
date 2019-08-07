import os

import numpy as np
import pandas as pd


import PIL
import keras_preprocessing.image as keras_image
from keras_preprocessing.image import ImageDataGenerator
import psutil


from .generator_utils import (get_channels_count,
                              fill_flow,
                              fill_depth,
                              load_pil_image,
                              resize_image)


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
                 flow_fill_method='random',
                 depth_multiplicator=1.0,
                 cached_images=None,
                 filter_invalid=True,
                 max_memory_consumption=0.8,
                 return_confidences=False,
                 trajectory_id='',
                 include_last=False):
        super().set_processing_attrs(image_data_generator,
                                     target_size,
                                     'rgb',
                                     data_format,
                                     save_to_dir=False,
                                     save_prefix='',
                                     save_format='png',
                                     subset=subset,
                                     interpolation=interpolation)

        self.df = dataframe
        self._include_last() if include_last else None
        self.df[image_col] = self.df[image_col].astype(str)

        self.directory = directory
        self.dtype = dtype
        self.samples = len(self.df)
        self.df_images = self.df[image_col]

        self.x_cols = [x_col] if isinstance(x_col, str) else x_col
        self.y_cols = [y_col] if isinstance(y_col, str) else y_col
        image_col = image_col or []
        self.image_cols = [image_col] if isinstance(image_col, str) else image_col

        print(self.image_cols, self.x_cols, self.y_cols, self.df.columns)
        assert set(self.image_cols) <= (set(self.x_cols) | set(self.y_cols))
        assert (set(self.x_cols) | set(self.y_cols)) <= set(self.df.columns)

        self.df[self.image_cols] = self.df[self.image_cols].astype(str)

        if isinstance(load_mode, str):
            self.load_mode = {col: load_mode for col in self.image_cols}
        else:
            assert len(load_mode) == len(self.image_cols)
            self.load_mode = dict(zip(self.image_cols, load_mode))

        if isinstance(preprocess_mode, str) or preprocess_mode is None:
            self.preprocess_mode = {col: preprocess_mode for col in self.image_cols}
        else:
            assert len(preprocess_mode) == len(self.image_cols)
            self.preprocess_mode = dict(zip(self.image_cols, preprocess_mode))

        self.image_shapes = {col: self.target_size + (get_channels_count(self.preprocess_mode[col]),) \
                             for col in self.image_cols}

        self._fill_flow = fill_flow(method=flow_fill_method)
        self._fill_depth = fill_depth(method='random')
        self.depth_multiplicator = depth_multiplicator

        self.set_cache(cached_images)
        self.max_memory_consumption = max_memory_consumption
        self.stop_caching = False

        self.filter_invalid = filter_invalid

        self.return_confidence = return_confidences

        self.trajectory_id = trajectory_id

        super(ExtendedDataFrameIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)

    @property
    def channel_counts(self):
        return [get_channels_count(self.preprocess_mode[col]) \
                for col in self.x_cols if col in self.image_cols]

    @property
    def input_shapes(self):
        return [self.image_shapes.get(col, (1,)) for col in self.x_cols]


    def _include_last(self):
        index = len(self.df)

        for col in self.df.columns:
            col_next = col + '_next'
            if col_next in self.df.columns:
                self.df.at[index, col] = self.df[col_next].iloc[index - 1]

    def _check_stop_caching(self):
        self.stop_caching = False
        if (self.cached_images is not None) and (len(self.cached_images) % 1000 == 0):
             self.stop_caching = psutil.virtual_memory().percent / 100 > self.max_memory_consumption

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
            pil_mode = None
            if load_mode == 'grayscale':
                pil_mode = 'L'
            elif load_mode == 'rgba':
                pil_mode = 'RGBA'
            elif load_mode == 'rgb':
                pil_mode = 'RGB'

            image_arr = load_pil_image(fpath, mode=pil_mode)

        if len(image_arr.shape) == 2:
            image_arr = np.expand_dims(image_arr, -1)

        if load_mode == 'seven_flow':
            image_arr = image_arr.transpose(1, 2, 0)

        if load_mode == 'zt_yr_dof_flow':
            image_arr = image_arr[[2,5],:,:].transpose(1, 2, 0)

        if load_mode == 'xyt_xyr_dof_flow':
            image_arr = image_arr[[0,1,4,5],:,:].transpose(1, 2, 0)

        image_arr = resize_image(image_arr,
                                 self.target_size,
                                 data_format=self.data_format,
                                 mode=self.interpolation)
        return image_arr

    def _preprocess_image(self, image_arr, load_mode, preprocess_mode):
        if load_mode == 'depth':
            image_arr *= self.depth_multiplicator

        if load_mode == 'disparity':
            image_arr /= self.depth_multiplicator

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
                self._check_stop_caching()
                if (self.cached_images is not None) and (not self.stop_caching):
                    self.cached_images[fpath] = image_arr

        if image_arr is None:
            return None

        return image_arr.copy()

    def _init_batch(self, cols, index_array):
        batch = []

        for col in cols:
            if col in self.image_cols:
                batch.append(
                    np.zeros((len(index_array),) + self.image_shapes[col], dtype=self.dtype))
            else:
                values = self.df[col].values[index_array]
                if self.return_confidence:
                    values = np.stack((values, np.ones_like(values)), axis=1)
                batch.append(values)

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
