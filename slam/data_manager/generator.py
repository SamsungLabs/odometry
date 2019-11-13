import os
import psutil
import numpy as np
from pathlib import Path
import keras_preprocessing.image as keras_image

from slam.utils import (get_channels_num,
                        get_fill_fn,
                        load_image_arr,
                        resize_image_arr)

from slam.linalg import Intrinsics, create_optical_flow_from_rt


def get_proba_fn(mode, proba=None, steps=None):
    if callable(proba):
        proba_fn = proba
    if mode == 'constant':
        proba_fn = lambda x: proba
    elif mode == 'linear':
        proba_fn = lambda x: float(x / steps)
    elif mode == 'exp':
        proba_fn = lambda x: np.exp(1 - (steps + 1) / (x + 1))
    elif mode == 'r_linear':
        proba_fn = lambda x: 1 - float(x / steps)
    elif mode == 'r_exp':
        proba_fn = lambda x: np.exp(1 - (steps + 1) / (x + 1))
    else:
        raise ValueError(f'Unknown mode option: "{mode}"')
    return proba_fn


def sample_coordinates(image_size):
    return np.random.uniform(low=(0, 0), high=image_size).astype(int)


class ExtendedDataFrameIterator(keras_image.iterator.BatchFromFilesMixin, keras_image.Iterator):
    def __init__(self,
                 dataframe,
                 directory,
                 image_data_generator,
                 x_col='x_col',
                 y_col='y_col',
                 image_col=None,
                 weight_col=None,
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
                 fill_flow_method='random',
                 fill_depth_method='random',
                 depth_multiplicator=1.0,
                 cached_images=None,
                 filter_invalid=True,
                 max_memory_consumption=0.8,
                 placeholder=None,
                 trajectory_id='',
                 include_last=False,
                 min_frame_ind_diff=0,
                 max_frame_ind_diff=float('inf'),
                 generate_flow_by_rt_proba=0,
                 generate_flow_by_rt_mode='constant',
                 generate_distribution=None,
                 generate_percentile=None,
                 augment_with_rectangle_proba=0,
                 augment_with_rectangle_mode='constant',
                 epochs=100,
                 **kwargs):

        if target_size == -1:
            path_to_first_image = os.path.join(directory, dataframe[image_col].iloc[0].values[0])
            first_image = load_image_arr(path_to_first_image)
            target_size = first_image.shape[:2]

        super().set_processing_attrs(image_data_generator,
                                     target_size,
                                     'rgb',
                                     data_format,
                                     save_to_dir=False,
                                     save_prefix='',
                                     save_format='png',
                                     subset=subset,
                                     interpolation=interpolation)

        dataframe['to_index'] = dataframe['path_to_rgb_next'].apply(lambda x: int(Path(x).stem))
        dataframe['from_index'] = dataframe['path_to_rgb'].apply(lambda x: int(Path(x).stem))
        index_diff = dataframe['to_index'] - dataframe['from_index']
        is_in_interval = (min_frame_ind_diff < index_diff) & (index_diff < max_frame_ind_diff)
        dataframe = dataframe[is_in_interval].reset_index(drop=True)

        self.df = dataframe
        self._include_last() if include_last else None

        self.directory = directory
        self.dtype = dtype
        self.samples = len(self.df)

        super(ExtendedDataFrameIterator, self).__init__(self.samples,
                                                        batch_size,
                                                        shuffle,
                                                        seed)

        self.x_cols = [x_col] if isinstance(x_col, str) else x_col.copy()
        self.y_cols = [y_col] if isinstance(y_col, str) else y_col.copy()
        assert (set(self.x_cols) | set(self.y_cols)) <= set(self.df.columns)

        self.return_cols = self.y_cols[:]

        weight_col = weight_col or []
        self.w_cols = [weight_col] if isinstance(weight_col, str) else weight_col

        image_col = image_col or []
        self.image_cols = [image_col] if isinstance(image_col, str) else image_col
        self.df[self.image_cols] = self.df[self.image_cols].astype(str)
        self.df_images = self.df[self.image_cols]

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

        self.image_shapes = {col: self.target_size + (get_channels_num(self.preprocess_mode[col]),)
                             for col in self.image_cols}

        self.fill_flow_fn = get_fill_fn(fill_flow_method, nan_value=np.nan, mean=0, std=1)
        self.fill_depth_fn = get_fill_fn(fill_depth_method, nan_value=0)
        self.depth_multiplicator = depth_multiplicator
        self.filter_invalid = filter_invalid

        self.trajectory_id = trajectory_id

        self.placeholder = placeholder or []
        for p in self.placeholder:
            self.return_cols.extend([col + '_' + p for col in self.y_cols])

        self.dof_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.df_dofs = self.df[self.dof_cols]

        # augmentation
        self.generate_flow_by_rt_proba_fn = get_proba_fn(generate_flow_by_rt_mode,
                                                         generate_flow_by_rt_proba,
                                                         steps=len(self) * epochs)
        self.generate_distribution = generate_distribution
        self.generate_percentile = generate_percentile

        if self.generate_distribution is not None:
            assert 'path_to_optical_flow' in self.x_cols
            assert any([col.endswith('depth') for col in self.image_cols])
            intrinsics_cols = ['f_x', 'f_y', 'c_x', 'c_y']
            self.df_intrinsics = self.df[intrinsics_cols]

            if self.generate_distribution == 'uniform':
                assert self.generate_percentile <= 100
                assert self.generate_percentile > 50
                self.gt_low_high_bounds = (
                    np.percentile(self.df_dofs, 100 - self.generate_percentile, axis=0),
                    np.percentile(self.df_dofs, self.generate_percentile, axis=0))

                print('Params of uniform distribution:')
                for i, col in enumerate(self.dof_cols):
                    print('\t', col, self.gt_low_high_bounds[0][i], self.gt_low_high_bounds[1][i])

            elif self.generate_distribution == 'normal':
                self.mean_std = list(zip(self.df_dofs.mean(axis=0), self.df_dofs.std(axis=0)))
                print('Params of normal distribution:')
                for i, col in enumerate(self.dof_cols):
                    print('\t', col, self.mean_std[i])
            elif self.generate_distribution == 'same':
                print('Generate flow from the same ground truth motion')
            else:
                raise ValueError(f'Unknown distribution: "{self.generate_distribution}"')

        self.augment_with_rectangle_proba_fn = get_proba_fn(augment_with_rectangle_mode,
                                                            augment_with_rectangle_proba,
                                                            steps=len(self) * epochs)

        self.batches_seen = 0

        self.cached_images = None
        self.set_cache(cached_images)
        self.max_memory_consumption = max_memory_consumption
        self.stop_caching = False

    @property
    def channel_counts(self):
        return [get_channels_num(self.preprocess_mode[col])
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

            image_arr = load_image_arr(fpath, mode=pil_mode)

        if len(image_arr.shape) == 2:
            image_arr = np.expand_dims(image_arr, -1)

        if load_mode == 'motion_maps':
            image_arr = image_arr.transpose((1, 2, 0))

        if load_mode == 'motion_maps_z':
            image_arr = image_arr[[2, 5]].transpose((1, 2, 0))

        if load_mode == 'motion_maps_xy':
            image_arr = image_arr[[0, 1, 4, 5]].transpose((1, 2, 0))

        image_arr = resize_image_arr(image_arr,
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
                image_arr = self.fill_depth_fn(image_arr)

        if load_mode == preprocess_mode:
            return image_arr

        if load_mode == 'flow_xy' and preprocess_mode == 'flow_xy_nan':
            isnan = (np.isnan(image_arr[:, :, 0]) | np.isnan(image_arr[:, :, 1])).astype(self.dtype)
            if isnan.any():
                image_arr[:, :, 0] = self.fill_flow_fn(image_arr[:, :, 0])
                image_arr[:, :, 1] = self.fill_flow_fn(image_arr[:, :, 1])

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

    def _init_batch(self, cols, index_array, add_placeholder=False):
        batch = []

        for col in cols:
            if col in self.image_cols:
                batch.append(
                    np.zeros((len(index_array),) + self.image_shapes[col], dtype=self.dtype))
            else:
                values = self.df[col].values[index_array]

                if add_placeholder:
                    ones = np.ones((len(values), len(self.placeholder)))
                    values = np.concatenate([np.expand_dims(values, -1), ones], axis=1)

                batch.append(values)

        return batch

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = self._init_batch(self.x_cols, index_array)
        batch_y = self._init_batch(self.y_cols, index_array, add_placeholder=len(self.placeholder) > 0)
        batch_w = self._init_batch(self.w_cols, index_array)

        generate_flow_by_rt_proba = self.generate_flow_by_rt_proba_fn(self.batches_seen)
        if generate_flow_by_rt_proba > 0:
            print(f'batch #{self.batches_seen} / {len(self)}: p={generate_flow_by_rt_proba}')

        # build batch of image data
        valid_samples = np.ones(len(index_array)).astype(bool)
        for index_in_batch, df_row_index in enumerate(index_array):
            generate_flow_by_rt = generate_flow_by_rt_proba > np.random.uniform()

            for col, fname in self.df_images.iloc[df_row_index].iteritems():
                if col == 'path_to_optical_flow' and generate_flow_by_rt:
                    continue

                image_arr = self._get_preprocessed_image(fname, self.load_mode[col], self.preprocess_mode[col])
                if image_arr is None:
                    valid_samples[index_in_batch] = False
                    continue

                if col.endswith('depth') and generate_flow_by_rt:
                    col = 'path_to_optical_flow'

                    if self.generate_distribution == 'uniform':
                        dofs = np.random.uniform(*(self.gt_low_high_bounds))
                    elif self.generate_distribution == 'normal':
                        dofs = np.array([np.random.normal(loc=mean, scale=std) for mean, std in self.mean_std])
                    elif self.generate_distribution == 'same':
                        dofs = self.df_dofs.iloc[df_row_index].values
                    else:
                        targets_row_index = np.random.randint(len(self.df))
                        dofs = self.df_dofs.iloc[targets_row_index].values

                    rotation_vector, translation_vector = dofs[:3], dofs[3:]

                    intrinsics_args = dict(self.df_intrinsics.iloc[df_row_index])
                    intrinsics_args.update({'width': image_arr.shape[1], 'height': image_arr.shape[0]})

                    image_arr = create_optical_flow_from_rt(image_arr[..., 0],
                                                            Intrinsics(**intrinsics_args),
                                                            rotation_vector,
                                                            translation_vector)
                    if image_arr is None:
                        valid_samples[index_in_batch] = False
                        continue

                    augment_with_rectangle_proba = self.augment_with_rectangle_proba_fn(self.batches_seen)
                    if augment_with_rectangle_proba > np.random.uniform():
                        y1, x1 = sample_coordinates(image_arr[..., 0].shape)
                        y2, x2 = sample_coordinates(image_arr[..., 0].shape)

                        y_dst, x_dst = min(y1, y2), min(x1, x2)
                        h, w = abs(y1 - y2), abs(x1 - x2)

                        y_src, x_src = sample_coordinates((image_arr.shape[0] - h, image_arr.shape[1] - w))
                        rectangle_src = image_arr[y_src:y_src + h, x_src:x_src + w]

                        noise = np.random.uniform(-0.1, 0.1)
                        image_arr[y_dst:y_dst + h, x_dst:x_dst + w] = rectangle_src + noise

                    for dof_name, dof_value in zip(self.dof_cols, dofs):
                        batch_y[self.y_cols.index(dof_name)][index_in_batch] = dof_value

                if col in self.x_cols:
                    batch_x[self.x_cols.index(col)][index_in_batch] = image_arr
                if col in self.y_cols:
                    batch_y[self.y_cols.index(col)][index_in_batch] = image_arr

        batch_x = [features[valid_samples] for features in batch_x]
        batch_y = [target[valid_samples] for target in batch_y]

        if np.sum(valid_samples) < 0.5 * len(index_array):
            print('Batch is too small: {} samples'.format(np.sum(valid_samples)))

        self.batches_seen += 1

        if batch_w:
            batch_w = [batch_w[0][valid_samples] for target in batch_y]
            return batch_x, batch_y, batch_w
        else:
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
