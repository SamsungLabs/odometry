import os
import json
import warnings
import pickle
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras_preprocessing.image import ImageDataGenerator

from odometry.data_manager.generator import ExtendedDataFrameIterator


class GeneratorFactory:
    def __init__(self,
                 dataset_root,
                 csv_name,
                 train_trajectories=None,
                 val_trajectories=None,
                 test_trajectories=None,
                 x_col=('path_to_rgb', 'path_to_rgb_next'),
                 y_col=('euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z'),
                 image_col=('path_to_rgb', 'path_to_rgb_next'),
                 train_generator_args=None,
                 val_generator_args=None,
                 test_generator_args=None,
                 validate_on_train_trajectory=False,
                 val_ratio=0.0,
                 number_of_folds=None,
                 fold_index=0,
                 train_sampling_step=1,
                 val_sampling_step=1,
                 test_sampling_step=1,
                 batch_size=128,
                 cached_images=None,
                 *args, **kwargs):

        params = locals()
        params.pop('self', None)
        mlflow.log_params({'generator.' + k: repr(v) for k, v in params.items() if 'trajectories' not in k})

        self.dataset_root = dataset_root

        dataset_config_path = os.path.join(dataset_root, 'prepare_dataset.json')
        try:
            with open(dataset_config_path, 'r') as f:
                dataset_config = json.load(f)
                mlflow.log_param('depth_checkpoint', dataset_config['depth_checkpoint'])
                mlflow.log_param('optical_flow_checkpoint', dataset_config['optical_flow_checkpoint'])
        except FileNotFoundError:
            warnings.warn('WARNING!!!. No prepare_dataset.json for this dataset. You need to rerun prepare_dataset.py'
                          f'for this dataset. Path {dataset_config_path}', UserWarning)
            mlflow.log_param('depth_checkpoint', None)
            mlflow.log_param('optical_flow_checkpoint', None)

        self.csv_name = csv_name

        self.x_col = list(x_col)
        self.y_col = list(y_col)
        self.image_col = list(image_col)

        self.batch_size = batch_size

        assert validate_on_train_trajectory == bool(val_ratio)
        if validate_on_train_trajectory:
            assert val_trajectories is None
            val_trajectories = train_trajectories

        self.train_trajectories = train_trajectories
        self.val_trajectories = val_trajectories
        self.test_trajectories = test_trajectories

        self.df_train = self._get_multi_df_dataset(self.train_trajectories)
        self.df_val = self._get_multi_df_dataset(self.val_trajectories)
        self.df_test = self._get_multi_df_dataset(self.test_trajectories)

        if number_of_folds is not None:
            val_ratio = 1. / number_of_folds

        if val_ratio:
            val_samples = int(np.ceil(val_ratio * len(self.df_val)))  # upper-round to cover all dataset with k folds
            start = val_samples * fold_index
            end = start + val_samples
            print(f'fold #{fold_index}: validate on samples {start} -- {end} (out of {len(self.df_val)})')
            self.df_train = pd.concat([self.df_train[:start], self.df_train[end:]])
            self.df_val = self.df_val[start:end]

        self.df_train = self.df_train.iloc[::train_sampling_step] if self.df_train is not None else None

        self.df_val = self.df_val.iloc[::val_sampling_step] if self.df_val is not None else None

        self.df_test = self.df_test.iloc[::test_sampling_step] if self.df_test is not None else None

        self.train_generator_args = train_generator_args or {}

        self.val_generator_args = val_generator_args or {}

        self.test_generator_args = test_generator_args or {}

        self.args = args
        self.kwargs = kwargs

        self.cached_images = cached_images
        if type(self.cached_images) == str:
            self.load_cache(self.cached_images)

        self.input_shapes = self.get_train_generator().input_shapes \
            if self.train_trajectories else self.get_val_generator().input_shapes

    def _get_multi_df_dataset(self, trajectories):

        df = None

        if not trajectories:
            return df

        for trajectory_name in tqdm(trajectories):
            current_df = pd.read_csv(os.path.join(self.dataset_root, trajectory_name, self.csv_name))
            current_df[self.image_col] = trajectory_name + '/' + current_df[self.image_col]
            current_df['trajectory_id'] = trajectory_name
            df = current_df if df is None else df.append(current_df, sort=False)

        df.index = range(len(df))
        return df

    def load_cache(self, cache_file):
        try:
            with open(cache_file, 'rb') as cache_fp:
                self.cached_images = pickle.load(cache_fp)
        except:
            print(f'Failed to load cached images from {cache_file}, initialized empty cache')
            self.cached_images = {}
        else:
            print(f'Successfully loaded cached images from {cache_file}')

    def dump_cache(self, cache_file):
        with open(cache_file, 'wb') as cache_fp:
            pickle.dump(self.cached_images, cache_fp)
        print(f'Saved cached images to {cache_file}')

    def _get_generator(self, dataframe, generator_args, trajectory=False):

        if dataframe is None:
            return None

        if trajectory:
            shuffle = False
            filter_invalid = False
        else:
            shuffle = True
            filter_invalid = True

        return ExtendedDataFrameIterator(
            dataframe,
            self.dataset_root,
            ImageDataGenerator(**generator_args),
            x_col=self.x_col,
            y_col=self.y_col,
            image_col=self.image_col,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=42,
            interpolation='nearest',
            cached_images=self.cached_images,
            filter_invalid=filter_invalid,
            *self.args, **self.kwargs)

    def get_train_generator(self):
        return self._get_generator(self.df_train, self.train_generator_args)

    def get_val_generator(self):
        return self._get_generator(self.df_val, self.val_generator_args, trajectory=True)

    def get_test_generator(self):
        return self._get_generator(self.df_test, self.test_generator_args, trajectory=True)
