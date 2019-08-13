import os
import json
import warnings
import pickle
import mlflow
import tqdm
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator

from slam.data_manager.generator import ExtendedDataFrameIterator
from slam.utils import mlflow_logging


class GeneratorFactory:

    @mlflow_logging(ignore=('train_trajectories', 'val_trajectories', 'test_trajectories'), prefix='gen_factory.')
    def __init__(self,
                 dataset_root,
                 csv_name='df.csv',
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
                 train_strides=1,
                 val_strides=1,
                 test_strides=1,
                 batch_size=128,
                 cached_images=None,
                 *args, **kwargs):

        self.dataset_root = dataset_root

        self._log_dataset_params()

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

        self.df_train, self.df_train_as_is = self._get_multi_df_dataset(self.train_trajectories, 'train', strides=train_strides)
        self.df_val, _ = self._get_multi_df_dataset(self.val_trajectories, 'val', strides=val_strides)
        self.df_test, _ = self._get_multi_df_dataset(self.test_trajectories, 'test', strides=test_strides)

        if number_of_folds is not None:
            val_ratio = 1. / number_of_folds

        if val_ratio:
            size = len(self.df_train)
            val_size = int(np.ceil(val_ratio * size)) # upper-round to cover all dataset with k folds
            start = val_size * fold_index
            end = val_size * (fold_index + 1)
            mask = np.zeros(size)
            mask[start:end] = 1
            print(f'fold #{fold_index}: validate on samples {start} -- {end} (out of {size})')
            self.df_train = self.df_train.iloc[~mask]
            self.df_val = self.df_val.iloc[mask]

        self.train_generator_args = train_generator_args or {}
        self.val_generator_args = val_generator_args or {}
        self.test_generator_args = test_generator_args or {}

        self.args = args
        self.kwargs = kwargs

        self.cached_images = cached_images
        if type(self.cached_images) == str:
            self.load_cache(self.cached_images)

    @property
    def input_shapes(self):
        return (self.get_train_generator().input_shapes if self.train_trajectories
                else self.get_val_generator().input_shapes)

    def _log_dataset_params(self):
        if mlflow.active_run():

            dataset_config_path = os.path.join(self.dataset_root, 'prepare_dataset.json')
            try:
                with open(dataset_config_path, 'r') as f:
                    dataset_config = json.load(f)
                    mlflow.log_param('depth_checkpoint', dataset_config['depth_checkpoint'])
                    mlflow.log_param('optical_flow_checkpoint', dataset_config['optical_flow_checkpoint'])
            except FileNotFoundError:
                warnings.warn('WARNING!!!. No prepare_dataset.json for this dataset. You need to rerun '
                              f'prepare_dataset.py for this dataset. Path {dataset_config_path}', UserWarning)
                mlflow.log_param('depth_checkpoint', None)
                mlflow.log_param('optical_flow_checkpoint', None)

    def _get_multi_df_dataset(self, trajectories, subset, strides=1):
        df = None
        df_as_is = None
        if not trajectories:
            return df, df_as_is

        strides = [strides] * len(trajectories) if isinstance(strides, int) else strides

        for trajectory_name, stride in tqdm.tqdm(zip(trajectories, strides),
                                                 total=len(trajectories),
                                                 desc=f'Collect {subset} trajectories'):
            current_df = pd.read_csv(os.path.join(self.dataset_root, trajectory_name, self.csv_name))
            current_df[self.image_col] = trajectory_name + '/' + current_df[self.image_col]

            for image_col in self.image_col:
                image_col_next = image_col + '_next'
                if image_col_next in current_df.columns:
                    current_df[image_col_next] = trajectory_name + '/' + current_df[image_col_next]

            current_df['trajectory_id'] = trajectory_name
            current_df['stride'] = stride
            df = current_df if df is None else df.append(current_df, sort=False)

            current_df_as_is = current_df.iloc[::stride]
            df_as_is = current_df_as_is if df_as_is is None else df_as_is.append(current_df_as_is, sort=False)

        df.index = range(len(df))
        df_as_is.index = range(len(df_as_is))
        return df, df_as_is

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

    def _get_generator_from_dataframe(self,
                                      dataframe,
                                      generator_args,
                                      trajectory=False,
                                      include_last=False,
                                      trajectory_id=''):

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
            include_last=include_last,
            trajectory_id=trajectory_id,
            *self.args, **self.kwargs)

    def _get_generators_list(self, dataframe, generator_args, trajectories, include_last=False):

        if dataframe is None:
            return None

        generators = list()
        for trajectory in trajectories:

            trajectory_dataframe = dataframe[dataframe['trajectory_id'] == trajectory].reset_index(drop=True)
            generator = self._get_generator_from_dataframe(trajectory_dataframe,
                                                           generator_args,
                                                           trajectory=True,
                                                           include_last=include_last,
                                                           trajectory_id=trajectory)
            generators.append(generator)

        return generators

    def _get_generator(self,
                       dataframe,
                       generator_args,
                       trajectories,
                       as_is=False,
                       as_list=False,
                       include_last=False):
        if as_list:
            return self._get_generators_list(dataframe,
                                             generator_args,
                                             trajectories,
                                             include_last=include_last)
        else:
            return self._get_generator_from_dataframe(dataframe,
                                                      self.train_generator_args,
                                                      trajectory=as_is,
                                                      include_last=include_last)

    def get_train_generator(self, as_is=False, as_list=False, include_last=False):

        df_train = self.df_train_as_is if as_is else self.df_train

        return self._get_generator(df_train,
                                   self.train_generator_args,
                                   self.train_trajectories,
                                   as_is=as_is,
                                   as_list=as_list,
                                   include_last=include_last)

    def get_val_generator(self, as_list=False, include_last=False):

        return self._get_generator(self.df_val,
                                   self.val_generator_args,
                                   self.val_trajectories,
                                   as_is=True,
                                   as_list=as_list,
                                   include_last=include_last)

    def get_test_generator(self, as_list=False, include_last=False):

        return self._get_generator(self.df_test,
                                   self.test_generator_args,
                                   self.test_trajectories,
                                   as_is=True,
                                   as_list=as_list,
                                   include_last=include_last)
