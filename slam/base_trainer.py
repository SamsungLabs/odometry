import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from scripts.aggregation import configs
from slam.utils import is_int
from slam.utils import read_csv

from slam.linalg import RelativeTrajectory


class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups):
        if isinstance(groups, list):
            groups = np.array(groups)
        elif not isinstance(groups, np.ndarray):
            raise RuntimeError('groups has not array like type')
        train = np.where(groups == 0)[0]

        test_ind = groups == 1
        if np.sum(test_ind) == 0:
            test = [0]
        else:
            test = np.where(groups == 1)[0]
        print(f'train split {train}')
        print(f'test split {test}')
        yield (train, test)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class Search:

    @staticmethod
    def get_coef_values():
        return [1., 2., 4., 8., ] + list(np.logspace(1, 6, num=6)) + [1e12]

    @staticmethod
    def get_default_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_root', type=str, required=True)
        parser.add_argument('--output_path', type=str, required=True)
        parser.add_argument('--config_type', type=str, required=True)
        parser.add_argument('--n_jobs', type=int, default=3)
        parser.add_argument('--n_iter', type=int, default=1)
        parser.add_argument('--coef', type=int, nargs='*', default=None)
        parser.add_argument('--coef_loop', type=int, nargs='*', default=None)
        parser.add_argument('--loop_threshold', type=int, nargs='*', default=None)
        parser.add_argument('--rotation_scale', type=float, nargs='*', default=None)
        parser.add_argument('--max_iterations', type=int, nargs='*', default=None)
        return parser

    def get_gt_trajectory(self, dataset_root, trajectory_name):
        gt_df = pd.read_csv(os.path.join(dataset_root, trajectory_name, 'df.csv'))
        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
        return gt_trajectory

    def get_predicted_df(self, multistride_paths):
        df_list = list()
        for stride, monostride_paths in multistride_paths.items():
            for path in monostride_paths:
                df = read_csv(path)
                if stride == 'loops':
                    df = df[df['diff'] > 49].reset_index()
                df_list.append(df)
        predicted_df = pd.concat(df_list, ignore_index=True)
        return predicted_df

    def get_group_id(self, multistride_paths):
        parent_dir = os.path.basename(os.path.dirname(multistride_paths['1'][0]))
        if parent_dir == 'val':
            group_id = 0
        elif parent_dir == 'test':
            group_id = 1
        else:
            raise RuntimeError(
                f'Unexpected parent dir of prediction {multistride_paths["1"][0]}. Parent dir must "val" or "test"')
        return group_id

    def get_trajectory_names(self, prefix):
        val_dirs = list(Path(prefix).glob(f'*val*'))
        paths = [val_dir.as_posix() for val_dir in val_dirs]
        try:
            last_dir = max(paths, key=lambda x: self.get_epoch_from_dirname(x))
        except Exception as e:
            raise RuntimeError(f'Could not find val directories in paths: {paths}', e)
        val_trajectory_names = Path(last_dir).joinpath('val').glob('*.csv')
        test_trajectory_names = Path(prefix).joinpath('test/test').glob('*.csv')
        trajectory_names = list(val_trajectory_names) + list(test_trajectory_names)
        trajectory_names = [trajectory_name.stem for trajectory_name in trajectory_names]
        # Handaling bug with strides in names of trajectory
        handled_trajectory_names = list()
        for trajectory_name in trajectory_names:
            split = trajectory_name.split('_')
            if len(split) > 1 and is_int(split[0]):
                trajectory_name = '_'.join(split[1:])
            handled_trajectory_names.append(trajectory_name)
        assert len(trajectory_names) > 0
        return handled_trajectory_names

    def get_epoch_from_dirname(self, dirname):
        position = dirname.find('_val_RPE')
        if position == -1:
            raise RuntimeError(f'Could not find epoch number in {dirname}')
        return int(dirname[position - 3: position])

    def get_path(self, prefix, trajectory_name):
        paths = list(Path(prefix).rglob(f'*{trajectory_name}.csv'))

        if len(paths) == 1:
            return paths[0].as_posix()
        elif len(paths) > 1:
            return max(paths, key=lambda x: self.get_epoch_from_dirname(x.parent.parent.name)).as_posix()
        else:
            raise RuntimeError(f'Could not find trajectory {trajectory_name} in dir {prefix}')

    def get_coefs(self, vals, current_level, max_depth):
        if current_level == max_depth:
            coefs = list()
            for v in vals:
                coefs.append([v])
            return coefs
        else:
            coefs = self.get_coefs(vals, current_level + 1, max_depth)
            new_coefs = list()
            for v in vals:
                for c in coefs:
                    new_coefs.append([v] + c)
            return new_coefs

    def start(self,
              dataset_root,
              config_type,
              n_jobs,
              n_iter,
              output_path=None,
              **kwargs):

        config = getattr(configs, config_type)
        trajectory_names = self.get_trajectory_names(config['1'][0])
        strides = [int(stride) for stride in config.keys() if stride != 'loops']
        if 'kitti' in config['1'][0]:
            rpe_indices = 'kitti'
        else:
            rpe_indices = 'full'

        X = []
        y = []
        groups = []

        for trajectory_name in trajectory_names:
            trajectory_paths = dict()
            for k, v in config.items():
                trajectory_paths[k] = [self.get_path(prefix, trajectory_name) for prefix in config[k]]

            predicted_df = self.get_predicted_df(trajectory_paths)
            group_id = self.get_group_id(trajectory_paths)
            gt_trajectory = self.get_gt_trajectory(dataset_root, trajectory_name)

            X.append(predicted_df)
            y.append(gt_trajectory)
            groups.append(group_id)

        coef_values = self.get_coef_values()
        if kwargs['coef']:
            coefs = [kwargs['coef']]
        else:
            coefs = self.get_coefs(coef_values, 1, len(config.keys()) - 1)

        param_distributions = {
            'coef': [dict(zip(strides, c)) for c in coefs],
            'coef_loop': kwargs['coef_loop'] or coef_values,
            'loop_threshold': kwargs['loop_threshold'] or [50, 100],
            'rotation_scale': kwargs['rotation_scale'] or np.logspace(-10, 0, 11, base=2),
            'max_iterations': kwargs['max_iterations'] or [1000]
        }

        print(param_distributions)

        result = self.search(X,
                             y,
                             groups,
                             param_distributions,
                             rpe_indices=rpe_indices,
                             n_jobs=n_jobs,
                             n_iter=n_iter,
                             verbose=True,
                             trajectory_names=trajectory_names,
                             **kwargs)

        if output_path:
            result.to_csv(output_path)

    def search(self,
               X,
               y,
               groups,
               param_distributions,
               rpe_indices,
               n_iter,
               n_jobs=3,
               verbose=True):
        raise RuntimeError('This is the method of abstract class')