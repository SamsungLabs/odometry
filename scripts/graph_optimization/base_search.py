import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.graph_optimization import g2o_configs
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
    """
    This class is base for any g2o optimization algorithm. It defines all function that load predictions,
    gt_trajectories configs for optimization.
    Child classes just need to implement abstract method "search".
    """
    def __init__(self):
        self.mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.std_cols = [c + '_confidence' for c in self.mean_cols]

    @staticmethod
    def get_coef_values():
        return [1., 2., 4.] + list(np.logspace(1, 6, num=6)) + [1e12]

    @staticmethod
    def get_default_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_root', type=str, required=True)
        parser.add_argument('--output_path', type=str, required=True)
        parser.add_argument('--config_type', type=str, required=True)
        parser.add_argument('--val_mode', type=str, default='last')
        return parser

    @staticmethod
    def get_gt_trajectory(dataset_root, trajectory_name):
        gt_df = pd.read_csv(os.path.join(dataset_root, trajectory_name, 'df.csv'))
        gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
        return gt_trajectory

    def get_predicted_df(self, multistride_paths):
        df_list = list()
        for stride, monostride_paths in multistride_paths.items():
            for path in monostride_paths:
                print(path)
                df = read_csv(path)
                for c in self.std_cols:
                    df[c] = 1
                if stride == 'loops':
                    df = df[df['diff'] > 49].reset_index()
                df_list.append(df)
        predicted_df = pd.concat(df_list, ignore_index=True)
        return predicted_df

    @staticmethod
    def get_group_id(multistride_paths):
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
        predictions = list(Path(prefix).rglob(f'*.csv'))
        last_dir = self.get_val_trajectory_path(predictions, mode='last').parent.parent
        print('last_dir', last_dir)
        val_trajectory_names = last_dir.joinpath('val').glob('*.csv')
        test_trajectory_names = Path(prefix).joinpath('test/test').glob('*.csv')
        trajectory_names = list(val_trajectory_names) + list(test_trajectory_names)
        trajectory_names = [trajectory_name.stem for trajectory_name in trajectory_names]
        handled_trajectory_names = list()
        for trajectory_name in trajectory_names:
            split = trajectory_name.split('_')
            if len(split) > 1 and is_int(split[0]):
                trajectory_name = '_'.join(split[1:])
            handled_trajectory_names.append(trajectory_name)
        assert len(trajectory_names) > 0
        return handled_trajectory_names

    @staticmethod
    def get_epoch_from_dirname(dirname):
        position = dirname.find('_val')
        if position == -1:
            raise RuntimeError(f'Could not find epoch number in {dirname}')
        return int(dirname[position - 3: position])

    @staticmethod
    def get_metric_from_dirname(dirname: str):
        position = dirname.find('_val')
        if position == -1:
            raise RuntimeError(f'Could not find epoch number in {dirname}')
        return dirname.split('_')[-1]

    @staticmethod
    def is_test(paths):
        check = np.zeros(len(paths))
        for index, path in enumerate(paths):
            if path.parent.parent.name == 'test':
                check[index] = True
            else:
                check[index] = False
        test = np.sum(check) == len(paths)
        val = np.sum(check) == 0

        if test:
            return True
        elif val:
            return False
        else:
            raise RuntimeError('Seems like test trajectories are in validation')

    @staticmethod
    def get_test_trajectory_path(paths):
        if len(paths) == 1:
            return paths[0].as_posix()
        else:
            raise RuntimeError('Found more than one trajectory in test')

    def get_val_trajectory_path(self, paths, mode):
        """Modes: 'last' or 'best'"""
        if mode == 'last':
            last_dir = None
            max_epoch = 0
            for path in paths:
                current_dir = path.parent.parent.name
                if current_dir == 'final':
                    return path
                epoch = self.get_epoch_from_dirname(current_dir)
                if epoch > max_epoch:
                    max_epoch = epoch
                    last_dir = path
            return last_dir
        elif mode == 'best':
            return min(paths, key=lambda x: self.get_metric_from_dirname(x.parent.parent.name))
        else:
            raise RuntimeError(f'Wrong mode for get_val_trajectory_path. Expected "last" or "best". Got {mode}.')

    def get_path(self, prefix, trajectory_name, stride, val_mode):
        paths = list(Path(prefix).rglob(f'{stride}_{trajectory_name}.csv'))
        if len(paths) == 0:
            paths = list(Path(prefix).rglob(f'*{trajectory_name}.csv'))
        if len(paths) == 0:
            raise RuntimeError(f'Could not find trajectory {trajectory_name} in dir {prefix}')

        if self.is_test(paths):
            return self.get_test_trajectory_path(paths)
        else:
            return self.get_val_trajectory_path(paths, mode=val_mode).as_posix()

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

    @staticmethod
    def get_rpe_mode(config):
        if 'kitti' in config['1'][0]:
            rpe_indices = 'kitti'
        else:
            rpe_indices = 'full'
        return rpe_indices

    def get_data(self,
                 config,
                 dataset_root,
                 trajectory_names,
                 val_mode='last'):
        X = []
        y = []
        groups = []

        for trajectory_name in trajectory_names:
            trajectory_paths = dict()
            for stride in config.keys():
                trajectory_paths[stride] = [self.get_path(prefix, trajectory_name, stride, val_mode) for prefix in
                                            config[stride]]

            predicted_df = self.get_predicted_df(trajectory_paths)
            group_id = self.get_group_id(trajectory_paths)
            gt_trajectory = self.get_gt_trajectory(dataset_root, trajectory_name)

            X.append(predicted_df)
            y.append(gt_trajectory)
            groups.append(group_id)
        return X, y, groups

    def start(self,
              dataset_root,
              config_type,
              n_jobs,
              n_iter,
              output_path=None,
              **kwargs):

        config = getattr(g2o_configs, config_type)
        trajectory_names = self.get_trajectory_names(config['1'][0])
        strides = [int(stride) for stride in config.keys() if stride != 'loops']
        rpe_indices = self.get_rpe_mode(config)

        X, y, groups = self.get_data(config=config,
                                     dataset_root=dataset_root,
                                     trajectory_names=trajectory_names)
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
        return result

    def search(self,
               X,
               y,
               groups,
               param_distributions,
               rpe_indices,
               n_iter,
               n_jobs=3,
               verbose=True,
               trajectory_names=None,
               **kwargs):
        raise RuntimeError('This is the method of abstract class')
