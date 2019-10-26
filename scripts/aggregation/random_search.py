import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

import __init_path__
import env

from slam.linalg import RelativeTrajectory
from slam.aggregation import random_search
from slam.utils import read_csv
from . import configs

def get_epoch_from_dirname(dirname):
    position = dirname.find('_val_RPE')
    if position == -1:
        raise RuntimeError(f'Could not find epoch number in {dirname}')
    return int(dirname[position - 3: position])


def get_path(prefix, trajectory_name):
    paths = list(Path(prefix).rglob(f'*{trajectory_name}.csv'))

    if len(paths) == 1:
        return paths[0].as_posix()
    elif len(paths) > 1:
        return max(paths, key=lambda x: get_epoch_from_dirname(x.parent.parent.name)).as_posix()
    else:
        raise RuntimeError(f'Could not find trajectory {trajectory_name} in dir {prefix}')


def get_trajectory_names(prefix):
    val_dirs = list(Path(prefix).glob(f'*val*'))
    paths = [val_dir.as_posix() for val_dir in val_dirs]
    last_dir = max(paths, key=lambda x: get_epoch_from_dirname(x))
    val_trajectory_names = Path(last_dir).joinpath('val').glob('*.csv')
    test_trajectory_names = Path(prefix).joinpath('test/test').glob('*.csv')
    trajectory_names = list(val_trajectory_names) + list(test_trajectory_names)
    trajectory_names = [trajectory_name.stem for trajectory_name in trajectory_names]
    trajectory_names = ['_'.join(trajectory_name.split('_')[1:]) for trajectory_name in trajectory_names]  # Bug handling
    assert len(trajectory_names) > 0
    return trajectory_names


def get_predicted_df(multistride_paths):
    df_list = list()
    for stride, monostride_paths in multistride_paths.items():
        for path in monostride_paths:
            df = read_csv(path)

            if stride == 'loops':
                df = df[df['diff'] > 49].reset_index()

            df_list.append(df)

    predicted_df = pd.concat(df_list, ignore_index=True)

    parent_dir = os.path.basename(os.path.dirname(multistride_paths['1'][0]))
    if parent_dir == 'val':
        group_id = 0
    elif parent_dir == 'test':
        group_id = 1
    else:
        raise RuntimeError(f'Unexpected parent dir of prediction {multistride_paths["1"][0]}. Parent dir must "val" or "test"')

    return predicted_df, group_id


def get_gt_trajectory(dataset_root, trajectory_name):
    gt_df = pd.read_csv(os.path.join(dataset_root, trajectory_name, 'df.csv'))
    gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
    return gt_trajectory


def main(dataset_root,
         strides,
         paths,
         n_jobs,
         n_iter,
         output_path=None,
         config_type=None,
         **kwargs):
    config = None
    if config_type is None:
        assert len(strides) == len(paths)
    else:
        config = getattr(configs, config_type)

    X = []
    y = []
    groups = []

    trajectory_names = get_trajectory_names(paths[0])

    for trajectory_name in trajectory_names:
        if config is None:
            trajectory_paths = {stride: [get_path(prefix, trajectory_name)] for stride, prefix in zip(strides, paths)}
        else:
            trajectory_paths = dict()
            for k, v in config.items():
                trajectory_paths[k] = [get_path(prefix, trajectory_name) for prefix in config[k]]

        predicted_df, group_id = get_predicted_df(trajectory_paths)
        gt_trajectory = get_gt_trajectory(dataset_root, trajectory_name)

        X.append(predicted_df)
        y.append(gt_trajectory)
        groups.append(group_id)                 

    strides = [int(stride) for stride in strides if stride != 'loops']

    if kwargs['coef']:
        coefs = [kwargs['coef']]
    else:
        coefs = list()
        coefs.append([1] * len(strides))
        coefs.append(list(np.arange(1, len(strides) + 1, 1)))
        coefs.append(list(np.arange(1, len(strides)*2 + 1, 2)))
        coefs.append(list(np.arange(1, len(strides)*4 + 1, 4)))
        for i in range(1, len(strides)):
            c = [1] * (len(strides) - i) + [1000000] * i
            coefs.append(c)
                           
        for i in range(len(strides)):
            c = [1000000] * (len(strides))
            c[i] = 1
            coefs.append(c)

    param_distributions = {
        'coef': [dict(zip(strides, c)) for c in coefs],
        'coef_loop': kwargs['coef_loop'] or [1, 100, 300, 500, 1000000],
        'loop_threshold': kwargs['loop_threshold'] or [50, 100],
        'rotation_scale': kwargs['rotation_scale'] or np.logspace(-10, 0, 11, base=2),
        'max_iterations': kwargs['max_iterations'] or [1000]
    }

    print(param_distributions)
                           
    if 'kitti' in paths[0]:
        rpe_indices = 'kitti'
    else:
        rpe_indices = 'full'
                        
    result = random_search(X,
                           y,
                           groups,
                           param_distributions,
                           rpe_indices=rpe_indices,
                           n_jobs=n_jobs,
                           n_iter=n_iter,
                           verbose=True)

    if output_path:
        result.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--strides', type=str, nargs='+')
    parser.add_argument('--paths', type=str, nargs='+')
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--coef', type=int, nargs='*', default=None)
    parser.add_argument('--coef_loop', type=int, nargs='*', default=None)
    parser.add_argument('--loop_threshold', type=int, nargs='*', default=None)
    parser.add_argument('--rotation_scale', type=float, nargs='*', default=None)
    parser.add_argument('--max_iterations', type=int, nargs='*', default=None)
    parser.add_argument('--config_type',
                        type=str,
                        choices=['euroc',
                                 'euroc_clr',
                                 'euroc_mn',
                                 'kitti',
                                 'kitti_clr',
                                 'kitti_mn',
                                 'tum',
                                 'tum_clr',
                                 'tum_mn'],
                        default=None)
    args = parser.parse_args()
    assert (args.strides is not None and args.paths is not None) or args.config_type is not None
    main(**vars(args))
