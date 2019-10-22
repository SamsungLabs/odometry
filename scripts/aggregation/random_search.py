import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

import __init_path__
import env

from slam.linalg import RelativeTrajectory
from scripts.aggregation import random_search


def get_path(prefix, trajectory_name):
    paths = list(Path(prefix).rglob(f'*{trajectory_name}.csv'))

    if len(paths) == 1:
        return paths[0].as_posix()
    elif len(paths) > 1:
        return max(paths,
                   key=lambda x: int(x.parent.parent.name[x.parent.parent.name.find('_val_RPE') - 3: x.parent.parent.name.find('_val_RPE')])
                  ).as_posix()
    else:
        raise RuntimeError(f'Could not find trajectory {trajectory_name} in dir {prefix}')

def get_trajectory_names(prefix):
    val_dirs = list(Path(prefix).glob(f'*val*'))
    paths = [val_dir.as_posix() for val_dir in val_dirs]
    last_dir = max(paths, key=lambda x: int(x[x.find('_val_RPE') - 3: x.find('_val_RPE')]))
    val_trajectory_names = Path(last_dir).joinpath('val').glob('*.csv')
    test_trajectory_names = Path(prefix).joinpath('test/test').glob('*.csv')
    trajectory_names = list(val_trajectory_names) + list(test_trajectory_names)
    trajectory_names = [trajectory_name.stem for trajectory_name in trajectory_names]
    trajectory_names = ['_'.join(tajectory_name.split('_')[1:]) for tajectory_name in trajectory_names]  # Bug handling
    return trajectory_names


def read_csv(path):
    df = pd.read_csv(path)
    df.rename(columns={'path_to_rgb': 'from_path',
                       'path_to_rgb_next': 'to_path'},
              inplace=True)

    stem_fn = lambda x: int(Path(x).stem)
    df['to_index'] = df['to_path'].apply(stem_fn)
    df['from_index'] = df['from_path'].apply(stem_fn)

    df['diff'] = df['to_index'] - df['from_index']

    mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
    std_cols = [c + '_confidence' for c in mean_cols]

    df = pd.concat((df, pd.DataFrame(columns=std_cols)), axis=1)
    df.fillna(1., inplace=True)
    return df


def get_predicted_df(paths):
    df_list = list()
    for stride, path in paths.items():
        df = read_csv(path)

        if stride == 'loops':
            df = df[df['diff'] > 49].reset_index()

        df_list.append(df)

    predicted_df = pd.concat(df_list, ignore_index=True)

    if os.path.basename(os.path.dirname(paths['1'])) == 'val':
        group_id = 1
    elif os.path.basename(os.path.dirname(paths['1'])) == 'test':
        group_id = 0
    else:
        raise RuntimeError(f'Unexpected parent dir of prediction {paths["1"]}. Parent dir must "val" or "test"')

    return predicted_df, group_id


def get_gt_trajectory(dataset_root, trajectory_name):
    gt_df = pd.read_csv(os.path.join(dataset_root, trajectory_name, 'df.csv'))
    gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
    return gt_trajectory


def main(dataset_root, strides, paths, n_jobs, n_iter, output_path=None, **kwargs):
    assert len(strides) == len(paths)
    X = []
    y = []
    groups = []

    trajectory_names = get_trajectory_names(paths[0])

    for trajectory_name in trajectory_names:
        trajectory_paths = {stride: get_path(prefix, trajectory_name) for stride, prefix in zip(strides, paths)}
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
        for i in range(1, len(strides) - 1):
            c = [1] * (len(strides) - i) + [1000000] * i
            coefs.append(c)

    param_distributions = {
        'coef': [dict(zip(strides, c)) for c in coefs],
        'coef_loop': kwargs['coef_loop'] or [0, 1, 100, 300, 500],
        'loop_threshold': kwargs['loop_threshold'] or [50, 100],
        'rotation_scale': kwargs['rotation_scale'] or np.logspace(-10, 0, 11, base=2),
        'max_iterations': kwargs['max_iterations'] or [100, 1000]
    }

    print(param_distributions)
    result = random_search(X, y, groups, param_distributions, n_jobs=n_jobs, n_iter=n_iter, verbose=True)

    if output_path:
        df = pd.DataFrame(result)
        df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--strides', type=str, nargs='+', required=True)
    parser.add_argument('--paths', type=str, nargs='+', required=True)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--coef', type=int, nargs='*', default=None)
    parser.add_argument('--coef_loop', type=int, nargs='*', default=None)
    parser.add_argument('--loop_threshold', type=int, nargs='*', default=None)
    parser.add_argument('--rotation_scale', type=float, nargs='*', default=None)
    parser.add_argument('--max_iterations', type=int, nargs='*', default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
