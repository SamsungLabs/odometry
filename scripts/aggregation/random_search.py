import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import argparse


import __init_path__
import env

from slam.linalg import RelativeTrajectory
from scripts.aggregation import random_search


def get_path(prefix, trajectory_name):
    paths = glob(f'{prefix}/*/val/*{trajectory_name}.csv')
    if len(paths) == 1:
        return paths[0]
    elif len(paths) > 1:
        return max(paths, key=lambda x: int(x[x.find('_val_RPE') - 3: x.find('_val_RPE')]))
    else:
        return glob(f'{prefix}/test/test/*{trajectory_name}.csv')[0]
    

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
    return predicted_df

def get_gt_trajectory(trajectory_name):
    gt_df = pd.read_csv(os.path.join(env.KITTI_MIXED_PATH, '1', trajectory_name, 'df.csv'))
    gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
    return gt_trajectory

def main(trajectory_names, strides, pathes, n_jobs, n_iter):
    
    X = []
    y = []
    for trajectory_name in trajectory_names:
        paths = {stride: get_path(prefix, trajectory_name) for stride, prefix in zip(strides, pathes)}

        predicted_df = get_predicted_df(paths)
        gt_trajectory = get_gt_trajectory(trajectory_name)

        X.append(predicted_df)
        y.append(gt_trajectory)

    strides = [int(stride) for stride in strides if stride != 'loops']

    coefs = [
        [1, 1, 1],
        [1, 2, 4],
        [1, 6, 12],
        [1, 1, 1000000],
        [1, 1000000, 1000000]
    ]

    param_distributions = {
        'coef': [dict(zip(strides, c)) for c in coefs],
        'coef_loop': [0, 1, 100, 300, 500], 
        'loop_threshold': [50, 100], 
        'rotation_scale': np.logspace(-10, 0, 11, base=2),
        'max_iterations': [100, 1000]
    }
    random_search(X, y, param_distributions, n_jobs=n_jobs, n_iter=n_iter, verbose=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_names', type=str, nargs='+', required=True)
    parser.add_argument('--strides', type=str, nargs='+', required=True)
    parser.add_argument('--pathes', type=str, nargs='+', required=True)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))