import numpy as np
import argparse

import __init_path__
import env


from slam.aggregation import random_search
from scripts.aggregation.base_search import (get_path,
                                             get_gt_trajectory,
                                             get_predicted_df,
                                             get_trajectory_names,
                                             get_group_id)

from scripts.aggregation import configs


def get_coefs(vals, current_level, max_depth):
    if current_level == max_depth:
        coefs = list()
        for v in vals:
            coefs.append([v])
        return coefs
    else:
        coefs = get_coefs(vals, current_level + 1, max_depth)
        new_coefs = list()
        for v in vals:
            for c in coefs:
                new_coefs.append([v] + c)
        return new_coefs


def main(dataset_root,
         config_type,
         n_jobs,
         n_iter,
         output_path=None,
         **kwargs):
    config = getattr(configs, config_type)
    trajectory_names = get_trajectory_names(config['1'][0])
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
            trajectory_paths[k] = [get_path(prefix, trajectory_name) for prefix in config[k]]

        predicted_df = get_predicted_df(trajectory_paths)
        group_id = get_group_id(trajectory_paths)
        gt_trajectory = get_gt_trajectory(dataset_root, trajectory_name)

        X.append(predicted_df)
        y.append(gt_trajectory)
        groups.append(group_id)

    coef_values = [1, 2, 4] + list(np.logspace(1, 6, num=6)) + [1e12]
    if kwargs['coef']:
        coefs = [kwargs['coef']]
    else:
        coefs = get_coefs(coef_values, 1, len(config.keys()) - 1)

    param_distributions = {
        'coef': [dict(zip(strides, c)) for c in coefs],
        'coef_loop': kwargs['coef_loop'] or coef_values,
        'loop_threshold': kwargs['loop_threshold'] or [50, 100],
        'rotation_scale': kwargs['rotation_scale'] or np.logspace(-10, 0, 11, base=2),
        'max_iterations': kwargs['max_iterations'] or [1000]
    }

    print(param_distributions)

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
    parser.add_argument('--config_type', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=3)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--coef', type=int, nargs='*', default=None)
    parser.add_argument('--coef_loop', type=int, nargs='*', default=None)
    parser.add_argument('--loop_threshold', type=int, nargs='*', default=None)
    parser.add_argument('--rotation_scale', type=float, nargs='*', default=None)
    parser.add_argument('--max_iterations', type=int, nargs='*', default=None)
    args = parser.parse_args()
    main(**vars(args))
