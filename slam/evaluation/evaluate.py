import numpy as np
import pandas as pd
from functools import partial
from collections import OrderedDict
from copy import copy

from slam.utils import Toolbox


def calculate_relative_distances(points):
    """
    Calculates distances between neighboring points along the trajectory.

    Args:
        points:               nparray or tensor, n x 3

    Returns:
        relative_distances:   same dtype as points, n x 1
    """
    relative_distances = ((points[1:] - points[:-1]) ** 2).sum(1) ** 0.5
    return relative_distances


def calculate_curve_length(points):
    """
    Calculates physical length of trajectory.

    Args:
        points:         nparray or tensor, n x 3

    Returns:
        curve length:   float or single tensor
    """
    relative_distances = calculate_relative_distances(points)
    return relative_distances.sum()


def find_closest_index(arr, value):
    for index in range(len(arr)):
        if arr[index] > value:
            return index
    return -1


def calculate_cumulative_distances(points):
    """
    Calculates distance from starting point to every point of trajectory.

    Args:
        points:                 nparray or tensor, n x 3

    Returns:
        cumulative_distances:   same dtype as points, n x 1
    """
    relative_distances = calculate_relative_distances(points)
    cumulative_distances = relative_distances.cumsum(0)
    return cumulative_distances


def get_pairs_of_indices(trajectory_length, step, stride=None, distances=None):
    if distances is None:
        first_indices = np.arange(trajectory_length - step)
        second_indices = first_indices + step
    else:
        first_indices, second_indices = [], []
        first_index = 0
        second_distance = step
        while first_index < trajectory_length:
            second_distance = (distances[first_index - 1] if first_index else 0) + step
            if second_distance > distances[-1]:
                break
            first_indices.append(first_index)
            second_indices.append(find_closest_index(distances, second_distance) + 1)
            first_index += stride

    return first_indices, second_indices


def get_steps(trajectory_length, indices):
    if indices == 'full':
        return list(range(1, trajectory_length))
    elif indices == 'sqrt':
        return [x ** 2 for x in range(1, int(np.sqrt(trajectory_length)))]
    elif indices == 'log':
        return [2 ** x for x in range(0, int(np.log2(trajectory_length)))]
    elif indices == 'kitti':
        return list(range(100, 801, 100))
    else:
        raise ValueError(f'Unknown indices option: "{indices}"')


def calculate_relative_pose_error(gt_trajectory, predicted_trajectory,
                                  rpe_indices='full', rpe_mode='rpe',
                                  backend='numpy', cuda=False):
    """
    Calculates RPE translation and RPE rotation for 2 global trajectories.

    Args:
        gt_trajectory:        GlobalTrajectory
        predicted_trajectory: GlobalTrajectory
        rpe_indices:          'sqrt' (fast yet inaccurate),
                              'log',
                              'full' (slow but most accurate),
                              'kitti' (for distance-based metrics on KITTI)
        rpe_mode:             'rpe' of 'rmse'
        backend:              'numpy' or 'torch'
        cuda:                 whether to use GPU (only for backend='torch')

    Returns:
        RPE translation
        RPE rotation
        RPE divider = 1 (rpe_mode='rmse') or number of index pairs (rpe_mode='rpe')
    """

    trajectory_length = len(gt_trajectory)
    num_samples = 0
    steps = get_steps(trajectory_length, rpe_indices)

    tb = Toolbox(backend=backend, cuda=cuda)

    gt_points = tb.from_numpy(gt_trajectory.points[..., None])
    R_gt = tb.from_numpy(gt_trajectory.rotation_matrices)
    R_gt_inv = tb.btranspose(R_gt)

    predicted_points = tb.from_numpy(predicted_trajectory.points[..., None])
    R_predicted = tb.from_numpy(predicted_trajectory.rotation_matrices)
    R_predicted_inv = tb.btranspose(R_predicted)

    R = tb.bmm(R_gt, R_predicted_inv)

    if rpe_indices == 'kitti':
        get_indices_fn = partial(get_pairs_of_indices,
                                 stride=(1 if rpe_mode == 'rmse' else 10),
                                 distances=calculate_cumulative_distances(gt_points))
        get_scale_fn = lambda step: 100. / step
    else:
        get_indices_fn = get_pairs_of_indices
        get_scale_fn = lambda step: 1.

    rpe_translation = 0
    rpe_rotation = 0

    for step in steps:
        first_indices, second_indices = get_indices_fn(trajectory_length, step)

        if len(first_indices) == 0:
            continue

        delta_predicted = predicted_points[second_indices] - predicted_points[first_indices]
        delta_gt = gt_points[second_indices] - gt_points[first_indices]

        E_translation = tb.bmm(R[first_indices], delta_predicted) - delta_gt
        l2_norms = (E_translation ** 2).sum((1, 2))

        E_rotation = tb.bmm(tb.bmm(R_gt_inv[second_indices], R[first_indices]), R_predicted[second_indices])
        radians = tb.acos(tb.clip((tb.btrace(E_rotation) - 1) / 2, -1, 1))
        thetas = radians * 180 / np.pi

        if rpe_mode == 'rmse':
            t_err = l2_norms.mean() ** 0.5
            r_err = (thetas ** 2).mean() ** 0.5
        else:
            t_err = (l2_norms ** 0.5).sum()
            r_err = thetas.sum()

        scale = get_scale_fn(step)
        rpe_translation += t_err * scale
        rpe_rotation += r_err * scale

        num_samples += len(first_indices)

    if rpe_mode == 'rmse':
        rpe_translation /= len(steps)
        rpe_rotation /= len(steps)
        divider = 1.
    else:
        divider = num_samples

    rpe_translation = tb.item(rpe_translation)
    rpe_rotation = tb.item(rpe_rotation)

    return rpe_translation, rpe_rotation, divider


def calculate_absolute_trajectory_error(gt_trajectory, predicted_trajectory):
    """
    Calculates ATE for 2 global trajectories.

    Args:
        gt_trajectory:        GlobalTrajectory
        predicted_trajectory: GlobalTrajectory

    Returns:
        ATE
    """
    predicted_trajectory_aligned = predicted_trajectory.align_with(gt_trajectory)
    elementwise_differences = (predicted_trajectory_aligned.points - gt_trajectory.points).reshape(-1, 3, 1)
    pointwise_distances = np.sum(elementwise_differences ** 2, axis=(1, 2))
    return np.mean(pointwise_distances) ** 0.5


def calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices='full',
                      backend='numpy', cuda=False):
    ate = calculate_absolute_trajectory_error(gt_trajectory, predicted_trajectory)
    rpe_t, rpe_r, divider = calculate_relative_pose_error(gt_trajectory, predicted_trajectory,
                                                          rpe_indices=rpe_indices, rpe_mode='rpe',
                                                          backend=backend, cuda=cuda)
    rmse_t, rmse_r, _ = calculate_relative_pose_error(gt_trajectory, predicted_trajectory,
                                                      rpe_indices=rpe_indices, rpe_mode='rmse',
                                                      backend=backend, cuda=cuda)
    metrics = {
       'ATE': ate,
       'RMSE_t': rmse_t,
       'RMSE_r': rmse_r,
       'RPE_t': rpe_t,
       'RPE_r': rpe_r,
       'RPE_divider': divider
    }

    return metrics


def calculate_loops_metrics(gt_df, predicted_df, loop_threshold):
    df_merged = pd.merge(gt_df, predicted_df, on=('to_index', 'from_index'))
    index_difference = df_merged.to_index - df_merged.from_index
    loops_df = df_merged[index_difference >= loop_threshold].reset_index(drop=True)
    if not len(loops_df):
        return {'loops_MAE_t': 0, 'loops_MAE_r': 0}
    translation_dofs = ['t_x', 't_y', 't_z']
    rotation_dofs = ['euler_x', 'euler_y', 'euler_z']

    loop_metrics_intermediate = {}
    for dof in translation_dofs + rotation_dofs:
        loop_metrics_intermediate[f'loops_MAE_{dof}'] = \
            np.abs((loops_df[f'{dof}_x'] - loops_df[f'{dof}_y']).values).mean()
    loop_metrics = {}
    loop_metrics[f'loops_MAE_t'] = np.mean([loop_metrics_intermediate[f'loops_MAE_{dof}'] for dof in translation_dofs])
    loop_metrics[f'loops_MAE_r'] = np.mean([loop_metrics_intermediate[f'loops_MAE_{dof}'] for dof in rotation_dofs])
    return loop_metrics


def normalize_metrics(metrics):
    normalized_metrics = copy(metrics)
    normalized_metrics['RPE_t'] /= normalized_metrics['RPE_divider']
    normalized_metrics['RPE_r'] /= normalized_metrics['RPE_divider']
    del normalized_metrics['RPE_divider']
    return normalized_metrics


def average_metrics(records):
    if len(records) == 0:
        return dict()

    averaged_metrics = OrderedDict()

    for metric_name in ('ATE', 'RMSE_t', 'RMSE_r', 'loops_MAE_t', 'loops_MAE_r'):
        if metric_name in records[0]:
            averaged_metrics[metric_name] = np.mean([record[metric_name] for record in records])

    for metric_name in ('RPE_t', 'RPE_r', 'RPE_divider'):
        if metric_name in records[0]:
            averaged_metrics[metric_name] = np.sum([record[metric_name] for record in records])

    return normalize_metrics(averaged_metrics)
