import numpy as np
from collections import OrderedDict


def calculate_distances_along_trajectory(points):
    distances_along_trajectory = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances_along_trajectory


def calculate_curve_length(trajectory):
    distances_along_trajectory = calculate_distances_along_trajectory(trajectory)
    return np.sum(distances_along_trajectory)


def find_closest_index(arr, value):
    '''Finds the index of the closest value in an array.'''
    for index in range(len(arr)):
        if arr[index] > value:
            return index
    return -1


def calculate_cumulative_distances_along_trajectory(trajectory):
    '''
    Computes the translational distances along a trajectory.
    Args:
        trajectory: nparray (3,n)
    '''
    distances_along_trajectory = calculate_distances_along_trajectory(trajectory)
    cumulative_distances = np.concatenate([[0], np.cumsum(distances_along_trajectory)])
    return cumulative_distances


def get_pairs_of_indices(trajectory_length, step, stride=None, distances=None):
    if distances is not None:
        first_indices, second_indices = [], []
        first_index = 0
        second_distance = step
        while first_index < trajectory_length:
            second_distance = distances[first_index] + step
            if second_distance > distances[-1]:
                break
            first_indices.append(first_index)
            second_indices.append(find_closest_index(distances, second_distance))
            first_index += stride
    else:
        first_indices = np.arange(trajectory_length - step)
        second_indices = first_indices + step
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
    raise Exception('Unsupported indices option: "{}"'.format(indices))


def calculate_relative_pose_error(gt_trajectory, predicted_trajectory, indices='full', mode='rpe'):
    '''
    Calculates RPE translation and RPE rotation for 2 global trajectories.
    Args:
        gt_trajectory: GlobalTrajectory
        predicted_trajectory: GlobalTrajectory
        indices: `sqrt`(fast yet inaccurate),
                 `log`,
                 `full`(slow but most accurate),
                 `kitti`(for distance-based metrics on KITTI)
        mode: `rpe` of `rmse`
    Returns:
        RPE translation, RPE rotation
    '''

    rpe_translation = 0
    rpe_rotation = 0

    distance_based = (indices == 'kitti')
    if distance_based:
        distances = calculate_cumulative_distances_along_trajectory(gt_trajectory)
        stride = 1 if mode == 'rmse' else 10

    trajectory_length = len(gt_trajectory)
    num_samples = 0
    steps = get_steps(trajectory_length, indices)

    gt_points = gt_trajectory.points
    gt_rotation_matrices = predicted_trajectory.rotation_matrices

    predicted_points = predicted_trajectory.points
    predicted_rotation_matrices = predicted_trajectory.rotation_matrices

    for step in steps:
        if distance_based:
            first_indices, second_indices = get_pairs_of_indices(trajectory_length, step, stride, distances)
            scale = 100. / step
        else:
            first_indices, second_indices = get_pairs_of_indices(trajectory_length, step)
            scale = 1.

        if len(first_indices) == 0:
            continue

        R_first_gt = gt_rotation_matrices[first_indices]
        R_second_gt = gt_rotation_matrices[second_indices]
        R_second_inv_gt = R_second_gt.transpose((0, 2, 1))

        R_first_predicted = predicted_rotation_matrices[first_indices]
        R_second_predicted = predicted_rotation_matrices[second_indices]
        R_first_inv_predicted = R_first_predicted.transpose((0, 2, 1))

        delta_predicted = (predicted_points[second_indices] - predicted_points[first_indices]).reshape((-1, 3, 1))
        delta_gt = (gt_points[second_indices] - gt_points[first_indices]).reshape((-1, 3, 1))

        E_translation = R_second_inv_gt @ (R_first_gt @ R_first_inv_predicted @ delta_predicted - delta_gt)
        l2_norms = np.sum(E_translation ** 2, axis=(1, 2))

        E_rotation = R_second_inv_gt @ R_first_gt @ R_first_inv_predicted @ R_second_predicted
        thetas = np.arccos(np.clip((np.trace(E_rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))

        if mode == 'rmse':
            t_err = np.mean(l2_norms) ** 0.5
            r_err = np.mean(thetas ** 2) ** 0.5
        else:
            t_err = np.sum(l2_norms ** 0.5)
            r_err = np.sum(thetas)

        rpe_translation += t_err * scale    
        rpe_rotation += np.rad2deg(r_err) * scale
        num_samples += len(first_indices)

    if mode == 'rmse':
        rpe_translation /= len(steps)
        rpe_rotation /= len(steps)
        divider = 1.
    else:
        divider = num_samples

    return rpe_translation, rpe_rotation, divider


def calculate_absolute_trajectory_error(gt_trajectory, predicted_trajectory):
    '''
    Calculates ATE for 2 global trajectories.
    Args:
        gt_trajectory: instance of GlobalTrajectory
        predicted_trajectory: instance of GlobalTrajectory
    Returns:
        ATE
    '''
    predicted_trajectory_aligned = predicted_trajectory.align_with(gt_trajectory)
    elementwise_differences = (predicted_trajectory.points - gt_trajectory.points).reshape(-1, 3, 1)
    pointwise_distances = np.sum(elementwise_differences ** 2, axis=(1, 2))
    return np.mean(pointwise_distances) ** 0.5


def calculate_metrics(gt_trajectory, predicted_trajectory, indices='full', prefix=''):
    ate = calculate_absolute_trajectory_error(gt_trajectory, predicted_trajectory)
    rpe_t, rpe_r, divider = calculate_relative_pose_error(gt_trajectory, predicted_trajectory,
                                                          indices=indices, mode='rpe')
    rmse_t, rmse_r, _ = calculate_relative_pose_error(gt_trajectory, predicted_trajectory,
                                                      indices=indices, mode='rmse')
    metrics = {
       'ATE': ate,
       'RMSE_t': rmse_t,
       'RMSE_r': rmse_r,
       'RPE_t': rpe_t,
       'RPE_r': rpe_r,
       'RPE_divider': divider
    }

    return metrics


def normalize_metrics(metrics):
    normalized_metrics = OrderedDict()
    for metric_name in metrics:
        normalized_metrics[metric_name] = metrics[metric_name]

    normalized_metrics['RPE_t'] /= normalized_metrics['RPE_divider']
    normalized_metrics['RPE_r'] /= normalized_metrics['RPE_divider']
    del normalized_metrics['RPE_divider']
    return normalized_metrics


def average_metrics(records):
    if len(records) == 0:
        return []

    averaged_metrics = OrderedDict()
    for metric_name in ('ATE', 'RMSE_t', 'RMSE_r'):
        averaged_metrics[metric_name] = np.mean([record[metric_name] for record in records])

    for metric_name in ('RPE_t', 'RPE_r', 'RPE_divider'):
        total_average_metrics[metric_name] = np.sum([record[metric_name] for record in records])

    return normalize_metrics(total_average_metrics)
