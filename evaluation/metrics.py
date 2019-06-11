import numpy as np


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


def get_steps(trajectory_length, rpe_mode):
    if rpe_mode == 'full':
        return list(range(1, trajectory_length))
    elif rpe_mode == 'sqrt':
        return [x ** 2 for x in range(1, int(np.sqrt(trajectory_length)))]
    elif rpe_mode == 'log':
        return [2 ** x for x in range(0, int(np.log2(trajectory_length)))]
    elif rpe_mode == 'kitti':
        return list(range(100, 801, 100))
    raise Exception('Unsupported RPE mode "{}"'.format(rpe_mode))


def RPE(gt_rotation_matrices, predicted_rotation_matrices, gt_trajectory, predicted_trajectory, rpe_mode='full'):
    return RelativePoseError(gt_rotation_matrices, predicted_rotation_matrices, gt_trajectory, predicted_trajectory, rpe_mode, take_root=False)


def RMSE(gt_rotation_matrices, predicted_rotation_matrices, gt_trajectory, predicted_trajectory, rpe_mode='full'):
    return RelativePoseError(gt_rotation_matrices, predicted_rotation_matrices, gt_trajectory, predicted_trajectory, rpe_mode, take_root=True)


def RelativePoseError(gt_rotation_matrices, predicted_rotation_matrices, gt_trajectory, predicted_trajectory, rpe_mode='full', take_root=False):
    '''
    Calculates RPE translation and RPE rotation for 2 global trajectories.
    Args:
        gt_rotation_matrices:  nparray (n,3,3)
        predicted_rotation_matrices: nparray (n,3,3)
        gt_trajectory: GlobalTrajectory
        predicted_trajectory: GlobalTrajectory
        mode: `sqrt`(fastpredicted yet inaccurate),
              `log`,
              `full`(slow but most accurate),
              `kitti`(for distance-based metrics on KITTI)
    Returns:
        RPE translation, RPE rotation
    '''

    rpe_translation = 0
    rpe_rotation = 0

    distance_based = (rpe_mode == 'kitti')
    if distance_based:
        distances = calculate_cumulative_distances_along_trajectory(gt_trajectory)
        if take_root:
            stride = 1
        else:
            stride = 10

    trajectory_length = len(gt_trajectory)
    num_samples = 0
    steps = get_steps(trajectory_length, rpe_mode)
    
    predicted_points = predicted_trajectory.points
    gt_points = gt_trajectory.points
    
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

        if take_root: # RMSE
            t_err = np.mean(l2_norms) ** 0.5
            r_err = np.mean(thetas ** 2) ** 0.5
        else: # RPE
            t_err = np.sum(l2_norms ** 0.5)
            r_err = np.sum(thetas)

        rpe_translation += t_err * scale    
        rpe_rotation += np.rad2deg(r_err) * scale # numpy.arccos returns angle in radians
        num_samples += len(first_indices)

    if take_root:
        rpe_translation /= len(steps)
        rpe_rotation /= len(steps)
        divider = 1.
    else:
        divider = num_samples

    return rpe_translation, rpe_rotation, divider


def ATE(gt_trajectory, predicted_trajectory):
    '''
    Calculates ATE for 2 global trajectories.
    Args:
        gt_trajectory: instance of GlobalTrajectory
        predicted_trajectory: instance of GlobalTrajectory
    Returns:
        ATE
    '''
    predicted_trajectory_aligned = predicted_trajectory.align_with(gt_trajectory)
    X = (predicted_trajectory.points - gt_trajectory.points).reshape(-1, 3, 1)
    l2_norms = np.sum(X ** 2, axis=(1, 2))
    return np.mean(l2_norms) ** 0.5
