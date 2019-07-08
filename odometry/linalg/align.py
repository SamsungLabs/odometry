import numpy as np


def align(trajectory_points, reference_trajectory_points, by='mean'):
    '''
    Align two trajectories using the method of Horn (closed-form).

    Args:
        trajectory_points:           nx3
        reference_trajectory_points: nx3

    Returns:
        rotation_matrix: 3x3
        translation:     3x1
        scale:           float
    '''
    np.set_printoptions(precision=3, suppress=True)

    n = len(trajectory_points)
    if n < len(reference_trajectory_points):
        by = 'start'
        reference_trajectory_points = reference_trajectory_points[:n]

    if by == 'mean':
        align_point = trajectory_points.mean(0)
        reference_align_point = reference_trajectory_points.mean(0)

    elif by == 'start':
        align_point = trajectory_points[0]
        reference_align_point = reference_trajectory_points[0]

    align_point = align_point[None]
    reference_align_point = reference_align_point[None]
    trajectory_points_shifted = trajectory_points - align_point
    reference_trajectory_points_shifted = reference_trajectory_points - reference_align_point

    W = np.zeros((3, 3))
    for index in range(n):
        W += np.outer(trajectory_points_shifted[index], reference_trajectory_points_shifted[index])
    
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    
    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    rotation_matrix = U @ S @ Vh
    trajectory_points_rotated = trajectory_points_shifted @ rotation_matrix.T
    
    dots = 0.0
    norms = 0.0
    for index in range(n):
        dots += np.dot(reference_trajectory_points_shifted[index], trajectory_points_rotated[index])
        norms += np.inner(trajectory_points_shifted[index], trajectory_points_shifted[index])

    scale = float(dots / norms)
    translation = (reference_align_point - scale * (align_point @ rotation_matrix.T))[0]
    return rotation_matrix, translation, scale
