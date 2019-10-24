import numpy as np


def convert_rotation_matrix_to_euler_angles(R):
    assert np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6), R

    sin_y = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

    singular = sin_y < 1e-6

    if not singular:
        x = np.arctan2( R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sin_y )
        z = np.arctan2( R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sin_y )
        z = 0

    return np.array([x, y, z])


def shortest_path_with_normalization(angle1, angle2):
    phases = angle1 - angle2
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


def convert_euler_angles_to_rotation_matrix(euler_angles_xyz):
    yaw   = euler_angles_xyz[2]
    pitch = euler_angles_xyz[1]
    roll  = euler_angles_xyz[0]

    cos_r = np.cos(roll)
    sin_r = np.sin(roll)
    cos_p = np.cos(pitch)
    sin_p = np.sin(pitch)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    R_x = np.array([[ 1,      0,      0    ],
                    [ 0,      cos_r, -sin_r],
                    [ 0,      sin_r,  cos_r]])
    R_y = np.array([[ cos_p,  0,      sin_p],
                    [ 0,      1,      0    ],
                    [-sin_p,  0,      cos_p]])
    R_z = np.array([[ cos_y, -sin_y,  0    ],
                    [ sin_y,  cos_y,  0    ],
                    [ 0,      0,      1    ]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def get_relative_se3_matrix(global_se3_matrix, next_global_se3_matrix):
    return np.linalg.inv(global_se3_matrix) @ next_global_se3_matrix


def form_se3(rotation_matrix, translation):
    """create SE3 matrix from rotation matrix and translation vector"""
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3:4] = np.reshape(translation, (3, 1))
    return se3


def split_se3(se3):
    """split SE3 matrix into rotation matrix and translation vector"""
    rotation_matrix = se3[:3, :3]
    translation = se3[:3, 3:4].ravel()
    return rotation_matrix, translation


def euler_to_quaternion(euler_angles_xyz):
    """euler_x,euler_y,euler_z in
       q_w, q_x, q_y, q_z out"""
    yaw   = euler_angles_xyz[2]
    pitch = euler_angles_xyz[1]
    roll  = euler_angles_xyz[0]

    cos_r = np.cos(roll/2)
    sin_r = np.sin(roll/2)
    cos_p = np.cos(pitch/2)
    sin_p = np.sin(pitch/2)
    cos_y = np.cos(yaw/2)
    sin_y = np.sin(yaw/2)

    q_x = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    q_y = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    q_z = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y
    q_w = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y

    return q_w, q_x, q_y, q_z


def quaternion_to_euler(quaternion):
    """q_w, q_x, q_y, q_z in,
       euler_x,euler_y,euler_z out"""
    q_w, q_x, q_y, q_z = quaternion

    t0 = 2.0 * (q_w * q_x + q_y * q_z)
    t1 = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (q_w * q_y - q_z * q_x)
    t2 = np.clip(t2, -1, 1)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (q_w * q_z + q_x * q_y)
    t4 = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


def get_covariance_matrix_from_euler_uncertainty(translation_xyz, euler_angles_xyz):
    """get euler_x,euler_y,euler_z,
        output matrix 6x6 with t_x, t_y, t_z, euler_z (yaw), euler_y (pitch), euler_x (roll)"""
    return np.diag(np.hstack([translation_xyz, euler_angles_xyz[::-1]]))


def convert_euler_uncertainty_to_quaternion_uncertainty(euler_angles_xyz, covariance_matrix_euler=np.eye(6)):
    """get  matrix 6x6 with t_x, t_y, t_z, euler_z, euler_y, euler_x (yaw, pitch, roll),
        output matrix 7x7 with t_x, t_y, t_z, q_w, q_x, q_y, q_z"""
    yaw   = euler_angles_xyz[2]
    pitch = euler_angles_xyz[1]
    roll  = euler_angles_xyz[0]

    cos_r = np.cos(roll/2)
    sin_r = np.sin(roll/2)
    cos_p = np.cos(pitch/2)
    sin_p = np.sin(pitch/2)
    cos_y = np.cos(yaw/2)
    sin_y = np.sin(yaw/2)

    ccc = cos_r * cos_p * cos_y
    ccs = cos_r * cos_p * sin_y
    csc = cos_r * sin_p * cos_y
    css = cos_r * sin_p * sin_y
    scc = sin_r * cos_p * cos_y
    scs = sin_r * cos_p * sin_y
    ssc = sin_r * sin_p * cos_y
    sss = sin_r * sin_p * sin_y

    derivatives = 0.5 * np.array([[ scc-ccs,  scs-csc,  css-scc],
                                  [-csc-scs, -ssc-ccs,  ccc+sss],
                                  [ scc-css,  ccc-sss,  ccs-ssc],
                                  [ ccc+sss, -css-scc, -csc-scs]])

    jacobian = np.eye(7, 6)
    jacobian[3:, 3:] = derivatives
    covariance_matrix_quaternion = jacobian @ covariance_matrix_euler @ jacobian.T

    return covariance_matrix_quaternion


def create_optical_flow_from_rt(depth, intrinsics, rotation_vector, translation_vector):
    w, h = intrinsics.width, intrinsics.height
    R = convert_euler_angles_to_rotation_matrix(rotation_vector)
    t = translation_vector.reshape(3, -1)
    meshgrid = np.meshgrid(np.arange(0., w), np.arange(0., h))
    xyz_points = intrinsics.create_frustrum(*meshgrid, depth)
    xyz_points_after_transform = R.T @ (xyz_points.reshape(3, -1) - t)
    xyz_points_after_transform = xyz_points_after_transform.reshape((3, h, w))
    if (xyz_points_after_transform[2] <= 0).any():
        return None

    xy_pixels_after_transform = intrinsics.backward(xyz_points_after_transform[:2] / xyz_points_after_transform[2])
    flow = (xy_pixels_after_transform - np.c_[meshgrid])
    flow = np.transpose(flow, (1, 2, 0))
    flow[...,0] /= w
    flow[...,1] /= h
    return flow
