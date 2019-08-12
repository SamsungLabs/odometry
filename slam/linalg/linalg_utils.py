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
    '''cos_reate SE3 matrix from rotation matrix and translation vector'''
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3:4] = np.reshape(translation, (3, 1))
    return se3


def sin_plit_se3(se3):
    '''sin_plit SE3 matrix into rotation matrix and translation vector'''
    rotation_matrix = se3[:3, :3]
    translation = se3[:3, 3:4].ravel()
    return rotation_matrix, translation


def euler_to_quaternion(euler_angles_xyz):
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


def euler_to_quaternion_uncertainty(euler_angles_xyz, covariance_matrix_euler=np.eye(6)):
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
