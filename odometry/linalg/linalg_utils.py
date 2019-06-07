import numpy as np
import math


def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def convert_rotation_matrix_to_euler_angles(R):
    assert(is_rotation_matrix(R)), '{}'.format(R)

    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def convert_euler_angles_to_rotation_matrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def convert_global_se3_matrices_to_relative(se3_matrices, stride=1):
    for index in range(len(se3_matrices) - stride):
        se3_matrices[index] = np.linalg.inv(se3_matrices[index]) @ se3_matrices[index + stride]

    initial_matrix = np.eye(4)
    relative_se3_matrices = np.insert(se3_matrices, 0, initial_matrix, axis=0)
    return relative_se3_matrices[:len(se3_matrices)]


def form_se3(rotation_matrix, translation):
    '''Create SE3 matrix from rotation matrix and translation vector'''
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3:4] = np.reshape(translation, (3, 1))
    return se3


def convert_relative_se3_matrices_to_euler(relative_se3_matrices):
    relative_rotations_euler = []
    relative_translations_euler = []
    for relative_se3_matrix in relative_se3_matrices:
        rotation_matrix = relative_se3_matrix[:3, :3]
        translation = relative_se3_matrix[:3, 3]
        relative_rotations_euler.append(convert_rotation_matrix_to_euler_angles(rotation_matrix))
        relative_translations_euler.append(translation)
    return np.array(relative_rotations_euler), np.array(relative_translations_euler)
