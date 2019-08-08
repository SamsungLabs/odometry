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


def get_relative_se3_matrix(global_se3_matrix, next_global_se3_matrix):
    return np.linalg.inv(global_se3_matrix) @ next_global_se3_matrix


def form_se3(rotation_matrix, translation):
    '''Create SE3 matrix from rotation matrix and translation vector'''
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3:4] = np.reshape(translation, (3, 1))
    return se3


def split_se3(se3):
    '''Split SE3 matrix into rotation matrix and translation vector'''
    rotation_matrix = se3[:3, :3]
    translation = se3[:3, 3:4].ravel()
    return rotation_matrix, translation
