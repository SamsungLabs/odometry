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
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),       0],
                    [math.sin(theta[2]),    math.cos(theta[2]),        0],
                    [0,                     0,                         1]
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


def euler_to_quaternion(euler_angels):
    (yaw, pitch, roll) = (euler_angels[0], euler_angels[1], euler_angels[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(quaternion):
    (x, y, z, w) = (quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    t0 =  2.0 * (w * x + y * z)
    t1 =  1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 =  2.0 * (w * y - z * x)
    t2 =  1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 =  2.0 * (w * z + x * y)
    t4 =  1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def euler_to_quaternion_uncertainty(euler_angels, covariance_matrix_euler=np.eye(6,6)):
    
    yaw, pitch, roll = euler_angels[0], euler_angels[1], euler_angels[2]
    
    ccc = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2)
    ccs = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    csc = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    css = np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    scc = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2)
    ssc = np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    scs = np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    sss = np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    devers_q_on_eulers = [[ (scc-ccs)/2  ,  (scs-csc)/2  ,  (css-scc)/2],
                          [-(csc+scs)/2  , -(ssc+ccs)/2  ,  (ccc+sss)/2],
                          [ (scc-css)/2  ,  (ccc-sss)/2  ,  (ccs-ssc)/2],
                          [ (ccc+sss)/2  , -(css+scc)/2  , -(csc+scs)/2]]
    jacobian = np.vstack((
                         np.hstack(( np.eye(3,3)     , np.zeros((3,3))    )),
                         np.hstack(( np.zeros((4,3)) , devers_q_on_eulers ))
                        ))
    covariance_matrix_quaternion = jacobian @ covariance_matrix_euler @ jacobian.T
    
    return covariance_matrix_quaternion
