import math
import numpy as np
from collections import namedtuple
from pyquaternion import Quaternion

from odometry.linalg.linalg_utils import (convert_euler_angles_to_rotation_matrix,
                                          convert_rotation_matrix_to_euler_angles)

class QuaternionWithTranslation:
    def __init__(self, q = Quaternion(), t = [0, 0, 0]):
        self.quaternion = q
        self.translation = t.copy()

    def __str__(self):
        return 'quaternion: {}, translation: {}'.format(self.quaternion, self.translation)

    def to_quaternion(self):
        a = self.quaternion.elements
        t = self.translation
        return {'qw': a[0], 'qx': a[1], 'qy': a[2], 'qz': a[3], 'tx': t[0], 'ty': t[1], 'tz': t[2]}

    def from_quaternion(self, q):
        self.translation = [q['tx'], q['ty'], q['tz']]
        self.quaternion = Quaternion([q['qw'], q['qx'], q['qy'], q['qz']])

    def to_transformation_matrix(self):
        T = self.quaternion.transformation_matrix
        t = self.translation
        T[0, -1] = t[0]
        T[1, -1] = t[1]
        T[2, -1] = t[2]
        return {'row0': T[0, :].tolist(), 'row1': T[1, :].tolist(), 'row2': T[2, :].tolist(), 'row3': T[3, :].tolist()}

    def from_transformation_matrix(self, transformation):
        T = matrix = np.zeros(np.array([4,4]))
        T[0, :] = np.array(transformation['row0'])
        T[1, :] = np.array(transformation['row1'])
        T[2, :] = np.array(transformation['row2'])
        T[3, :] = np.array(transformation['row3'])
        self.quaternion = Quaternion(matrix=T)
        self.translation = [T[0, -1], T[1, -1], T[2, -1]]

    def to_euler_angles(self):
        angles = convert_rotation_matrix_to_euler_angles(self.quaternion.rotation_matrix)
        t = self.translation
        return {'euler_x': angles[0], 'euler_y': angles[1], 'euler_z': angles[2], 'x': t[0], 'y': t[1], 'z': t[2]}

    def from_euler_angles(self, euler_angles_and_translation):
        theta = [euler_angles_and_translation['euler_x'], euler_angles_and_translation['euler_y'], euler_angles_and_translation['euler_z']]
        rotation_matrix = convert_euler_angles_to_rotation_matrix(theta)
        self.quaternion = Quaternion(matrix=rotation_matrix)
        self.translation = [euler_angles_and_translation['x'], euler_angles_and_translation['y'], euler_angles_and_translation['z']]

    def to_axis_angle(self):
        axis = self.quaternion.axis
        theta = self.quaternion.radians
        t = self.translation
        return {'axis_x': axis[0], 'axis_y': axis[1], 'axis_z': axis[2], 'theta': theta, 'tx': t[0], 'ty': t[1], 'tz': t[2]}

    def from_axis_angle(self, axis_angle):
        self.translation = [axis_angle['tx'], axis_angle['ty'], axis_angle['tz']]
        self.quaternion = Quaternion(axis=[axis_angle['axis_x'], axis_angle['axis_y'], axis_angle['axis_z']], angle=axis_angle['theta'])

    def to_semi_global(self, origin):
        q_origin = origin.quaternion
        t_origin = origin.translation.copy()
        q_current = self.quaternion
        t_current = self.translation.copy()

        transformation_current = q_current.transformation_matrix
        transformation_current[0, -1] = t_current[0]
        transformation_current[1, -1] = t_current[1]
        transformation_current[2, -1] = t_current[2]

        transformation_origin = q_origin.transformation_matrix
        transformation_origin[0, -1] = t_origin[0]
        transformation_origin[1, -1] = t_origin[1]
        transformation_origin[2, -1] = t_origin[2]

        transformation_relative = np.linalg.inv(transformation_origin)@transformation_current

        quaternion = Quaternion(matrix=transformation_current)
        translation = transformation_relative[0:3, -1]
        return QuaternionWithTranslation(quaternion, translation)
