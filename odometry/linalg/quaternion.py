import numpy as np
from collections import namedtuple
from pyquaternion import Quaternion

from odometry.linalg.linalg_utils import (convert_euler_angles_to_rotation_matrix,
                                          convert_rotation_matrix_to_euler_angles)


class QuaternionWithTranslation:
    def __init__(self, q=Quaternion(), t=[0, 0, 0]):
        self.quaternion = q
        self.translation = np.array(t.copy())

    def __str__(self):
        return 'quaternion: {}, translation: {}'.format(self.quaternion, self.translation)

    def copy(self):
        return QuaternionWithTranslation(self.quaternion.copy(), self.translation.copy())

    @property
    def rotation_matrix(self):
        return self.quaternion.rotation_matrix.copy()

    def to_quaternion(self):
        q = self.quaternion.elements
        t = self.translation
        return {'q_w': q[0], 'q_x': q[1], 'q_y': q[2], 'q_z': q[3], 't_x': t[0], 't_y': t[1], 't_z': t[2]}

    @classmethod
    def from_quaternion(cls, q):
        translation = [q['t_x'], q['t_y'], q['t_z']]
        quaternion = Quaternion([q['q_w'], q['q_x'], q['q_y'], q['q_z']])
        return cls(q=quaternion, t=translation)

    def to_transformation_matrix(self):
        T = self.quaternion.transformation_matrix
        t = self.translation.copy()
        T[:3, -1] = t
        return T

    @classmethod
    def from_transformation_matrix(cls, transformation_matrix):
        quaternion = Quaternion(matrix=transformation_matrix).normalised
        translation = transformation_matrix[:3, -1]
        return cls(q=quaternion, t=translation)

    def to_euler_angles(self):
        euler_angles = convert_rotation_matrix_to_euler_angles(self.rotation_matrix)
        t = self.translation
        return {'euler_x': euler_angles[0], 'euler_y': euler_angles[1], 'euler_z': euler_angles[2],
                't_x': t[0], 't_y': t[1], 't_z': t[2]}

    @classmethod
    def from_euler_angles(cls, euler_angles_with_translation):
        euler_angles = [euler_angles_with_translation['euler_x'], euler_angles_with_translation['euler_y'], euler_angles_with_translation['euler_z']]
        rotation_matrix = convert_euler_angles_to_rotation_matrix(euler_angles)
        translation = [euler_angles_with_translation['t_x'], euler_angles_with_translation['t_y'], euler_angles_with_translation['t_z']]
        return cls.from_rotation_matrix((rotation_matrix, translation))

    def to_rotation_matrix(self):
        R = self.rotation_matrix
        t = self.translation.copy()
        return R, t

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix_with_translation):
        rotation_matrix, translation = rotation_matrix_with_translation
        quaternion = Quaternion(matrix=rotation_matrix).normalised
        return cls(q=quaternion, t=translation)

    def to_semi_global(self, origin):
        transformation_current = self.to_transformation_matrix()
        transformation_origin = origin.to_transformation_matrix()
        transformation_relative = np.linalg.inv(transformation_origin) @ transformation_current
        return QuaternionWithTranslation.from_transformation_matrix(transformation_relative)
