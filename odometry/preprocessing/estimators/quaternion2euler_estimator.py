import os
import shutil
import copy
import numpy as np

from pyquaternion import Quaternion

from odometry.preprocessing.estimators.base_estimator import BaseEstimator
from odometry.linalg import convert_rotation_matrix_to_euler_angles


class Quaternion2EulerEstimator(BaseEstimator):

    def __init__(self, *args, **kwargs):
        super(Quaternion2EulerEstimator, self).__init__(*args, **kwargs)
        self.name = 'Quaternion2Euler'

    def run(self, row, dataset_root=None):
        quaternion = Quaternion(row[self.input_col].values)
        euler_angles = convert_rotation_matrix_to_euler_angles(quaternion.rotation_matrix)
        row = self._drop_input(row)
        return self._add_output(row, euler_angles)
