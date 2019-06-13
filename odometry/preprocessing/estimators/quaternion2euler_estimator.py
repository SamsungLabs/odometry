import os
import shutil
import copy
import numpy as np

from pyquaternion import Quaternion

from odometry.preprocessing.estimators.base_estimator import BaseEstimator
from odometry.linalg.linalg_utils import convert_rotation_matrix_to_euler_angles


class Quaternion2EulerEstimator(BaseEstimator):

    def run(self, row, dataset_root=None):
        quaternion = Quaternion(row[self.input_col].values)
        euler_angles = convert_rotation_matrix_to_euler_angles(quaternion.rotation_matrix)
        for key in self.input_col:
            del row[key]
        return self._extend(row, euler_angles)
