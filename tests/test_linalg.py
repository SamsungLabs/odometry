from slam import linalg
import unittest
import numpy as np


class TestCovarianceConverter(unittest.TestCase):

    def generate_data(self, non_zero_angle_ind):
        euler_angles = np.zeros(3)
        euler_angles[non_zero_angle_ind] = 1
        covariance_matrix = np.diag(np.hstack([np.zeros(3), euler_angles]))
        covariance_matrix = linalg.convert_euler_uncertainty_to_quaternion_uncertainty(euler_angles, covariance_matrix)

        answer = np.zeros((7, 7))

        quaternion = linalg.euler_to_quaternion(euler_angles)

        non_zero_quaternion_ind = np.squeeze((np.argwhere(np.array(quaternion) > 0) + 3))
        for i in non_zero_quaternion_ind:
            for j in non_zero_quaternion_ind:
                answer[i, j] = 1

        covariance_matrix[np.abs(covariance_matrix) > 1e-3] = 1

        return covariance_matrix, answer

    def test_1(self):
        covariance_matrix, answer = self.generate_data(0)
        self.assertTrue(np.allclose(covariance_matrix, answer))

    def test_2(self):
        covariance_matrix, answer = self.generate_data(1)
        self.assertTrue(np.allclose(covariance_matrix, answer))

    def test_3(self):
        covariance_matrix, answer = self.generate_data(2)
        self.assertTrue(np.allclose(covariance_matrix, answer))
