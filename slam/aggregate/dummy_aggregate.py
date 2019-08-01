import numpy as np
from pyquaternion import Quaternion
from collections import OrderedDict

from slam.linalg import (convert_euler_angles_to_rotation_matrix, 
                         form_se3, 
                         split_se3,
                         GlobalTrajectory)

class AggregateTrajectory:
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.global_se3_matrices = OrderedDict({0: np.eye(4, 4)})

    def get(self, index):
        return self.global_se3_matrices.get(index, None)
    
    def save(self, index, se3):
        self.global_se3_matrices[index] = se3
    
    @staticmethod
    def average(first_se3, second_se3):
        if second_se3 is None:
            return first_se3
        
        first_rotation_matrix, first_translation = split_se3(first_se3)
        second_rotation_matrix, second_translation = split_se3(second_se3)

        first_quaternion = Quaternion(matrix=first_rotation_matrix)
        second_quaternion = Quaternion(matrix=second_rotation_matrix)
        average_quaternion = Quaternion.slerp(first_quaternion, second_quaternion, amount=0.5)

        average_rotation_matrix = average_quaternion.rotation_matrix
        average_translation = np.array((first_translation + second_translation) / 2)

        average_se3 = form_se3(average_rotation_matrix, average_translation)
        return average_se3
        
    def append(self, df):
        for index, row in df.iterrows():
            from_index = row['from']
            to_index = row['to']
            euler_angles = row[['euler_x', 'euler_y', 'euler_z']].values
            translation = row[['t_x', 't_y', 't_z']].values
            
            rotation_matrix = convert_euler_angles_to_rotation_matrix(euler_angles)
            relative_se3 = form_se3(rotation_matrix, translation)
            
            from_se3 = self.get(from_index)
            to_se3 = from_se3 @ relative_se3
            to_se3_from_history = self.get(to_index)
            
            average_to_se3 = self.average(to_se3, to_se3_from_history)
            self.save(to_index, average_to_se3)

    def get_trajectory(self):
        return GlobalTrajectory.from_transformation_matrices(self.global_se3_matrices.values())
