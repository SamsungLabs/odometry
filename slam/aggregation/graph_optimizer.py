import g2o
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from slam.aggregation.base_aggregator import BaseAggregator
from slam.linalg import (GlobalTrajectory,
                         QuaternionWithTranslation,
                         convert_euler_angles_to_rotation_matrix)

from slam.utils import mlflow_logging


@mlflow_logging(prefix='aggregator', name='GraphOptimizer')
class GraphOptimizer(BaseAggregator):
    def __init__(self, max_iterations=100, verbose=False, online=False):
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        self.optimizer = g2o.SparseOptimizer()
        self.optimizer.set_verbose(True)
        self.optimizer.set_algorithm(solver)

        self.current_pose = None
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.online = online
        self.clear()

    def clear(self):
        self.optimizer.clear()
        self.optimizer.set_verbose(self.verbose)
        vertex = self.create_vertex(np.eye(3), np.zeros(3), index=0)
        self.optimizer.add_vertex(vertex)
        self.current_pose = np.identity(6)

    @property
    def length(self):
        return len(self.optimizer.vertices())

    def append(self, df):

        for _, row in df.iterrows():

            is_adjustment_measurements = (row.to_index - row.from_index) == 1

            if is_adjustment_measurements:
                current_pose = self.update_current_pose(row)
                index = len(self.optimizer.vertices())
                vertex = self.create_vertex(current_pose.quaternion.rotation_matrix, current_pose.translation, index)
                self.optimizer.add_vertex(vertex)

            edge = self.create_edge(row)
            self.optimizer.add_edge(edge)

        if self.online:
            self.optimize()

    def get_previous_pose(self) -> np.ndarray:
        previous_vertex = self.optimizer.vertex(self.length - 1)
        previous_estimate = previous_vertex.estimate()
        position = previous_estimate.position()
        quaternion_g2o = previous_estimate.Quaternion()
        quaternion = Quaternion(matrix=quaternion_g2o.rotation_matrix())
        transformation_matrix = QuaternionWithTranslation(quaternion, position).to_transformation_matrix()
        return transformation_matrix

    def update_current_pose(self, row):
        previous_transformation_matrix = self.get_previous_pose()

        relative_pose = QuaternionWithTranslation.from_euler_angles(row.to_dict())
        relative_transformation_matrix = relative_pose.to_transformation_matrix()
        current_transformation_matrix = previous_transformation_matrix @ relative_transformation_matrix
        current_pose = QuaternionWithTranslation().from_transformation_matrix(current_transformation_matrix)
        return current_pose

    def create_pose(self, orientation: np.ndarray, translation: np.ndarray) -> g2o.Isometry3d:
        pose = g2o.Isometry3d()
        pose.set_translation(translation)
        q = g2o.Quaternion(orientation)
        pose.set_rotation(q)
        return pose

    def create_vertex(self, orientation: np.ndarray, translation: np.ndarray, index: int) -> g2o.VertexSE3:
        pose = self.create_pose(orientation, translation)
        vertex = g2o.VertexSE3()
        vertex.set_estimate(pose)
        vertex.set_id(index)
        vertex.set_fixed(index == 0)
        return vertex

    def create_edge(self, row: pd.Series) -> g2o.EdgeSE3:
        euler_angles = row[['euler_x', 'euler_y', 'euler_z']].values
        translation = row[['t_x', 't_y', 't_z']].values
        rotation_matrix = convert_euler_angles_to_rotation_matrix(euler_angles)

        measurement = self.create_pose(rotation_matrix, translation)

        edge = g2o.EdgeSE3()
        edge.set_measurement(measurement)

        edge.set_information(np.eye(6))
        edge.set_vertex(0, self.optimizer.vertex(int(row['from_index'])))
        edge.set_vertex(1, self.optimizer.vertex(int(row['to_index'])))
        return edge

    def optimize(self):
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(self.max_iterations)

    def get_trajectory(self):

        if not self.online:
            self.optimize()

        optimized_trajectory = GlobalTrajectory()
        for index in range(self.length):
            estimate = self.optimizer.vertex(index).estimate()
            qt = QuaternionWithTranslation.from_rotation_matrix((estimate.R, estimate.t))
            optimized_trajectory.append(qt)

        return optimized_trajectory
