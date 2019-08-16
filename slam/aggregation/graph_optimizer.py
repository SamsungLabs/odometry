import g2o
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from slam.aggregation.base_aggregator import BaseAggregator
from slam.linalg import (GlobalTrajectory,
                         QuaternionWithTranslation,
                         RelativeTrajectory,
                         convert_euler_angles_to_rotation_matrix)


class GraphOptimizer(BaseAggregator):
    def __init__(self):
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        self.optimizer = g2o.SparseOptimizer()
        self.optimizer.set_verbose(True)
        self.optimizer.set_algorithm(solver)

        self.measurements = pd.DataFrame()
        self.max_iterations = 100

    def append(self, df):
        self.measurements = self.measurements.append(df).reset_index(drop=True)

    @staticmethod
    def create_pose(orientation: np.ndarray, translation: np.ndarray) -> g2o.Isometry3d:
        pose = g2o.Isometry3d()
        pose.set_translation(translation)
        q = g2o.Quaternion(orientation)
        pose.set_rotation(q)
        return pose

    @staticmethod
    def create_vertex(orientation: np.ndarray, translation: np.ndarray, index: int) -> g2o.VertexSE3:
        pose = GraphOptimizer.create_pose(orientation, translation)
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

    def get_trajectory(self):
        is_adjustment_measurements = (self.measurements.to_index - self.measurements.from_index) == 1
        adjustment_measurements = self.measurements[is_adjustment_measurements].reset_index(drop=True)
        trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()

        for index, position in enumerate(trajectory.positions):
            vertex = self.create_vertex(position.quaternion.elements, position.translation, index)
            self.optimizer.add_vertex(vertex)

        for index, row in self.measurements.iterrows():
            edge = self.create_edge(row)
            self.optimizer.add_edge(edge)

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(self.max_iterations)

        optimized_trajectory = GlobalTrajectory()
        for index in range(len(self.optimizer.vertices())):
            estimate = self.optimizer.vertex(index).estimate()
            qt = QuaternionWithTranslation.from_rotation_matrix((estimate.R, estimate.t))
            optimized_trajectory.append(qt)

        return optimized_trajectory
