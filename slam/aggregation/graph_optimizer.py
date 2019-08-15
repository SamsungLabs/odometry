import g2o
import numpy as np
import pandas as pd

from slam.aggregation.base_aggregator import BaseAggregator
from slam.linalg import (GlobalTrajectory,
                         RelativeTrajectory,
                         convert_euler_angles_to_rotation_matrix,
                         form_se3)


class GraphOptimizer(BaseAggregator):
    def __init__(self):
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        self.optimizer = g2o.SparseOptimizer()
        self.optimizer.set_verbose(True)
        self.optimizer.set_algorithm(solver)

        self.measurements = pd.DataFrame()

    def append(self, df):
        self.measurements = self.measurements.append(df).reset_index(drop=True)

    def get_trajectory(self):
        is_adjustment_measurements = (self.measurements.to_index - self.measurements.from_index) == 1
        adjustment_measurements = self.measurements[is_adjustment_measurements].reset_index(drop=True)
        trajectory = RelativeTrajectory().from_dataframe(adjustment_measurements).to_global()

        for index, position in enumerate(trajectory.positions):
            pose = g2o.Isometry3d()
            pose.set_translation(position.translation)
            q = g2o.Quaternion(position.quaternion.elements)
            pose.set_rotation(q)
            vertex = g2o.VertexSE3()
            vertex.set_estimate(pose)
            vertex.set_id(index)
            vertex.set_fixed(True) if index == 0 else None
            self.optimizer.add_vertex(vertex)

        for index, row in self.measurements.iterrows():
            euler_angles = row[['euler_x', 'euler_y', 'euler_z']].values
            translation = row[['t_x', 't_y', 't_z']].values
            rotation_matrix = convert_euler_angles_to_rotation_matrix(euler_angles)

            measurement = g2o.Isometry3d()
            measurement.set_translation(translation)
            q = g2o.Quaternion(rotation_matrix)
            measurement.set_rotation(q)
            edge = g2o.EdgeSE3()
            edge.set_measurement(measurement)

            edge.set_information(np.eye(6))
            edge.set_vertex(0, self.optimizer.vertex(int(row['from_index'])))
            edge.set_vertex(1, self.optimizer.vertex(int(row['to_index'])))
            self.optimizer.add_edge(edge)

        self.optimizer.initialize_optimization()
        self.optimizer.optimize(100)

        transformation_matrices = list()
        for index in range(len(self.optimizer.vertices())):
            estimate = self.optimizer.vertex(index).estimate()
            transformation_matrix = form_se3(estimate.R, estimate.t)
            transformation_matrices.append(transformation_matrix)

        optimized_trajectory = GlobalTrajectory().from_transformation_matrices(transformation_matrices)
        return optimized_trajectory