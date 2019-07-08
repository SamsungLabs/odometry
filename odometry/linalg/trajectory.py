import numpy as np
import pandas as pd
import plotly.offline as ply
import plotly.graph_objs as go
from pyquaternion import Quaternion

from odometry.linalg.quaternion import QuaternionWithTranslation
from odometry.linalg.align import align
from odometry.linalg.linalg_utils import (convert_euler_angles_to_rotation_matrix,
                                          convert_rotation_matrix_to_euler_angles)

class AbstractTrajectory:
    def __init__(self):
        self.positions = []
        self.id = None

    def __repr__(self):
        s = str(self.id) + '\n'
        for pos in self.positions:
            s += pos.__str__()
            s += '\n'
        return s

    def __len__(self):
        return len(self.positions)

    def append(self, qt):
        self.positions.append(qt)

    @classmethod
    def from_quaternions(cls, quaternions_with_translation):
        trajectory = cls()
        for quat in quaternions_with_translation:
            qt = QuaternionWithTranslation.from_quaternion(quat)
            trajectory.append(qt)
        return trajectory

    def to_quaternions(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_quaternion())
        return result

    @classmethod
    def from_transformation_matrices(cls, transformations):
        trajectory = cls()
        for m in transformations:
            qt = QuaternionWithTranslation.from_transformation_matrix(m)
            trajectory.append(qt)
        return trajectory

    def to_transformation_matrices(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_transformation_matrix())
        return result

    @classmethod
    def from_euler_angles(cls, euler_angles_with_translation):
        trajectory = cls()
        for angle in euler_angles_with_translation:
            qt = QuaternionWithTranslation.from_euler_angles(angle)
            trajectory.append(qt)
        return trajectory

    def to_euler_angles(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_euler_angles())
        return result

    @classmethod
    def from_dataframe(cls, df):
        euler_angles_with_translation = df.to_dict(orient='records')
        return cls.from_euler_angles(euler_angles_with_translation)

    def to_dataframe(self):
        euler_angles_with_translation = self.to_euler_angles()
        return pd.DataFrame(euler_angles_with_translation)

    def to_global(self):
        return self

    def to_relative(self):
        return self


class GlobalTrajectory(AbstractTrajectory):

    def __init__(self):
        super().__init__()
        self.id = 'global'

    @classmethod
    def from_quaternions(cls, quaternions_with_translation):
        return super(GlobalTrajectory, cls).from_quaternions(quaternions_with_translation)

    @classmethod
    def from_transformation_matrices(cls, transformations):
        return super(GlobalTrajectory, cls).from_transformation_matrices(transformations)

    @classmethod
    def from_euler_angles(cls, euler_angles_with_translation):
        return super(GlobalTrajectory, cls).from_euler_angles(euler_angles_with_translation)

    @classmethod
    def from_dataframe(cls, df):
        return super(GlobalTrajectory, cls).from_dataframe(df)

    def to_semi_global(self):
        origin = self.positions[0].copy()
        semi_global = GlobalTrajectory()
        for pos in self.positions:
            semi_global.append(pos.to_semi_global(origin))
        return semi_global

    def to_relative(self):
        relative_trajectory = RelativeTrajectory()
        for pos_previous, pos_current in zip(self.positions[:-1], self.positions[1:]):
            relative_trajectory.append(pos_current.to_semi_global(pos_previous))
        return relative_trajectory

    @property
    def points(self):
        points = np.zeros((len(self), 3))
        for i, pos in enumerate(self.positions):
            points[i, :] = pos.translation[:]
        return points

    @property
    def rotation_matrices(self):
        rotation_matrices = np.zeros((len(self), 3, 3))
        for i, pos in enumerate(self.positions):
            rotation_matrices[i, :] = pos.rotation_matrix[:]
        return rotation_matrices

    def plot(self, file_name):
        line = go.Scatter3d(x=self.points[:, 0],
                            y=self.points[:, 1],
                            z=self.points[:, 2],
                            mode='lines',
                            name='trajectory')

        data = [line]
        layout = go.Layout( scene=dict(
                            xaxis=dict(
                                autorange=True),
                            yaxis=dict(
                                autorange=True),
                            zaxis=dict(
                                autorange=True))
                          )
        fig = go.Figure(data=data, layout=layout)
        ply.plot(fig, filename=file_name)

    def align_with(self, reference_trajectory, by='mean'):
        rotation_matrix, translation, scale = align(self.points, reference_trajectory.points, by=by)
        trajectory_aligned = GlobalTrajectory()
        for pos in self.positions:
            t_current = pos.translation
            rotation_matrix_current = pos.rotation_matrix
            t_aligned = scale * (t_current[None] @ rotation_matrix.T)[0] + translation
            rotation_matrix_aligned = rotation_matrix_current @ rotation_matrix.T
            qt_aligned = QuaternionWithTranslation.from_rotation_matrix((rotation_matrix_aligned, t_aligned))
            trajectory_aligned.append(qt_aligned)

        return trajectory_aligned


class RelativeTrajectory(AbstractTrajectory):

    def __init__(self):
        super().__init__()
        self.id = 'relative'

    @classmethod
    def from_quaternions(cls, quaternions_with_translation):
        return super(RelativeTrajectory, cls).from_quaternions(quaternions_with_translation)

    @classmethod
    def from_transformation_matrices(cls, transformations):
        return super(RelativeTrajectory, cls).from_transformation_matrices(transformations)

    @classmethod
    def from_euler_angles(cls, euler_angles_with_translation):
        return super(RelativeTrajectory, cls).from_euler_angles(euler_angles_with_translation)

    @classmethod
    def from_dataframe(cls, df):
        return super(RelativeTrajectory, cls).from_dataframe(df)

    def to_global(self):
        global_trajectory = GlobalTrajectory()
        global_trajectory.append(QuaternionWithTranslation())
        transformation_cumulative = QuaternionWithTranslation().to_transformation_matrix()
        for pos in self.positions:
            transformation_current = pos.to_transformation_matrix()
            transformation_cumulative = transformation_cumulative @ transformation_current
            qt_cumulative = QuaternionWithTranslation.from_transformation_matrix(transformation_cumulative)
            global_trajectory.append(qt_cumulative)

        return global_trajectory
