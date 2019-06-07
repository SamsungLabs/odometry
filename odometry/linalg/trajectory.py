import numpy as np

from pyquaternion import Quaternion
from linalg.quaternion import QuaternionWithTranslation
from linalg.align import align

import plotly.offline as ply
import plotly.graph_objs as go

from linalg.linalg_utils import (convert_euler_angles_to_rotation_matrix, 
                                 convert_rotation_matrix_to_euler_angles)

class AbstractTrajectory:
    def __init__(self):
        self.positions = []

    def __str__(self) :
        s = ''
        for pos in self.positions:
            s += pos.__str__()
            s += '\n'
        return s
    
    def __len__(self):
        return len(self.positions)

    def from_quaternions(self, qt, quaternions_with_translation):
        self.positions = []
        for quat in quaternions_with_translation:
            qt.from_quaternion(quat)
            self.positions.append(qt)

    def to_quaternions(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_quaternion())
        return result

    def from_transformation_matrices(self, id, transformations):
        self.positions = []
        for m in transformations:
            qt = QuaternionWithTranslation()
            qt.from_transformation_matrix(m)
            self.positions.append(qt)

    def to_transformation_matrices(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_transformation_matrix())
        return result

    def from_euler_angles(self, id, euler_angles_and_translation):
        self.positions = []
        for angle in euler_angles_and_translation:
            qt = QuaternionWithTranslation()
            qt.from_euler_angles(angle)
            self.positions.append(qt)

    def to_euler_angles(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_euler_angles())
        return result

    def from_axis_angle(self, id, axis_angle_and_translation):
        self.positions = []
        for angle in axis_angle_and_translation:
            qt = QuaternionWithTranslation()
            qt.from_axis_angle(angle)
            self.positions.append(qt)

    def to_axis_angle(self):
        result = []
        for pos in self.positions:
            result.append(pos.to_axis_angle())
        return result


class GlobalTrajectory(AbstractTrajectory):

    def __init__(self):
        super().__init__()
        self.id = 'global'

    def from_tum(self, tum_file, step=1, nlines=-1):
        self.positions = []
        f = open(tum_file)
        i = 0
        for l in f.readlines():
            if i%step != 0:
                i += 1
                continue
            if nlines > 0 and i > nlines:
                break

            split_line = l.split()
            #TUM format:  tx ty tz qx qy qz qw
            arr = np.array(split_line[-7:]).astype(np.float)
            t = arr[0:3]
            q = Quaternion(w=arr[6], x=arr[3], y=arr[4], z=arr[5])
            self.positions.append(GlobalQuaternionWithTranslation(q, t))
            i += 1
        f.close()

    def from_quaternions(self, quaternions_with_translation):
        self.positions = []
        for quat in quaternions_with_translation:
            qt = QuaternionWithTranslation().from_quaternion(quat)
            self.positions.append(qt)

    def from_transformation_matrices(self, transformations):
        super().from_transformation_matrices(self.id, transformations)

    def from_euler_angles(self, euler_angles_and_translation):
        super().from_euler_angles(self.id, euler_angles_and_translation)

    def from_axis_angle(self, axis_angle_and_translation):
        super().from_axis_angle(self.id, axis_angle_and_translation)

    def to_semi_global(self):
        origin = QuaternionWithTranslation(self.positions[0].quaternion, self.positions[0].translation)
        result = GlobalTrajectory()
        for pos in self.positions:
            result.positions.append(pos.to_semi_global(origin))
        return result

    def global_to_relative(self):
        relative_trajectory = RelativeTrajectory()
        for i in range(1, len(self.positions)):
            pos_current = self.positions[i]
            pos_previous = self.positions[i-1]
            q_current = pos_current.quaternion
            t_current = pos_current.translation
            q_previous = pos_previous.quaternion
            t_previous = pos_previous.translation

            transformation_current = q_current.transformation_matrix
            transformation_current[0, -1] = t_current[0]
            transformation_current[1, -1] = t_current[1]
            transformation_current[2, -1] = t_current[2]

            transformation_previous = q_previous.transformation_matrix
            transformation_previous[0, -1] = t_previous[0]
            transformation_previous[1, -1] = t_previous[1]
            transformation_previous[2, -1] = t_previous[2]

            transformation_relative = np.linalg.inv(transformation_previous)@transformation_current

            q_relative = Quaternion(matrix=transformation_relative).normalised
            t_relative = [transformation_relative[0, -1], transformation_relative[1, -1], transformation_relative[2, -1]]
            relative_trajectory.positions.append(QuaternionWithTranslation(q_relative, t_relative))
        return relative_trajectory
    
    @property
    def points(self):
        points = np.zeros(np.array([len(self.positions), 3]))
        i = 0
        for pos in self.positions:
            points[i, :] = pos.translation[:]
            i += 1
        return points

    def plot(self, file_name):
        line_trajectory = go.Scatter3d(x=self.points[:, 0],
                                 y=self.points[:, 1],
                                 z=self.points[:, 2],
                                 mode='lines',
                                 name='trajectory')

        data = [line_trajectory]
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
        for point in self.points:
            t_aligned = scale * np.dot(point, rotation_matrix) + translation
            rotation_matrix_aligned = convert_euler_angles_to_rotation_matrix(np.dot(point, rotation_matrix))
            q_aligned = Quaternion(matrix=rotation_matrix_aligned).normalised
            trajectory_aligned.positions.append(QuaternionWithTranslation(q_aligned, t_aligned))

        return trajectory_aligned


class RelativeTrajectory(AbstractTrajectory):

    def __init__(self):
        super().__init__()
        self.id = 'relative'

    def from_quaternions(self, quaternions_with_translation):
        super().from_quaternions(self.id, quaternions_with_translation)

    def from_transformation_matrices(self, transformations):
        super().from_transformation_matrices(self.id, transformations)

    def from_euler_angles(self, euler_angles_and_translation):
        super().from_euler_angles(self.id, euler_angles_and_translation)

    def from_axis_angle(self, axis_angle_and_translation):
        super().from_axis_angle(self.id, axis_angle_and_translation)

    def relative_to_global(self):
        q_cumulative = Quaternion()
        t_cumulative = [0, 0, 0]
        global_trajectory = GlobalTrajectory()
        global_trajectory.positions.append(QuaternionWithTranslation(q_cumulative, t_cumulative))
        for pos in self.positions:

            transformation_cumulative = q_cumulative.transformation_matrix
            transformation_cumulative[0, -1] = t_cumulative[0]
            transformation_cumulative[1, -1] = t_cumulative[1]
            transformation_cumulative[2, -1] = t_cumulative[2]

            transformation_current = pos.quaternion.transformation_matrix
            transformation_current[0, -1] = pos.translation[0]
            transformation_current[1, -1] = pos.translation[1]
            transformation_current[2, -1] = pos.translation[2]

            transformation_cumulative = transformation_cumulative@transformation_current
            q_cumulative = Quaternion(matrix=transformation_cumulative).normalised
            t_cumulative = [transformation_cumulative[0, -1], transformation_cumulative[1, -1], transformation_cumulative[2, -1]]

            global_trajectory.positions.append(QuaternionWithTranslation(q_cumulative, t_cumulative))
        return global_trajectory
