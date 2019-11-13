import os
import numpy as np
import pyquaternion
from functools import partial
from PIL import Image
from collections import OrderedDict

from slam.linalg import split_se3
from .elementwise_parser import ElementwiseParser


class KITTIParser(ElementwiseParser):

    def __init__(self,
                 src_dir):
        super(KITTIParser, self).__init__(src_dir)

        self.name = 'KITTIParser'

        self.image_dir_left = os.path.join(self.src_dir, 'image_2')
        if not os.path.exists(self.image_dir_left):
            raise RuntimeError(f'Could not find image sub dir for trajectory: {self.image_dir_left}')

        self.image_dir_right = os.path.join(self.src_dir, 'image_3')
        if not os.path.exists(self.image_dir_right):
            raise RuntimeError(f'Could not find image sub dir for trajectory: {self.image_dir_right}')

        self.calib_txt = os.path.join(self.src_dir, 'calib.txt')
        if not os.path.exists(self.calib_txt):
            raise RuntimeError(f'Could not find calib.txt for trajectory: {self.calib_txt}')

        trajectory_id = os.path.basename(src_dir)
        dataset_root = os.path.dirname(src_dir)
        self.pose_filepath = os.path.join(os.path.dirname(dataset_root), 'poses', '{}.txt'.format(trajectory_id))
        if not os.path.exists(self.pose_filepath):
            self.pose_filepath = None

        self.cols = ['path_to_rgb', 'path_to_rgb_right']

        np.allclose = partial(np.allclose, atol=1e-6)

    def _load_poses(self):
        self.pose_matrices = []
        with open(self.pose_filepath) as pose_fp:
            for line in pose_fp:
                t_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                t_w_cam0 = t_w_cam0.reshape(3, 4)
                t_w_cam0 = np.vstack((t_w_cam0, [0, 0, 0, 1]))
                self.pose_matrices.append(t_w_cam0)

    def _load_calib(self):
        data = dict()

        with open(self.calib_txt) as calib_fp:
            for line in calib_fp:
                key = line.split(':')[0]
                line_split = line.lstrip(f'{key}:').split()
                data[key] = np.array(line_split, dtype=float).reshape((3, 4))

        # Create 3x4 projection matrices
        P_rect_20 = np.reshape(data['P2'], (3, 4))
        P_rect_30 = np.reshape(data['P3'], (3, 4))

        # Compute the rectified extrinsics from cam0 to camN
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        T_cam0_velo = np.reshape(data['Tr'], (3, 4))
        T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
        T_cam2_velo = T2.dot(T_cam0_velo)
        T_cam3_velo = T3.dot(T_cam0_velo)

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo2 = np.linalg.inv(T_cam2_velo).dot(p_cam)
        p_velo3 = np.linalg.inv(T_cam3_velo).dot(p_cam)
        baseline_distance = np.linalg.norm(p_velo3 - p_velo2)

        calib = dict()
        calib.update({'f_x': P_rect_20[0, 0],
                      'f_y': P_rect_20[1, 1],
                      'c_x': P_rect_20[0, 2],
                      'c_y': P_rect_20[1, 2]})
        calib['baseline_distance'] = baseline_distance
        calib['T_body_cam'] = T2
        return calib

    @staticmethod
    def _construct_image_filepaths(image_dir):
        return [os.path.join(image_dir, image_filename)
                for image_filename in sorted(os.listdir(image_dir))]

    def _load_data(self):
        self.image_filepaths = self._construct_image_filepaths(self.image_dir_left)
        self.image_right_filepaths = self._construct_image_filepaths(self.image_dir_right)

        self.calib = self._load_calib()
        width, height = Image.open(self.image_filepaths[0]).size
        self.calib['f_x'] /= width
        self.calib['f_y'] /= height
        self.calib['c_x'] /= width
        self.calib['c_y'] /= height

        if self.pose_filepath:
            self._load_poses()
            assert len(self.pose_matrices) == len(self.image_filepaths) == len(self.image_right_filepaths)
            self.trajectory = list(zip(self.image_filepaths,
                                       self.image_right_filepaths,
                                       self.pose_matrices))
        else:
            self.trajectory = list(zip(self.image_filepaths,
                                       self.image_right_filepaths))

    @staticmethod
    def get_quaternion(item):
        rotation_matrix, translation = split_se3(item[2])
        quaternion = pyquaternion.Quaternion(matrix=rotation_matrix).elements
        return quaternion

    @staticmethod
    def get_translation(item):
        rotation_matrix, translation = split_se3(item[2])
        return translation

    def _parse_item(self, item):
        parsed_item = OrderedDict()
        parsed_item['path_to_rgb'] = item[0]
        parsed_item['path_to_rgb_right'] = item[1]

        if self.pose_filepath:
            parsed_item.update(OrderedDict(zip(['q_w', 'q_x', 'q_y', 'q_z'], self.get_quaternion(item))))
            parsed_item.update(OrderedDict(zip(['t_x', 't_y', 't_z'], self.get_translation(item))))

        parsed_item.update(self.calib)

        return parsed_item
