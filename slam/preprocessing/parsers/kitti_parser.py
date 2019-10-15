import os
import numpy as np
import pyquaternion
from functools import partial

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

    @staticmethod
    def _construct_image_filepaths(image_dir):
        return [os.path.join(image_dir, image_filename)
                for image_filename in sorted(os.listdir(image_dir))]

    def _load_data(self):
        self.image_filepaths = self._construct_image_filepaths(self.image_dir_left)
        self.image_right_filepaths = self._construct_image_filepaths(self.image_dir_right)

        if self.pose_filepath:
            self._load_poses()
            assert len(self.pose_matrices) == len(self.image_filepaths) == self.image_right_filepaths
            self.trajectory = list(zip(self.image_filepaths,
                                       self.image_right_filepaths,
                                       self.pose_matrices))
        else:
            self.trajectory = list(zip(self.image_filepaths,
                                       self.image_right_filepaths))

    @staticmethod
    def get_quaternion(item):
        rotation_matrix, translation = split_se3(item[1])
        quaternion = pyquaternion.Quaternion(matrix=rotation_matrix).elements
        return quaternion

    @staticmethod
    def get_translation(item):
        rotation_matrix, translation = split_se3(item[1])
        return translation

    def _parse_item(self, item):
        parsed_item = dict()
        parsed_item['path_to_rgb'] = item[0]
        parsed_item['path_to_rgb_right'] = item[1]

        if self.pose_filepath:
            parsed_item.update(dict(zip(['q_w', 'q_x', 'q_y', 'q_z'], self.get_quaternion(item))))
            parsed_item.update(dict(zip(['t_x', 't_y', 't_z'], self.get_translation(item))))

        return parsed_item
