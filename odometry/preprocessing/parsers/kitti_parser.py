import os
import numpy as np
import pyquaternion
from functools import partial

from odometry.linalg import split_se3
from odometry.preprocessing.parsers.elementwise_parser import ElementwiseParser
        
        
class KITTIParser(ElementwiseParser):

    def __init__(self, 
                 trajectory_dir,
                 trajectory_id,
                 dataset_root='/dbstore/datasets/KITTI_odometry_2012/dataset/sequences'):
        super(KITTIParser, self).__init__(trajectory_dir)
        self.src_dir = os.path.join(dataset_root, trajectory_id)
        self.image_dir = os.path.join(self.src_dir, 'image_2')
        self.pose_filepath = os.path.join(os.path.dirname(dataset_root), 'poses', '{}.txt'.format(trajectory_id))
        self.cols = ['path_to_rgb']

        np.allclose = partial(np.allclose, atol=1e-6)

    def _load_poses(self):
        self.pose_matrices = []
        with open(self.pose_filepath) as pose_fp:
            for line in pose_fp:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                self.pose_matrices.append(T_w_cam0)

    def _load_image_filepaths(self):
        self.image_filepaths = [os.path.join(self.image_dir, image_filename) \
                                for image_filename in sorted(os.listdir(self.image_dir))]

    def _load_data(self):
        self._load_image_filepaths()
        self._load_poses()
        assert len(self.pose_matrices) == len(self.image_filepaths)
        self.trajectory = list(zip(self.pose_matrices, self.image_filepaths))

    @staticmethod
    def get_quaternion(item):
        rotation_matrix, translation = split_se3(item[0])
        quaternion = pyquaternion.Quaternion(matrix=rotation_matrix).elements
        return quaternion

    @staticmethod
    def get_translation(item):
        rotation_matrix, translation = split_se3(item[0])
        return translation

    @staticmethod
    def get_path_to_rgb(item):
        return item[1]

    def _parse_item(self, item):
        parsed_item = {}
        parsed_item['path_to_rgb'] = self.get_path_to_rgb(item)
        parsed_item.update(dict(zip(['q_w', 'q_x', 'q_y', 'q_z'], self.get_quaternion(item))))
        parsed_item.update(dict(zip(['t_x', 't_y', 't_z'], self.get_translation(item))))
        return parsed_item
