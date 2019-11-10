import os
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from collections import Iterable

from slam.preprocessing.parsers.tum_parser import TUMParser


class EuRoCParser(TUMParser):

    def __init__(self,
                 src_dir,
                 gt_txt_path=None,
                 rgb_txt_path=None,
                 cols=None):

        src_dir = os.path.join(src_dir, 'mav0')
        gt_txt_path = os.path.join(src_dir, 'state_groundtruth_estimate0/data.csv')
        rgb_txt_path = os.path.join(src_dir, 'cam0/data.csv')

        super(EuRoCParser, self).__init__(src_dir,
                                          gt_txt_path=gt_txt_path,
                                          rgb_txt_path=rgb_txt_path,
                                          cols=['path_to_rgb'])

        self.image_dir = os.path.join(self.src_dir, 'cam0', 'data')
        if not os.path.exists(self.image_dir):
            raise RuntimeError(f'Could not find image sub dir for trajectory: {self.image_dir}')

        self.calib_txt_path = os.path.join(self.src_dir, 'cam0/sensor.yaml')
        if not os.path.exists(self.calib_txt_path):
            raise RuntimeError(f'Could not find monocular calibration file for trajectory: {self.calib_txt_path}')

        self.name = 'EuRoCParser'

    def _load_txt(self, txt_path, columns, scale=1e-9):
        df = pd.read_csv(txt_path, index_col=False)
        df = df[df.columns[:len(columns)]]
        df.columns = columns
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float) * scale
        return df

    def _load_gt_txt(self):
        return self._load_txt(self.gt_txt_path,
                              columns=['timestamp_gt', 't_x', 't_y', 't_z', 'q_w', 'q_x', 'q_y', 'q_z'])

    def _load_calib(self):
        calib = dict()

        with open(self.calib_txt_path) as calib_fp:
            data_left = yaml.load(calib_fp)

        with open(self.calib_txt_path.replace('cam0', 'cam1')) as calib_fp:
            data_right = yaml.load(calib_fp)

        D_left = np.array(data_left['distortion_coefficients']) / 16.
        D_right = np.array(data_right['distortion_coefficients']) / 16.
        T_left = np.array(data_left['T_BS']['data']).reshape((4, 4))
        T_right = np.array(data_right['T_BS']['data']).reshape((4, 4))

        baseline_distance = np.linalg.norm((np.linalg.inv(T_left) @ T_right)[:3, 3])
        calib['baseline_distance'] = baseline_distance

        calib['D'] = D_left
        calib['D_right'] = D_right
        calib['T_body_cam'] = T_left

        # stereo
        K = np.array([[458.654, 0.0, 367.215],
                      [0.0, 457.296, 248.375],
                      [0.0, 0.0, 1.0]])
        R = np.array([[0.999966347530033, -0.001422739138722922, 0.008079580483432283],
                      [0.001365741834644127, 0.9999741760894847, 0.007055629199258132],
                      [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]])
        P = np.array([[435.2046959714599, 0, 367.4517211914062],
                      [0, 435.2046959714599, 252.2008514404297],
                      [0, 0, 1]])

        K_right = np.array([[457.587, 0.0, 379.999],
                            [0.0, 456.134, 255.238],
                            [0.0, 0.0, 1]])
        R_right = np.array([[0.9999633526194376, -0.003625811871560086, 0.007755443660172947],
                            [0.003680398547259526, 0.9999684752771629, -0.007035845251224894],
                            [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]])

        width, height = data_left['resolution']
        stereo_intrinsics_dict = {
            'f_x': P[0, 0] / width,
            'f_y': P[1, 1] / height,
            'c_x': P[0, 2] / width,
            'c_y': P[1, 2] / height,
        }

        calib.update(stereo_intrinsics_dict)
        calib['K'] = K
        calib['R'] = R
        calib['P'] = P

        calib['K_right'] = K_right
        calib['R_right'] = R_right
        return calib

    def _load_rgb_right_txt(self):
        return self._load_txt(self.rgb_txt_path.replace('cam0', 'cam1'),
                              columns=['timestamp_rgb_right', 'path_to_rgb_right'])

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_rgb_right_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_rgb_right', 'timestamp_gt']
        self.calib = self._load_calib()

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)
        self.df.path_to_rgb = self.image_dir + '/' + self.df.path_to_rgb
        self.df.path_to_rgb_right = self.image_dir.replace('cam0', 'cam1') + '/' + self.df.path_to_rgb_right

        for k, v in self.calib.items():
            if isinstance(v, Iterable):
                self.df[k] = [v for _ in range(len(self.df))]
            else:
                self.df[k] = v
