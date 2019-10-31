import os
import numpy as np
import pandas as pd
from PIL import Image

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

        self.image_dir_left = os.path.join(self.src_dir, 'cam0', 'data')
        if not os.path.exists(self.image_dir_left):
            raise RuntimeError(f'Could not find image sub dir for trajectory: {self.image_dir_left}')

        self.image_dir_right = os.path.join(self.src_dir, 'cam1', 'data')
        if not os.path.exists(self.image_dir_right):
            raise RuntimeError(f'Could not find image sub dir for trajectory: {self.image_dir_right}')

        self.calib_txt_path = os.path.join(self.src_dir, 'cam0/sensor.yaml')
        if not os.path.exists(self.calib_txt_path):
            raise RuntimeError(f'Could not find calib.txt for trajectory: {self.calib_txt_path}')

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
            lines = [line.strip() for line in calib_fp.readlines()]

            for i, line in enumerate(lines):
                if line.startswith('data:'):
                    extrinsics_str = ''.join([line.lstrip('data: ')] + lines[i+1:i+4])
                    extrinsics = np.array(eval(extrinsics_str)).reshape(4, 4)
                    translation_between_cameras = extrinsics[:3, 3]
                    baseline_distance = np.linalg.norm(translation_between_cameras)
                    calib['baseline_distance'] = baseline_distance

                if line.startswith('intrinsics:'):
                    f_x, f_y, c_x, c_y = eval(line.lstrip('intrinsics: '))
                    calib.update({'f_x': f_x, 'f_y': f_y, 'c_x': c_x, 'c_y': c_y})

        return calib

    def _load_rgb_right_txt(self):
        return self._load_txt(self.rgb_txt_path.replace('cam0', 'cam1'),
                              columns=['timestamp_rgb_right', 'path_to_rgb_right'])

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_rgb_right_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_rgb_right', 'timestamp_gt']
        self.intrinsics_dict = self._load_calib()

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)
        self.df.path_to_rgb = self.image_dir_left + '/' + self.df.path_to_rgb
        self.df.path_to_rgb_right = self.image_dir_right + '/' + self.df.path_to_rgb_right

        width, height = Image.open(os.path.join(self.src_dir, self.df.path_to_rgb.values[0])).size
        self.intrinsics_dict['f_x'] /= width
        self.intrinsics_dict['f_y'] /= height
        self.intrinsics_dict['c_x'] /= width
        self.intrinsics_dict['c_y'] /= height

        for k, v in self.intrinsics_dict.items():
            self.df[k] = v
