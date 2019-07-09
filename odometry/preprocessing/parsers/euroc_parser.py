import os
import numpy as np
import pandas as pd

from odometry.preprocessing.parsers.tum_parser import TUMParser


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
        self.name = 'EuRoCParser'

    def _load_txt(self, txt_path, columns):
        df = pd.read_csv(txt_path, index_col=False)
        df = df[df.columns[:len(columns)]]
        df.columns = columns
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float) * 1e-9
        return df
    
    def _load_gt_txt(self):
        return self._load_txt(self.gt_txt_path,
                              columns=['timestamp_gt', 't_x', 't_y', 't_z', 'q_w', 'q_x', 'q_y', 'q_z'])

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)
        self.df.path_to_rgb = self.df.path_to_rgb.apply(lambda f: os.path.join('cam0', 'data', f))


class ZJUParser(TUMParser):

    def __init__(self, src_dir):
        src_dir = src_dir
        gt_txt_path = os.path.join(src_dir, 'groundtruth/euroc_gt.csv')
        rgb_txt_path = os.path.join(src_dir, 'camera/data.csv')

        super(ZJUParser, self).__init__(src_dir,
                                          gt_txt_path=gt_txt_path,
                                          rgb_txt_path=rgb_txt_path,
                                          cols=['path_to_rgb'])
        self.name = 'ZJUParser'

    def _load_txt(self, txt_path, columns):
        df = pd.read_csv(txt_path, index_col=False)
        df = df[df.columns[:len(columns)]]
        df.columns = columns
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float)
        return df

    def _load_gt_txt(self):
        columns=['timestamp_gt', 't_x', 't_y', 't_z', 'q_w', 'q_x', 'q_y', 'q_z']
        df = self._load_txt(self.gt_txt_path,
                              columns=columns)
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float) * 1e-9
        return df

    def _load_rgb_txt(self):
        return self._load_txt(self.rgb_txt_path, columns=['timestamp_rgb', 'path_to_rgb'])

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)
        self.df.path_to_rgb = self.df.path_to_rgb.apply(lambda f: os.path.join('camera', 'images', f))
