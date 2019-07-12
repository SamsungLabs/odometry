import os
import numpy as np
import pandas as pd

from odometry.preprocessing.parsers.tum_parser import TUMParser


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

    def _load_txt(self, txt_path, columns, scale=1):
        df = pd.read_csv(txt_path, index_col=False)
        df = df[df.columns[:len(columns)]]
        df.columns = columns
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float) * scale
        return df

    def _load_gt_txt(self):
        return self._load_txt(self.gt_txt_path,
                              columns=['timestamp_gt', 't_x', 't_y', 't_z', 'q_w', 'q_x', 'q_y', 'q_z'],
                              scale=1e-9)

    def _load_data(self):
        self.dataframes = [self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_rgb', 'timestamp_gt']

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)
        self.df.path_to_rgb = self.df.path_to_rgb.apply(lambda f: os.path.join('camera', 'images', f))
