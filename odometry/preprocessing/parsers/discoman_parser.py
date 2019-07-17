import os
import json
import pandas as pd

from .base_parser import BaseParser


class DISCOMANCSVParser(BaseParser):
    """For DISCOMAN from verison v10 with csv files
    """
    def __init__(self, src_dir):
        super(DISCOMANCSVParser, self).__init__(src_dir)

        self.name = DISCOMANCSVParser

        self.csv_path = os.path.join(src_dir, "camera_gt.csv")
        if not os.path.exists(self.csv_path):
            raise RuntimeError(f'Could not find csv file {self.csv_path}')

    def _load_data(self):
        self.df = pd.read_csv(self.csv_path, index_col=False)
    
    def _create_dataframe(self):
        self.df = self.df[::5].reset_index(drop=True)
        self.df = self.df.rename(columns={'id': 'timestamp', 
                                          'position.x': 't_x',
                                          'position.y': 't_y',
                                          'position.z': 't_z',
                                          'quaternion.w': 'q_w',
                                          'quaternion.x': 'q_x',
                                          'quaternion.y': 'q_y',
                                          'quaternion.z': 'q_z'})
        self.df.timestamp = self.df.timestamp.apply(lambda x: str(x).zfill(6))
        self.df['path_to_depth'] = self.df.timestamp + '_depth.png'
        self.df['path_to_rgb'] = self.df.timestamp + '_raycast.jpg'
