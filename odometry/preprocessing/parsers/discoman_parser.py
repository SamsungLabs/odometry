import os
import json
import pandas as pd

from .base_parser import BaseParser
from .elementwise_parser import ElementwiseParser


class DISCOMANJSONParser(ElementwiseParser):

    """For DISCOMAN verison v10 with json
    """

    def __init__(self, src_dir):
        super(DISCOMANJSONParser, self).__init__(src_dir)

        self.name = 'DISCOMANParser'
        self.json_path = os.path.join(src_dir, '0_traj.json')
        if not os.path.exists(self.json_path):
            raise RuntimeError(f'Could not find json file {self.json_path}')

    def _load_data(self):
        with open(self.json_path) as read_file:
            data = json.load(read_file)
        self.trajectory = data['trajectory']['frames'][::5]

    @staticmethod    
    def get_path_to_rgb(item):
        return '{}_raycast.jpg'.format(item['id'])

    @staticmethod
    def get_path_to_depth(item):
        return '{}_depth.png'.format(item['id'])

    @staticmethod
    def get_timestamp(item):
        return item['id']

    @staticmethod
    def get_quaternion(item):
        return item['state']['global']['orientation']

    @staticmethod
    def get_translation(item):
        return item['state']['global']['position']

    def _parse_item(self, item):
        parsed_item = dict()
        parsed_item['timestamp'] = self.get_timestamp(item)
        parsed_item['path_to_rgb'] = self.get_path_to_rgb(item)
        parsed_item['path_to_depth'] = self.get_path_to_depth(item)
        parsed_item.update(dict(zip(['q_w', 'q_x', 'q_y', 'q_z'], self.get_quaternion(item))))
        parsed_item.update(dict(zip(['t_x', 't_y', 't_z'], self.get_translation(item))))
        return parsed_item
    
    
class OldDISCOMANParser(DISCOMANJSONParser):
    """For DISCOMAN verison earlies than v10
    """

    def __init__(self, src_dir):
        super(OldDISCOMANParser).__init__(src_dir)
        self.name = 'old DISCOMANParser'

    def _load_data(self):
        with open(self.json_path) as read_file:
            data = json.load(read_file)
            self.trajectory = data['data']

    @staticmethod
    def get_path_to_rgb(item):
        return '{}_raycast.jpg'.format(str(item['time']).zfill(6))

    @staticmethod
    def get_path_to_depth(item):
        return '{}_depth.png'.format(str(item['time']).zfill(6))

    @staticmethod
    def get_timestamp(item):
        return item['time']

    @staticmethod
    def get_quaternion(item):
        return item['info']['agent_state']['orientation']

    @staticmethod
    def get_translation(item):
        return item['info']['agent_state']['position']
    
    
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
