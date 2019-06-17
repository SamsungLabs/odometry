import os
import json
import tqdm
import itertools
import collections
import numpy as np
import pandas as pd
import math
import pyquaternion
from functools import partial

from odometry.linalg import split_se3


class BaseParser:

    def __init__(self, trajectory_dir):
        self.dir = trajectory_dir
        self.cols = ['path_to_rgb', 'path_to_depth']

    def _load_data(self):
        raise NotImplementedError

    def _make_absolute_filepath(self):
        for col in self.cols:
            self.df[col] = self.df[col].apply(
                lambda filename: os.path.abspath(os.path.join(self.src_dir, filename)))

    def _create_dataframe(self):
        raise NotImplementedError

    def run(self):
        self._load_data()
        self._create_dataframe()
        self._make_absolute_filepath()
        return self.df


class ElementwiseParser(BaseParser):

    def _parse_item(self, item):
        raise NotImplementedError

    def _create_dataframe(self):
        trajectory_parsed = [self._parse_item(item) for item in self.trajectory]
        self.df = pd.DataFrame.from_dict(trajectory_parsed)


class TUMParser(BaseParser):

    def __init__(self,
                 trajectory_dir,
                 src_dir):
        super(TUMParser, self).__init__(trajectory_dir)
        self.src_dir = src_dir
        self.gt_txt_path = os.path.join(self.src_dir, 'groundtruth.txt')
        self.depth_txt_path = os.path.join(self.src_dir, 'depth.txt')
        self.rgb_txt_path = os.path.join(self.src_dir, 'rgb.txt')
        self.skiprows = 3

    @staticmethod
    def associate_timestamps(timestamps, other_timestamps, max_difference=0.02):
        timestamps = list(timestamps)
        other_timestamps = list(other_timestamps)
        potential_matches = [(timestamp, other_timestamp)
                             for timestamp in timestamps
                             for other_timestamp in other_timestamps
                             if abs(timestamp - other_timestamp) < max_difference]
        potential_matches.sort(key=lambda x: abs(x[0] - x[1]))

        matches = []
        for timestamp, other_timestamp in potential_matches:
            if timestamp in timestamps and other_timestamp in other_timestamps:
                timestamps.remove(timestamp)
                other_timestamps.remove(other_timestamp)
                matches.append((timestamp, other_timestamp))

        matches.sort()
        return list(zip(*matches))

    @staticmethod
    def associate_dataframes(dataframes, timestamp_cols):
        df = dataframes[0]
        timestamp_col = timestamp_cols[0]
        for other_df, other_timestamp_col in zip(dataframes[1:], timestamp_cols[1:]):
            timestamps, other_timestamps = \
                TUMParser.associate_timestamps(df[timestamp_col].values, other_df[other_timestamp_col].values)
            df = df[df[timestamp_col].isin(timestamps)]
            df.index = np.arange(len(df))
            other_df = other_df[other_df[other_timestamp_col].isin(other_timestamps)]
            other_df.index = timestamps

            assert len(df) == len(other_df)
            df = df.join(other_df, on=timestamp_col)
        return df

    def _load_txt(self, txt_path, columns):
        df = pd.read_csv(txt_path, skiprows=self.skiprows, sep=' ', index_col=False, names=columns)
        df.columns = columns
        timestamp_col = columns[0]
        df[timestamp_col] = df[timestamp_col].apply(float)
        return df

    def _load_gt_txt(self):
        return self._load_txt(self.gt_txt_path, columns=['timestamp_gt', 't_x', 't_y', 't_z', 'q_x', 'q_y', 'q_z', 'q_w'])

    def _load_rgb_txt(self):
        return self._load_txt(self.rgb_txt_path, columns=['timestamp_rgb', 'path_to_rgb'])

    def _load_depth_txt(self):
        return self._load_txt(self.depth_txt_path, columns=['timestamp_depth', 'path_to_depth'])

    def _load_data(self):
        self.dataframes = [self._load_depth_txt(), self._load_rgb_txt(), self._load_gt_txt()]
        self.timestamp_cols = ['timestamp_depth', 'timestamp_rgb', 'timestamp_gt']

    def _create_dataframe(self):
        self.df = self.associate_dataframes(self.dataframes, self.timestamp_cols)

    def __repr__(self):
        return 'TUMParser(dir={}, txt_path={})'.format(self.dir, self.gt_txt_path)


class RetailBotParser(TUMParser):
    
    def __init__(self,
                 trajectory_dir,
                 src_dir):
        super(RetailBotParser, self).__init__(trajectory_dir, src_dir)
        self.gt_txt_path = os.path.join(self.src_dir, 'pose.txt')
        self.depth_txt_path = os.path.join(self.src_dir, 'depth.txt')
        self.rgb_txt_path = os.path.join(self.src_dir, 'rgb.txt')
        self.skiprows = 0

    def __repr__(self):
        return 'RetailBotParser(dir={}, txt_path={})'.format(self.dir, self.gt_txt_path)


class DISCOMANParser(ElementwiseParser):

    def __init__(self,
                 trajectory_dir,
                 json_path):
        super(DISCOMANParser, self).__init__(trajectory_dir)
        self.src_dir = os.path.dirname(json_path)
        self.json_path = json_path

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
        parsed_item = {}
        parsed_item['timestamp'] = self.get_timestamp(item)
        parsed_item['path_to_rgb'] = self.get_path_to_rgb(item)
        parsed_item['path_to_depth'] = self.get_path_to_depth(item)
        parsed_item.update(dict(zip(['q_w', 'q_x', 'q_y', 'q_z'], self.get_quaternion(item))))
        parsed_item.update(dict(zip(['t_x', 't_y', 't_z'], self.get_translation(item))))
        return parsed_item

    def __repr__(self):
        return 'DISCOMANParser(dir={}, json_path={})'.format(self.dir, self.json_path)
    
    
class OldDISCOMANParser(DISCOMANParser):

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
    
    def __repr__(self):
        return 'OldDISCOMANParser(dir={}, json_path={})'.format(self.dir, self.json_path)
        
        
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
