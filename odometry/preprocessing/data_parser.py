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
from odometry.linalg.linalg_utils import (form_se3,
                                 convert_global_se3_matrices_to_relative,
                                 convert_relative_se3_matrices_to_euler)


class BaseParser:
    def __init__(self,
                 sequence_directory,
                 global_csv_filename):
        self.sequence_directory = sequence_directory
        self.global_csv_filename = global_csv_filename
        self.global_csv_path = os.path.join(self.sequence_directory, self.global_csv_filename)
        self.cols = ['path_to_rgb', 'path_to_depth']
    
    def _load_data(self):
        pass


    def _calculate_global_pose_matrices(self):
        pass

    def _make_absolute_filepath(self):
        for col in self.cols:
            self.global_dataframe[col] = self.global_dataframe[col].apply(
                lambda filename: os.path.join(self.directory, filename))
    
    def _create_global_dataframe(self):
        pass

    def parse(self):
        self._load_data()
        self._create_global_dataframe()
        self._make_absolute_filepath()
        self.global_dataframe.to_csv(self.global_csv_path, index=False)
        print('Parse ok...')


class TUMParser(BaseParser):
    def __init__(self,
                 sequence_directory,
                 directory,
                 global_csv_filename='global.csv'):
        super(TUMParser, self).__init__(sequence_directory, 
                                        global_csv_filename
                                        )
        self.directory = directory
        self.gt_txt_path = os.path.join(self.directory, 'groundtruth.txt')
        self.depth_txt_path = os.path.join(self.directory, 'depth.txt')
        self.rgb_txt_path = os.path.join(self.directory, 'rgb.txt')

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
        dataframe = dataframes[0]
        timestamp_col = timestamp_cols[0]
        for other_dataframe, other_timestamp_col in zip(dataframes[1:], timestamp_cols[1:]):
            timestamps, other_timestamps = \
                TUMParser.associate_timestamps(dataframe[timestamp_col].values, other_dataframe[other_timestamp_col].values)
            dataframe = dataframe[dataframe[timestamp_col].isin(timestamps)]
            dataframe.index = np.arange(len(dataframe))
            other_dataframe = other_dataframe[other_dataframe[other_timestamp_col].isin(other_timestamps)]
            other_dataframe.index = timestamps
            
            assert len(dataframe) == len(other_dataframe)
            dataframe = dataframe.join(other_dataframe, on=timestamp_col)
        return dataframe

    def _load_txt(self, txt_path, columns):
        dataframe = pd.read_csv(txt_path, skiprows=3, sep=' ', index_col=False, names=columns)
        dataframe.columns = columns
        timestamp_col = columns[0]
        dataframe[timestamp_col] = dataframe[timestamp_col].apply(float)
        return dataframe

    def _load_data(self):
        gt_dataframe = self._load_txt(self.gt_txt_path, columns=['timestamp_gt', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        depth_dataframe = self._load_txt(self.depth_txt_path, columns=['timestamp_depth', 'path_to_depth'])
        rgb_dataframe = self._load_txt(self.rgb_txt_path, columns=['timestamp_rgb', 'path_to_rgb'])
        self.dataframes = [depth_dataframe, rgb_dataframe, gt_dataframe]
        self.timestamp_cols = ['timestamp_depth', 'timestamp_rgb', 'timestamp_gt']

    def _create_global_dataframe(self):
        self.global_dataframe = self.associate_dataframes(self.dataframes, self.timestamp_cols)

    def __repr__(self):
        return 'TUMParser(dir={}, txt_path={}, global_csv_filename={})'.format(
            self.sequence_directory, self.gt_txt_path, self.global_csv_filename
            )
    
class RetailBotParser(TUMParser):
    
    def __init__(self,
                 sequence_directory,
                 directory,
                 global_csv_filename='global.csv'):
        super(RetailBotParser, self).__init__(sequence_directory, 
                                        global_csv_filename
                                        )
        self.directory = directory
        self.gt_txt_path = os.path.join(self.directory, 'pose.txt')
        self.depth_txt_path = os.path.join(self.directory, 'depth.txt')
        self.rgb_txt_path = os.path.join(self.directory, 'rgb.txt')
    
    def _load_txt(self, txt_path, columns):
        dataframe = pd.read_csv(txt_path, skiprows=0, sep=' ', index_col=False, names=columns)
        dataframe.columns = columns
        timestamp_col = columns[0]
        dataframe[timestamp_col] = dataframe[timestamp_col].apply(float)
        return dataframe
    
    def _load_pic(self, txt_path, columns):
        dataframe = pd.read_csv(txt_path, skiprows=0, sep=' ', index_col=False, names=columns)
        dataframe.columns = columns
        timestamp_col = columns[0]
        pic = columns[1]
        dataframe[timestamp_col] = dataframe[timestamp_col].apply(float)
        dataframe[pic] = dataframe[pic].apply(lambda x : self.directory + x[1:])
        return dataframe
    
    def _load_data(self):
        gt_dataframe = self._load_txt(self.gt_txt_path, columns=['timestamp_gt', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        depth_dataframe = self._load_pic(self.depth_txt_path, columns=['timestamp_depth', 'path_to_depth'])
        rgb_dataframe = self._load_pic(self.rgb_txt_path, columns=['timestamp_rgb', 'path_to_rgb'])
        self.dataframes = [depth_dataframe, rgb_dataframe, gt_dataframe]
        self.timestamp_cols = ['timestamp_depth', 'timestamp_rgb', 'timestamp_gt']
        
        
class DISCOMANParser(BaseParser):
    def __init__(self,
                 sequence_directory,
                 json_path,
                 global_csv_filename='global.csv'):
        super(DISCOMANParser, self).__init__(sequence_directory,
                                             global_csv_filename,)
        self.directory = os.path.dirname(json_path)
        self.image_directory = os.path.dirname(json_path)
        self.depth_directory = os.path.dirname(json_path)
        self.json_path = json_path

    def _load_data(self):
        with open(self.json_path) as read_file:
            data = json.load(read_file)
        self.trajectory = data['trajectory']['frames']

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
    def get_global_quaternion(item):
        return item['state']['global']['orientation']

    @staticmethod
    def get_global_translation(item):
        return item['state']['global']['position']

    @staticmethod
    def get_global_pose_matrix(item):
        return pyquaternion.Quaternion(item['state']['global']['orientation']).rotation_matrix

    def _create_global_dataframe(self):
        trajectory_parsed = []

        for point in self.trajectory[::5]:
            parsed_point = {}
            parsed_point['timestamp'] = self.get_timestamp(point)
            parsed_point['path_to_rgb'] = self.get_path_to_rgb(point)
            parsed_point['path_to_depth'] = self.get_path_to_depth(point)
            parsed_point.update(dict(zip(['qw', 'qx', 'qy', 'qz'], self.get_global_quaternion(point))))
            parsed_point.update(dict(zip(['x', 'y', 'z'], self.get_global_translation(point))))
            trajectory_parsed.append(parsed_point)

        self.global_dataframe = pd.DataFrame.from_dict(trajectory_parsed)

    def _calculate_global_pose_matrices(self):
        self.global_pose_matrices = np.array([self.get_global_pose_matrix(item) for item in self.trajectory])

    def __repr__(self):
        return 'JSONParser(dir={}, json_path={}, global_csv_filename={})'.format(
            self.sequence_directory, self.json_path, self.global_csv_filename)
    
    
class OldDISCOMANParser(DISCOMANParser):
    def __init__(self,
                 sequence_directory,
                 json_path,
                 global_csv_filename='global.csv'):
        super(OldDISCOMANParser, self).__init__(sequence_directory,
                                                json_path,
                                                global_csv_filename)

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
    def get_global_quaternion(item):
        return item['info']['agent_state']['orientation']

    @staticmethod
    def get_global_translation(item):
        return item['info']['agent_state']['position']

    @staticmethod
    def get_global_pose_matrix(item):
        return pyquaternion.Quaternion(item['info']['agent_state']['orientation']).rotation_matrix
    
    def _create_global_dataframe(self):
        trajectory_parsed = []

        for point in self.trajectory:
            parsed_point = {}
            parsed_point['timestamp'] = self.get_timestamp(point)
            parsed_point['path_to_rgb'] = self.get_path_to_rgb(point)
            parsed_point['path_to_depth'] = self.get_path_to_depth(point)
            parsed_point.update(dict(zip(['qw', 'qx', 'qy', 'qz'], self.get_global_quaternion(point))))
            parsed_point.update(dict(zip(['x', 'y', 'z'], self.get_global_translation(point))))
            trajectory_parsed.append(parsed_point)

        self.global_dataframe = pd.DataFrame.from_dict(trajectory_parsed)
        
        
class KITTIParser(BaseParser):
    def __init__(self, 
             sequence_directory,
             seq_id, 
             dataset_root='/dbstore/datasets/KITTI_odometry_2012/dataset/sequences',    
             global_csv_filename='global_csv'):
        self.sequence_directory = sequence_directory
        self.global_csv_filename = global_csv_filename
        self.dataset_root = dataset_root
        self.seq_id = seq_id
        self.directory = os.path.join(self.dataset_root, self.seq_id)
        self.global_csv_path = os.path.join(self.sequence_directory, self.global_csv_filename)
        self.cols = ['path_to_rgb']
        
    def read_poses_file(self):
        global_poses_matrixes = []
        with open(os.path.join(self.dataset_root,
                               '..', 'poses', '{}.txt'.format(self.seq_id))) as global_poses_fp:
            for line in global_poses_fp:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                global_poses_matrixes.append(T_w_cam0)
        return global_poses_matrixes

    def _load_data(self):
        root_to_rgb = os.path.join(self.dataset_root, self.seq_id, 'image_2')
        self.left_cam_frames = sorted(os.listdir(root_to_rgb))
        self.left_cam_frames = [os.path.join(root_to_rgb, img_name) for img_name in self.left_cam_frames]
        assert len(self.read_poses_file()) == len(self.left_cam_frames)
        
    def _create_global_dataframe(self):
        np.allclose = partial(np.allclose, atol=1e-6)
        trajectory_parsed = []
        for i, (pose, rgb) in enumerate(zip(self.read_poses_file(),self.left_cam_frames)):
            global_quaternion = pyquaternion.Quaternion(matrix=pose).elements
            global_translation = pose[:3, 3]
            parsed_point = {}
            parsed_point['path_to_rgb'] = rgb
            parsed_point.update(dict(zip(['qw', 'qx', 'qy', 'qz'], global_quaternion)))
            parsed_point.update(dict(zip(['x', 'y', 'z'], global_translation)))
            trajectory_parsed.append(parsed_point)
        self.global_dataframe = pd.DataFrame.from_dict(trajectory_parsed)
