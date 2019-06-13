import os
import json
import tqdm
import itertools
import collections
import numpy as np
import pandas as pd
import math
import pyquaternion

from odometry.linalg.linalg_utils import (form_se3,
                                          convert_global_se3_matrices_to_relative,
                                          convert_relative_se3_matrices_to_euler)


class BaseParser:
    def __init__(self,
                 sequence_directory,
                 global_csv_filename,
                 relative_csv_filename,
                 stride=1):
        self.sequence_directory = sequence_directory
        self.global_csv_filename = global_csv_filename
        self.relative_csv_filename = relative_csv_filename
        self.stride = stride
        
        self.global_csv_path = os.path.join(self.sequence_directory, self.global_csv_filename)
        self.relative_csv_path = os.path.join(self.sequence_directory, self.relative_csv_filename)
    
    def _load_data(self):
        pass

    @staticmethod
    def flatten(item):
        return list(itertools.chain(*item))

    def _calculate_global_pose_matrices(self):
        pass

    def _make_absolute_filepath(self):
        self.global_dataframe.path_to_rgb = self.global_dataframe.path_to_rgb.apply(
            lambda filename: os.path.join(self.directory, filename))
        self.global_dataframe.path_to_depth = self.global_dataframe.path_to_depth.apply(
            lambda filename: os.path.join(self.directory, filename))
    
    def _create_global_dataframe(self):
        pass

    def _create_relative_dataframe(self):
        self._calculate_global_pose_matrices()
        global_translations = self.global_dataframe[['x', 'y', 'z']].values
        se3_matrices = [form_se3(rotation_matrix, translation) \
                        for rotation_matrix, translation in zip(self.global_pose_matrices, global_translations)]
        relative_se3_matrices = convert_global_se3_matrices_to_relative(np.array(se3_matrices), self.stride)
        relative_rotations_euler, relative_translations_euler = \
            convert_relative_se3_matrices_to_euler(relative_se3_matrices)

        relative_data_list = [list(row[:-2]) + self.flatten(row[-2:]) for row in (
            zip(self.global_dataframe.path_to_rgb[:-self.stride],
                 self.global_dataframe.path_to_rgb[self.stride:],
                 self.global_dataframe.path_to_depth[:-self.stride],
                 self.global_dataframe.path_to_depth[self.stride:],
                 relative_rotations_euler[:-self.stride],
                 relative_translations_euler[:-self.stride]
                )
        )]

        self.relative_dataframe = pd.DataFrame(
            data=relative_data_list,
            columns=[
                'path_to_rgb', 'path_to_next_rgb',
                'path_to_depth', 'path_to_next_depth',
                'euler_x', 'euler_y', 'euler_z',
                'x', 'y', 'z'
            ])

    def parse(self):
        self._load_data()
        
        self._create_global_dataframe()
        self._make_absolute_filepath()
        self.global_dataframe.to_csv(self.global_csv_path, index=False)
        
        self._create_relative_dataframe()
        self.relative_dataframe.to_csv(self.relative_csv_path, index=False)


class DISCOMANParser(BaseParser):
    def __init__(self,
                 sequence_directory,
                 json_path,
                 global_csv_filename='global.csv',
                 relative_csv_filename='relative.csv',
                 stride=1):
        super(DISCOMANParser, self).__init__(sequence_directory,
                                             global_csv_filename,
                                             relative_csv_filename,
                                             stride)
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

        for point in self.trajectory:
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
        return 'JSONParser(dir={}, json_path={}, global_csv_filename={}, relative_csv_filename={}, stride={})'.format(
            self.sequence_directory, self.json_path, self.global_csv_filename, self.relative_csv_filename,
            self.stride)


class OldDISCOMANParser(DISCOMANParser):
    def __init__(self,
                 sequence_directory,
                 json_path,
                 global_csv_filename='global.csv',
                 relative_csv_filename='relative.csv',
                 stride=1):
        super(OldDISCOMANParser, self).__init__(sequence_directory,
                                                json_path,
                                                global_csv_filename,
                                                relative_csv_filename,
                                                stride)

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


class TUMParser(BaseParser):
    def __init__(self,
                 sequence_directory,
                 txt_path,
                 global_csv_filename='global.csv',
                 relative_csv_filename='relative.csv',
                 stride=1):
        super(TUMParser, self).__init__(sequence_directory, 
                                        global_csv_filename,
                                        relative_csv_filename,
                                        stride)
        self.directory = os.path.dirname(txt_path)
        self.image_directory = os.path.join(os.path.dirname(txt_path), 'rgb')
        self.depth_directory = os.path.join(os.path.dirname(txt_path), 'depth')
        self.gt_txt_path = txt_path
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
        dataframe = pd.read_csv(txt_path, skiprows=2, sep=' ', index_col=False, names=columns)
        #dataframe.drop(dataframe.columns[-1], axis=1, inplace=True)
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

    @staticmethod
    def get_global_pose_matrix(item):
        return pyquaternion.Quaternion(item).rotation_matrix
    
    def _calculate_global_pose_matrices(self):
        self.global_pose_matrices = np.array([self.get_global_pose_matrix(q) \
                                              for q in self.global_dataframe[['qw', 'qx', 'qy', 'qz']].values])

    def _create_global_dataframe(self):
        self.global_dataframe = self.associate_dataframes(self.dataframes, self.timestamp_cols)

    def __repr__(self):
        return 'TUMParser(dir={}, txt_path={}, global_csv_filename={}, relative_csv_filename={}, stride={})'.format(
            self.sequence_directory, self.gt_txt_path, self.global_csv_filename, self.relative_csv_filename,
            self.stride)
