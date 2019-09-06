import pandas as pd
from statistics import mean, median
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from slam.linalg import (convert_euler_angles_to_rotation_matrix, 
                         GlobalTrajectory,
                         RelativeTrajectory,
                         euler_to_quaternion,
                         shortest_path_with_normalization)

mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
std_cols = [c + '_confidence' for c in mean_cols]

def get_pair_frame_stat(df):
    relative_gt = df2slam_predict(df)
    is_adjustment = (relative_gt.to_index - relative_gt.from_index) <= 1
    adjustment_measurements = relative_gt[is_adjustment].reset_index(drop=True)
    gt_trajectory = RelativeTrajectory.from_dataframe(adjustment_measurements[['euler_x', 'euler_y',
                                                                               'euler_z', 't_x', 't_y',
                                                                               't_z']]).to_global()
    global_gt = gt_trajectory.to_dataframe()
    distances=[] 
    euclid_euler =[]
    components =[]
    abs_distance = []
    symetric_distance =[]
    intrinsic_distance =[]
    euclid_quaternion=[]
    translation_columns = ['t_x','t_y','t_z']

    rotation_columns = ['euler_x', 'euler_y', 'euler_z']
    for ind, values in relative_gt[['to_index', 'from_index']].iterrows():
        to_translation = global_gt.iloc[values.to_index][translation_columns]
        from_translation = global_gt.iloc[values.from_index][translation_columns]
        distances.append(np.linalg.norm(to_translation - from_translation)) 
        to_rotation = global_gt.iloc[values.to_index][rotation_columns]
        from_rotation = global_gt.iloc[values.from_index][rotation_columns]
        euler_x, euler_y, euler_z = shortest_path_with_normalization(to_rotation,from_rotation)
        euclid_euler_distance = np.sqrt(euler_x**2 + euler_y**2 + euler_z**2)
        euclid_euler.append(euclid_euler_distance)
        quaternion_from = Quaternion(euler_to_quaternion(from_rotation))
        quaternion_to = Quaternion(euler_to_quaternion(to_rotation))

        abs_distance.append(Quaternion.absolute_distance(quaternion_to.unit ,quaternion_from.unit))
        intrinsic_distance.append(Quaternion.distance(quaternion_to.unit ,quaternion_from.unit))
        symetric_distance.append(Quaternion.sym_distance(quaternion_to.unit ,quaternion_from.unit))
        euclid_quaternion.append(min((quaternion_to.unit - quaternion_from.unit).norm, 
                                     (quaternion_to.unit + quaternion_from.unit).norm))
#     relative_gt['euclidian_distances'] = distances
#     relative_gt['euclid_euler_distance'] = euclid_euler
#     relative_gt['abs_distance'] = abs_distance
#     relative_gt['symetric_distance'] = symetric_distance
#     relative_gt['intrinsic_distance'] = intrinsic_distance
#     relative_gt['euclid_quaternion'] = euclid_quaternion
    maxmin_dict={}
    for name, l in zip(('distances','euclid_euler', 'abs_distance', 
                  'symetric_distance',' intrinsic_distance', 'euclid_quaternion'),
                 (distances, euclid_euler, abs_distance, 
                  symetric_distance, intrinsic_distance, euclid_quaternion)):
        relative_gt[name] = l

        maxmin_dict.update({name :{'maximum':max(l), 'minimum' : min(l),
                            'mean': mean(l), 'median' :median(l)}})
    
    return relative_gt, maxmin_dict
        #return distances

def df2slam_predict(gt):
    predict = gt[mean_cols]

    for std_col in std_cols:
        if std_col not in gt.columns:
            predict[std_col] = 1

    predict['to_index'] = gt['path_to_rgb_next'].apply(lambda x: int(Path(x).stem))
    predict['from_index'] = gt['path_to_rgb'].apply(lambda x: int(Path(x).stem))
    return predict

def get_trajectory_stat(path_to_csv, loop_threshold):
    trajectory_stat = dict()
    
    pair_frame_df = pd.read_csv(path_to_csv)
    if len(pair_frame_df.columns) < 6:
        return trajectory_stat
    
    pair_frame_df = df2slam_predict(pair_frame_df)    

    pair_frame_df = get_pair_frame_stat(pair_frame_df)
    
    is_adjustment = (pair_frame_df.to_index - pair_frame_df.from_index) <= 1
    adjustment_measurements = pair_frame_df[is_adjustment].reset_index(drop=True)
    
    is_loop = (pair_frame_df.to_index - pair_frame_df.from_index) >= loop_threshold
    loop_measurements = pair_frame_df[is_loop].reset_index(drop=True)
     
    trajectory_stat['matches_per_frame'] = len(pair_frame_df) / len(adjustment_measurements)
    
    if len(loop_measurements) == 0:
        trajectory_stat['matches_per_frame_in_loop_closure'] = None
        trajectory_stat['num_of_loops'] = 0
        return trajectory_stat
        
    num_of_frames = len(loop_measurements.to_index.unique())
    if num_of_frames > 0:
        trajectory_stat['matches_per_frame_in_loop_closure'] = len(loop_measurements) / num_of_frames
    
    
    num_of_loops = 0
    previous_index = loop_measurements['to_index'].values[0]
    for index, row in loop_measurements.iterrows():
        if row.to_index - previous_index > loop_threshold:
            num_of_loops += 1
        previous_index = row.to_index
        
    trajectory_stat['num_of_loops'] = num_of_loops
    
    return trajectory_stat

def average(dataset_stat):
    return dataset_stat

def get_histogram(df):
    return dict(x=[], y=[])


def get_dataset_stat(dataset_root):
    dataset_root = Path(dataset_root)
    stat = dict()
    for directory in dataset_root.iterdir():
        if not directory.is_dir():
            continue
        print(directory.as_posix())
        trajectory_stat = get_trajectory_stat(directory/'df.csv', 100)
        print(trajectory_stat)
        stat[directory.name] = trajectory_stat
