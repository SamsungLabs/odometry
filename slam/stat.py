import pandas as pd
from pathlib import Path

mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
std_cols = [c + '_confidence' for c in mean_cols]

def get_pair_frame_stat(df):
    return df

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
