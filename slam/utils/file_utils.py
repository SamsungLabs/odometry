import os
import pandas as pd
from pathlib import Path


def chmod(path):
    mode = 0o755 if os.path.isdir(path) else 0o644
    os.chmod(path, mode)


def _create_file_path(save_dir, trajectory_id, ext, prediction_id='', subset=''):
    trajectory_name = trajectory_id.replace('/', '_')
    file_path = os.path.join(save_dir,
                             prediction_id,
                             subset,
                             trajectory_name + '.' + ext)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    chmod(os.path.dirname(file_path))
    return file_path


def create_vis_file_path(save_dir, trajectory_id, prediction_id='', subset='', sub_dir=''):
    return _create_file_path(save_dir=os.path.join(save_dir, 'visuals', sub_dir),
                             trajectory_id=trajectory_id,
                             prediction_id=prediction_id,
                             subset=subset,
                             ext='html')


def create_prediction_file_path(save_dir, trajectory_id, prediction_id='', subset=''):
    return _create_file_path(save_dir=os.path.join(save_dir, 'predictions'),
                             trajectory_id=trajectory_id,
                             prediction_id=prediction_id,
                             subset=subset,
                             ext='csv')


def read_csv(path):
    df = pd.read_csv(path)
    df.rename(columns={'path_to_rgb': 'from_path',
                       'path_to_rgb_next': 'to_path'},
              inplace=True)

    df['to_index'] = df['to_path'].apply(lambda x: int(Path(x).stem))
    df['from_index'] = df['from_path'].apply(lambda x: int(Path(x).stem))
    df['diff'] = df['to_index'] - df['from_index']

    mean_cols = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
    std_cols = [c + '_confidence' for c in mean_cols]

    df = pd.concat((df, pd.DataFrame(columns=std_cols)), axis=1)
    df.fillna(1., inplace=True)
    return df
