import os
import pandas as pd
from pathlib import Path
from typing import Dict

from slam.utils import is_int
from slam.utils import read_csv


from slam.linalg import RelativeTrajectory


def get_gt_trajectory(dataset_root, trajectory_name):
    gt_df = pd.read_csv(os.path.join(dataset_root, trajectory_name, 'df.csv'))
    gt_trajectory = RelativeTrajectory.from_dataframe(gt_df).to_global()
    return gt_trajectory


def get_predicted_df(multistride_paths: Dict[str]):
    df_list = list()
    for stride, monostride_paths in multistride_paths.items():
        for path in monostride_paths:
            df = read_csv(path)
            if stride == 'loops':
                df = df[df['diff'] > 49].reset_index()
            df_list.append(df)
    predicted_df = pd.concat(df_list, ignore_index=True)
    return predicted_df


def get_group_id(multistride_paths):
    parent_dir = os.path.basename(os.path.dirname(multistride_paths['1'][0]))
    if parent_dir == 'val':
        group_id = 0
    elif parent_dir == 'test':
        group_id = 1
    else:
        raise RuntimeError(
            f'Unexpected parent dir of prediction {multistride_paths["1"][0]}. Parent dir must "val" or "test"')
    return group_id


def get_trajectory_names(prefix):
    val_dirs = list(Path(prefix).glob(f'*val*'))
    paths = [val_dir.as_posix() for val_dir in val_dirs]
    try:
        last_dir = max(paths, key=lambda x: get_epoch_from_dirname(x))
    except Exception as e:
        raise RuntimeError(f'Could not find val directories in paths: {paths}', e)
    val_trajectory_names = Path(last_dir).joinpath('val').glob('*.csv')
    test_trajectory_names = Path(prefix).joinpath('test/test').glob('*.csv')
    trajectory_names = list(val_trajectory_names) + list(test_trajectory_names)
    trajectory_names = [trajectory_name.stem for trajectory_name in trajectory_names]
    # Handaling bug with strides in names of trajectory
    handled_trajectory_names = list()
    for trajectory_name in trajectory_names:
        split = trajectory_name.split('_')
        if len(split) > 1 and is_int(split[0]):
            trajectory_name = '_'.join(split[1:])
        handled_trajectory_names.append(trajectory_name)
    assert len(trajectory_names) > 0
    return handled_trajectory_names


def get_epoch_from_dirname(dirname):
    position = dirname.find('_val_RPE')
    if position == -1:
        raise RuntimeError(f'Could not find epoch number in {dirname}')
    return int(dirname[position - 3: position])


def get_path(prefix, trajectory_name):
    paths = list(Path(prefix).rglob(f'*{trajectory_name}.csv'))

    if len(paths) == 1:
        return paths[0].as_posix()
    elif len(paths) > 1:
        return max(paths, key=lambda x: get_epoch_from_dirname(x.parent.parent.name)).as_posix()
    else:
        raise RuntimeError(f'Could not find trajectory {trajectory_name} in dir {prefix}')
