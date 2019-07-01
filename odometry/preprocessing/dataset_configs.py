import os
from pathlib import Path

DATASET_TYPES = ['kitti_1', 'kitti_2', 'discoman_v10', 'discoman_debug']
def append_root(config, dataset_root):
    for k in ['train_trajectories', 'val_trajectories', 'test_trajectories']:
        if config[k] is not None:
            config[k] = [os.path.join(dataset_root, i) for i in config[k]]
    return config


def get_config(dataset_root, dataset_type):

    if dataset_type == 'kitti_1':
        config = get_kitti_1_config()
        config = append_root(config, dataset_root)
    elif dataset_type == 'kitti_2':
        config = get_kitti_2_config()
        config = append_root(config, dataset_root)
    elif dataset_type == 'discoman_v10':
        config = get_discoman_v10_config(dataset_root)
    elif dataset_type == 'discoman_debug':
        config = get_discoman_debug_config()
        config = append_root(config, dataset_root)
    else:
        raise RuntimeError('Unexpected name of config for training')
    return config


def get_kitti_1_config():
    config = {'train_trajectories': ['00',
                                     '01',
                                     '02',
                                     '03',
                                     '04',
                                     '05',
                                     '06',
                                     '07'
                                     ],

              'val_trajectories': ['08',
                                   '09',
                                   '10'
                                   ],

              'test_trajectories': None,
              'exp_name': 'kitti_1',
              'target_size': (96, 320),
              }
    return config


def get_kitti_2_config():
    config = {'train_trajectories': ['00',
                                     '02',
                                     '08',
                                     '09',
                                     ],
              'val_trajectories': ['03',
                                   '04',
                                   '05',
                                   '06',
                                   '07',
                                   '10'
                                   ],
              'test_trajectories': None,
              'exp_name': 'kitti_2',
              'target_size': (96, 320)
              }
    return config


def get_discoman_v10_config(dataset_root):
    config = {
        'train_trajectories': list(),
        'val_trajectories': list(),
        'test_trajectories': list(),
        'exp_name': 'discoman_v10',
        'target_size': (90, 160),
         }

    for trajectory in Path(dataset_root).joinpath('train').glob('*'):
        config['train_trajectories'].append(trajectory.as_posix())

    for trajectory in Path(dataset_root).joinpath('val').glob('*'):
        config['val_trajectories'].append(trajectory.as_posix())

    for trajectory in Path(dataset_root).joinpath('test').glob('*'):
        config['test_trajectories'].append(trajectory.as_posix())

    return config


def get_discoman_debug_config():
    config = {
        'train_trajectories': ['train/000001'],
        'val_trajectories': ['val/000230'],
        'test_trajectories': ['test/000200'],
        'exp_name': 'discoman_debug',
        'target_size': (90, 160),
         }
    return config


def get_tum_config():
    config = {'id': 3}
    return config
