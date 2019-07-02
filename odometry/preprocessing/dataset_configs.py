import os
from pathlib import Path

DATASET_TYPES = ['kitti_8/3', 'kitti_4/6', 'discoman_v10', 'discoman_debug', 'tum_debug']


def append_root(config, dataset_root):
    for key in ['train_trajectories', 'val_trajectories', 'test_trajectories']:
        if config[key] is not None:
            config[key] = [os.path.join(dataset_root, i) for i in config[key]]
    return config


def get_config(dataset_root, dataset_type):

    if dataset_type == 'kitti_8/3':
        config = get_kitti_8_3_config()
        config = append_root(config, dataset_root)
    elif dataset_type == 'kitti_4/6':
        config = get_kitti_4_6_config()
        config = append_root(config, dataset_root)
    elif dataset_type == 'discoman_v10':
        config = get_discoman_v10_config(dataset_root)
    elif dataset_type == 'discoman_debug':
        config = get_discoman_debug_config()
        config = append_root(config, dataset_root)
    elif dataset_type == 'tum_debug':
        config = get_tum_debug_config()
        config = append_root(config, dataset_root)
    else:
        raise RuntimeError(f'Unexpected name of config for training. Got {dataset_type}.')
    return config


def get_kitti_8_3_config():
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
              'exp_name': 'kitti_8/3',
              'target_size': (96, 320),
              }
    return config


def get_kitti_4_6_config():
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
              'exp_name': 'kitti_4/6',
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

    sub_dirs = ['train', 'val', 'test']
    for d in sub_dirs:
        for trajectory in Path(dataset_root).joinpath(d).glob('*'):
            config[f'{d}_trajectories'].append(trajectory.as_posix())

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


def get_tum_debug_config():
    config = {
        'train_trajectories': ['rgbd_dataset_freiburg2_dishes'],
        'val_trajectories': ['rgbd_dataset_freiburg1_teddy'],
        'test_trajectories': ['rgbd_dataset_freiburg3_large_cabinet'],
        'exp_name': 'tum_debug',
        'target_size': (120, 160),
         }
    return config


def get_tum_config():
    config = {'id': 3}
    return config
