import os
import sys
from typing import Dict
from pathlib import Path

DATASET_TYPES = ['kitti_8/3',
                 'kitti_4/6',
                 'discoman_v10',
                 'discoman_debug',
                 'tum_debug',
                 'fr1',
                 'fr2',
                 'fr3',
                 'tum']


def get_config(dataset_root: str, dataset_type: str) -> Dict:

    assert dataset_type in DATASET_TYPES

    this_module = sys.modules[__name__]
    dataset_type = dataset_type.replace('/', '_')
    config = getattr(this_module, f'get_{dataset_type}_config')(dataset_root)

    return config


def append_root(config, dataset_root):
    for key in ['train_trajectories', 'val_trajectories', 'test_trajectories']:
        if config[key] is not None:
            config[key] = [os.path.join(dataset_root, i) for i in config[key]]
    return config


def get_kitti_8_3_config(dataset_root):
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

    config = append_root(config, dataset_root)
    return config


def get_kitti_4_6_config(dataset_root):
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
    config = append_root(config, dataset_root)
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


def get_discoman_debug_config(dataset_root):
    config = {'train_trajectories': ['train/000001'],
              'val_trajectories': ['val/000230'],
              'test_trajectories': ['test/000200'],
              'exp_name': 'discoman_debug',
              'target_size': (90, 160),
              }
    config = append_root(config, dataset_root)
    return config


def get_tum_debug_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg2_dishes'],
              'val_trajectories': ['rgbd_dataset_freiburg1_teddy'],
              'test_trajectories': ['rgbd_dataset_freiburg3_large_cabinet'],
              'exp_name': 'tum_debug',
              'target_size': (120, 160),
              }
    config = append_root(config, dataset_root)
    return config


def get_fr1_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg1_desk',
                                     'rgbd_dataset_freiburg1_xyz',
                                     'rgbd_dataset_freiburg1_360',
                                     'rgbd_dataset_freiburg1_rpy',
                                     'rgbd_dataset_freiburg1_teddy',
                                     'rgbd_dataset_freiburg1_floor',
                                     'rgbd_dataset_freiburg1_plant',
                                     ],
              'val_trajectories': ['rgbd_dataset_freiburg1_room'],
              'test_trajectories': ['rgbd_dataset_freiburg1_desk2'],
              'exp_name': 'fr1',
              'target_size': (120, 160),
              }
    config = append_root(config, dataset_root)
    return config


def get_fr2_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg2_rgb_calibration',
                                     'rgbd_dataset_freiburg2_large_checkerboard_calibration',
                                     'rgbd_dataset_freiburg2_large_no_loop',
                                     'rgbd_dataset_freiburg2_xyz',
                                     'rgbd_dataset_freiburg2_rpy',
                                     'rgbd_dataset_freiburg2_360_hemisphere',
                                     'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                                     'rgbd_dataset_freiburg2_coke',
                                     'rgbd_dataset_freiburg2_metallic_sphere',
                                     'rgbd_dataset_freiburg2_flowerbouquet',
                                     'rgbd_dataset_freiburg2_large_with_loop',
                                     'rgbd_dataset_freiburg2_metallic_sphere2',
                                     'rgbd_dataset_freiburg2_dishes',
                                     'rgbd_dataset_freiburg2_pioneer_slam',
                                     'rgbd_dataset_freiburg2_pioneer_360',
                                     ],
              'val_trajectories': ['rgbd_dataset_freiburg2_pioneer_slam2',
                                   'rgbd_dataset_freiburg2_desk'
                                   ],
              'test_trajectories': ['rgbd_dataset_freiburg2_desk_with_person',
                                    'rgbd_dataset_freiburg2_pioneer_slam3'],
              'exp_name': 'fr2',
              'target_size': (120, 160),
              }
    config = append_root(config, dataset_root)
    return config


def get_fr3_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg3_calibration_rgb_depth',
                                     'rgbd_dataset_freiburg3_checkerboard_large',
                                     'rgbd_dataset_freiburg3_sitting_xyz',
                                     'rgbd_dataset_freiburg3_long_office_household',
                                     'rgbd_dataset_freiburg3_walking_xyz',
                                     'rgbd_dataset_freiburg3_walking_static',
                                     'rgbd_dataset_freiburg3_nostructure_notexture_far',
                                     'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                                     'rgbd_dataset_freiburg3_structure_notexture_far',
                                     'rgbd_dataset_freiburg3_walking_halfsphere',
                                     'rgbd_dataset_freiburg3_large_cabinet',
                                     'rgbd_dataset_freiburg3_structure_texture_near',
                                     'rgbd_dataset_freiburg3_sitting_halfsphere',
                                     'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
                                     'rgbd_dataset_freiburg3_nostructure_texture_far',
                                     'rgbd_dataset_freiburg3_sitting_static',
                                     'rgbd_dataset_freiburg3_structure_texture_far',
                                     'rgbd_dataset_freiburg3_walking_rpy',
                                     'rgbd_dataset_freiburg3_cabinet',
                                     'rgbd_dataset_freiburg3_structure_notexture_near',
                                     'rgbd_dataset_freiburg3_teddy',
                                     ],
              'val_trajectories': ['rgbd_dataset_freiburg2_pioneer_slam2',
                                   'rgbd_dataset_freiburg2_desk'
                                   ],
              'test_trajectories': ['rgbd_dataset_freiburg2_desk_with_person',
                                    'rgbd_dataset_freiburg2_pioneer_slam3'],
              'exp_name': 'fr3',
              'target_size': (120, 160),
              }
    config = append_root(config, dataset_root)
    return config


def get_tum_config(dataset_root):
    fr1_config = get_fr1_config(dataset_root)
    fr2_config = get_fr2_config(dataset_root)
    fr3_config = get_fr3_config(dataset_root)

    config = fr1_config
    config['exp_name'] = 'tum'

    subsets = ['train', 'val', 'test']
    for subset in subsets:
        config[f'{subset}_trajectories'].append(fr2_config[f'{subset}_trajectories'])
        config[f'{subset}_trajectories'].append(fr3_config[f'{subset}_trajectories'])
    return config

