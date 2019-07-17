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
                 'tum',
                 'zju',
                 'euroc']


def is_int(string: str):
    try:
        int(string)
        return True
    except:
        return False


def get_config(dataset_root: str, dataset_type: str) -> Dict:

    assert dataset_type in DATASET_TYPES

    this_module = sys.modules[__name__]
    dataset_type = dataset_type.replace('/', '_')
    config = getattr(this_module, f'get_{dataset_type}_config')(dataset_root)

    return config


def get_zju_config(dataset_root):
    config = {'train_trajectories': ['A0',
                                     'A3',
                                     'A4',
                                     'A5',
                                     'B0',
                                     'B2'
                                     ],
              'val_trajectories': ['A1',
                                   'A6',
                                   'B1'
                                   ],
              'test_trajectories': ['A2',
                                    'A7',
                                    'B3'
                                    ],
              'exp_name': 'zju',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_euroc_config(dataset_root):
    config = {'train_trajectories': ['MH_01_easy',
                                     'MH_03_medium',
                                     'MH_04_difficult',
                                     'V1_01_easy',
                                     'V1_03_difficult',
                                     'V2_01_easy',
                                     'V2_03_difficult'],
              'val_trajectories': ['MH_02_easy',
                                   'V1_02_medium'],
              'test_trajectories': ['MH_05_difficult',
                                    'V2_02_medium'],
              'exp_name': 'euroc',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_kitti_8_3_config(dataset_root):
    config = {'train_trajectories': ['00',
                                     '01',
                                     '02',
                                     '03',
                                     '04',
                                     '05',
                                     '06',
                                     '07'],
              'val_trajectories': ['08',
                                   '09',
                                   '10'],
              'test_trajectories': None,
              'exp_name': 'kitti_8/3',
              'target_size': (96, 320),
              }
    return config


def get_kitti_4_6_config(dataset_root):
    config = {'train_trajectories': ['00',
                                     '02',
                                     '08',
                                     '09'],
              'val_trajectories': ['03',
                                   '04',
                                   '05',
                                   '06',
                                   '07',
                                   '10'],
              'test_trajectories': None,
              'exp_name': 'kitti_4/6',
              'target_size': (96, 320),
              'rpe_indices': 'kitti',
             }
    return config


def get_discoman_v10_config(dataset_root):
    config = {'train_trajectories': list(),
              'val_trajectories': list(),
              'test_trajectories': list(),
              'exp_name': 'discoman_v10',
              'target_size': (90, 160),
              'rpe_indices': 'full',
             }

    sub_dirs = ['train', 'val', 'test']
    for d in sub_dirs:
        for trajectory in Path(dataset_root).joinpath(d).glob('*'):
            if is_int(trajectory.name):
                trajectory_name = trajectory.as_posix()[len(dataset_root) + int(dataset_root[-1] != '/'):]
                config[f'{d}_trajectories'].append(trajectory_name)

    return config


def get_discoman_debug_config(dataset_root):
    config = {'train_trajectories': ['train/000001'],
              'val_trajectories': ['val/000230'],
              'test_trajectories': ['test/000200'],
              'exp_name': 'discoman_debug',
              'target_size': (90, 160),
              'rpe_indices': 'full',
              }
    return config


def get_tum_debug_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'val_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'test_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'exp_name': 'tum_debug',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_fr1_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg1_desk',
                                     'rgbd_dataset_freiburg1_xyz',
                                     'rgbd_dataset_freiburg1_360',
                                     'rgbd_dataset_freiburg1_rpy',
                                     'rgbd_dataset_freiburg1_teddy',
                                     'rgbd_dataset_freiburg1_plant'],
              'val_trajectories': ['rgbd_dataset_freiburg1_room'],
              'test_trajectories': ['rgbd_dataset_freiburg1_desk2'],
              'exp_name': 'fr1',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_fr2_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg2_xyz',
                                     'rgbd_dataset_freiburg2_rpy',
                                     'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                                     'rgbd_dataset_freiburg2_coke',
                                     'rgbd_dataset_freiburg2_metallic_sphere',
                                     'rgbd_dataset_freiburg2_metallic_sphere2',
                                     'rgbd_dataset_freiburg2_dishes'],
              'val_trajectories': ['rgbd_dataset_freiburg2_flowerbouquet'],
              'test_trajectories': ['rgbd_dataset_freiburg2_pioneer_slam3',
                                    'rgbd_dataset_freiburg2_360_hemisphere'],
              'exp_name': 'fr2',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_fr3_config(dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg3_checkerboard_large',
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
                                     'rgbd_dataset_freiburg3_teddy'],
              'val_trajectories': ['rgbd_dataset_freiburg3_sitting_xyz_validation',
                                   'rgbd_dataset_freiburg3_walking_xyz_validation',
                                   'rgbd_dataset_freiburg3_walking_static_validation',
                                   'rgbd_dataset_freiburg3_nostructure_notexture_far_validation',
                                   'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop_validation',
                                   'rgbd_dataset_freiburg3_structure_notexture_far_validation',
                                   'rgbd_dataset_freiburg3_large_cabinet_validation',
                                   'rgbd_dataset_freiburg3_structure_texture_near_validation',
                                   'rgbd_dataset_freiburg3_nostructure_texture_near_withloop_validation',
                                   'rgbd_dataset_freiburg3_sitting_static_validation',
                                   'rgbd_dataset_freiburg3_walking_rpy_validation',
                                   'rgbd_dataset_freiburg3_cabinet_validation',
                                   'rgbd_dataset_freiburg3_structure_notexture_near_validation'],
              'test_trajectories': ['rgbd_dataset_freiburg3_structure_texture_far_validation',
                                    'rgbd_dataset_freiburg3_long_office_household_validation',
                                    'rgbd_dataset_freiburg3_sitting_halfsphere_validation',
                                    'rgbd_dataset_freiburg3_nostructure_texture_far_validation',
                                    'rgbd_dataset_freiburg3_walking_halfsphere_validation'],
              'exp_name': 'fr3',
              'target_size': (120, 160),
              'rpe_indices': 'full',
              }
    return config


def get_tum_config(dataset_root):
    fr1_config = get_fr1_config(dataset_root)
    fr2_config = get_fr2_config(dataset_root)
    fr3_config = get_fr3_config(dataset_root)

    config = fr1_config
    config['exp_name'] = 'tum'

    subsets = ['train', 'val', 'test']
    for subset in subsets:
        config[f'{subset}_trajectories'].extend(fr2_config[f'{subset}_trajectories'])
        config[f'{subset}_trajectories'].extend(fr3_config[f'{subset}_trajectories'])
    return config
