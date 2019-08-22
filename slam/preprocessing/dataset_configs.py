import os
import sys
from typing import Dict
from pathlib import Path

DATASET_TYPES = ['kitti_8/3',
                 'kitti_4/6',
                 'kitti_4/6_mixed',
                 'discoman_v10',
                 'discoman_v10_mixed',
                 'mini_discoman_v10',
                 'mini_discoman_v10_mixed',
                 'discoman_debug',
                 'tum_debug',
                 'tum_fr1',
                 'tum_fr2',
                 'tum_fr3',
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


def get_zju_config(_dataset_root):
    config = {'train_trajectories': ['A0',
                                     'A3',
                                     'A4',
                                     'A5',
                                     'B0',
                                     'B2'],
              'val_trajectories': ['A1',
                                   'A6',
                                   'B1'],
              'test_trajectories': ['A2',
                                    'A7',
                                    'B3'],
              'exp_name': 'zju',
              'target_size': (120, 160),
              'source_size': (480, 640),
              'depth_multiplicator': 1.0,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_euroc_config(_dataset_root):
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
              'source_size': (480, 640),
              'depth_multiplicator': 1.0,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_kitti_8_3_config(_dataset_root):
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
              'source_size': (384, 1280),
              'depth_multiplicator': 1.0,
              'rpe_indices': 'kitti',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_kitti_4_6_config(_dataset_root):
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
              'source_size': (384, 1280),
              'depth_multiplicator': 1.0,
              'rpe_indices': 'kitti',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_kitti_4_6_mixed_config(_dataset_root):
    config = {'train_trajectories': ['1/00',
                                     '1/02',
                                     '1/08',
                                     '1/09',
                                     '2/00',
                                     '2/02',
                                     '2/08',
                                     '2/09'],
              'val_trajectories': ['1/03',
                                   '1/04',
                                   '1/05',
                                   '1/06',
                                   '1/07',
                                   '1/10'],
              'test_trajectories': None,
              'exp_name': 'kitti_4/6_mixed',
              'target_size': (96, 320),
              'source_size': (384, 1280),
              'depth_multiplicator': 1.0,
              'rpe_indices': 'kitti',
              }

    sub_dirs = ['train', 'val', 'test']
    for d in sub_dirs:
        trajectories = config[f'{d}_trajectories']
        config[f'{d}_strides'] = [int(Path(t).parent.name) for t in trajectories] if trajectories else None

    return config


def get_discoman_v10_config(dataset_root):
    config = {'train_trajectories': list(),
              'val_trajectories': list(),
              'test_trajectories': list(),
              'exp_name': 'discoman_v10',
              'target_size': (90, 160),
              'source_size': (360, 640),
              'depth_multiplicator': 1.0 / 1000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }

    sub_dirs = ['train', 'val', 'test']
    for d in sub_dirs:
        for trajectory in Path(dataset_root).joinpath(d).glob('*'):
            if is_int(trajectory.name):
                trajectory_name = trajectory.relative_to(dataset_root).as_posix()
                config[f'{d}_trajectories'].append(trajectory_name)

    return config


def get_mini_discoman_v10_config(dataset_root):
    config = {'train_trajectories': list(),
              'val_trajectories': list(),
              'test_trajectories': list(),
              'exp_name': 'mini_discoman_v10',
              'target_size': (90, 160),
              'source_size': (360, 640),
              'depth_multiplicator': 1.0 / 1000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }

    sub_size = {'train': 33, 'val': 15, 'test': 8}  # as in TUM
    sub_dirs = ['train', 'val', 'test']

    for d in sub_dirs:
        for trajectory in Path(dataset_root).joinpath(d).glob('*'):
            if len(config[f'{d}_trajectories']) == sub_size[d]:
                break
            if is_int(trajectory.name):
                trajectory_name = trajectory.relative_to(dataset_root).as_posix()
                config[f'{d}_trajectories'].append(trajectory_name)

    return config


def get_discoman_v10_mixed_config(dataset_root):
    config = {'train_trajectories': list(),
              'val_trajectories': list(),
              'test_trajectories': list(),
              'exp_name': 'discoman_v10_mixed',
              'target_size': (90, 160),
              'source_size': (360, 640),
              'depth_multiplicator': 1.0 / 1000,
              'rpe_indices': 'full',
              'train_strides': list(),
              'val_strides': list(),
              'test_strides': list(),
              }

    sub_dirs = ['train', 'val', 'test']

    for stride_dir in Path(dataset_root).iterdir():
        stride = int(stride_dir.name)

        for d in sub_dirs:
            if d != 'train' and stride != 1:
                continue

            for trajectory in stride_dir.joinpath(d).glob('*'):
                if is_int(trajectory.name):
                    trajectory_name = trajectory.relative_to(dataset_root).as_posix()
                    config[f'{d}_trajectories'].append(trajectory_name)
                    config[f'{d}_strides'].append(stride)
    return config


def get_mini_discoman_v10_mixed_config(dataset_root):
    config = {'train_trajectories': list(),
              'val_trajectories': list(),
              'test_trajectories': list(),
              'exp_name': 'mini_discoman_v10_mixed',
              'target_size': (90, 160),
              'source_size': (360, 640),
              'depth_multiplicator': 1.0 / 1000,
              'rpe_indices': 'full',
              'train_strides': list(),
              'val_strides': list(),
              'test_strides': list(),
              }

    sub_size = {'train': 33, 'val': 15, 'test': 8} # as in TUM
    sub_dirs = ['train', 'val', 'test']
    for stride_dir in Path(dataset_root).iterdir():
        stride = int(stride_dir.name)

        for d in sub_dirs:
            if d in ('val', 'test') and stride != 1:
                continue

            for trajectory in stride_dir.joinpath(d).glob('*'):
                if len(config[f'{d}_trajectories']) == sub_size[d]:
                    break
                if is_int(trajectory.name):
                    trajectory_name = trajectory.relative_to(dataset_root).as_posix()
                    config[f'{d}_trajectories'].append(trajectory_name)
                    config[f'{d}_strides'].append(stride)
    return config


def get_discoman_debug_config(_dataset_root):
    config = {'train_trajectories': ['train/000001'],
              'val_trajectories': ['val/000230'],
              'test_trajectories': ['test/000200'],
              'exp_name': 'discoman_debug',
              'target_size': (90, 160),
              'source_size': (360, 640),
              'depth_multiplicator': 1.0 / 1000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_tum_debug_config(_dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'val_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'test_trajectories': ['rgbd_dataset_freiburg1_desk'],
              'exp_name': 'tum_debug',
              'target_size': (120, 160),
              'source_size': (480, 640),
              'depth_multiplicator': 1.0 / 5000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_tum_fr1_config(_dataset_root):
    config = {'train_trajectories': ['rgbd_dataset_freiburg1_desk',
                                     'rgbd_dataset_freiburg1_xyz',
                                     'rgbd_dataset_freiburg1_360',
                                     'rgbd_dataset_freiburg1_rpy',
                                     'rgbd_dataset_freiburg1_teddy',
                                     'rgbd_dataset_freiburg1_plant'],
              'val_trajectories': ['rgbd_dataset_freiburg1_room'],
              'test_trajectories': ['rgbd_dataset_freiburg1_desk2'],
              'exp_name': 'tum_fr1',
              'target_size': (120, 160),
              'source_size': (480, 640),
              'depth_multiplicator': 1.0 / 5000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_tum_fr2_config(_dataset_root):
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
              'exp_name': 'tum_fr2',
              'target_size': (120, 160),
              'source_size': (480, 640),
              'depth_multiplicator': 1.0 / 5000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_tum_fr3_config(_dataset_root):
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
              'exp_name': 'tum_fr3',
              'target_size': (120, 160),
              'source_size': (480, 640),
              'depth_multiplicator': 1.0 / 5000,
              'rpe_indices': 'full',
              'train_strides': 1,
              'val_strides': 1,
              'test_strides': 1,
              }
    return config


def get_tum_config(dataset_root):
    fr1_config = get_tum_fr1_config(dataset_root)
    fr2_config = get_tum_fr2_config(dataset_root)
    fr3_config = get_tum_fr3_config(dataset_root)

    config = fr1_config
    config['exp_name'] = 'tum'

    subsets = ['train', 'val', 'test']
    for subset in subsets:
        config[f'{subset}_trajectories'].extend(fr2_config[f'{subset}_trajectories'])
        config[f'{subset}_trajectories'].extend(fr3_config[f'{subset}_trajectories'])
    return config
