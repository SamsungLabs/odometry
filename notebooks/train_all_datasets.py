import argparse
import subprocess
import os
from typing import List
from pathlib import Path

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_lsf_command(dataset_type: str, arguments: argparse.Namespace) -> List[str]:

    if dataset_type == 'discoman_v10':
        dataset_root = '/dbstore/datasets/Odometry_team/discoman_v10/'
    elif dataset_type == 'kitti_4/6':
        dataset_root = '/dbstore/datasets/Odometry_team/KITTI_odometry_2012/'
    elif dataset_type == 'tum_debug':
        dataset_root = '/dbstore/datasets/Odometry_team/tum_rgbd/'
    else:
        raise RuntimeError('Unknown dataset_type')

    command = ['bsub',
               f'-o {Path.home().joinpath("lsf").joinpath("%J").as_posix()}',
               '-gpu "num=1:mode=shared"',
               'python',
               f'{os.path.join(os.path.dirname(os.path.realpath(__file__)), "train.py")}',
               f'--dataset_root {dataset_root}',
               f'--dataset_type {dataset_type}',
               f'--run_name {arguments.run_name}',
               f'--prediction_dir {arguments.prediction_dir}',
               f'--visuals_dir {arguments.visuals_dir}',
               f'--period {arguments.period}',
               f'--save_best_only {arguments.save_best_only}'
               ]
    return command


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Must be unique and specific',
                        required=True)
    parser.add_argument('--prediction_dir', '-p', type=str, help='Name of subdir to store predictions',
                        default='predictions')
    parser.add_argument('--visuals_dir', '-v', type=str, help='Name of subdir to store visualizations',
                        default='visuals')
    parser.add_argument('--period', type=str, help='Period of evaluating train and val metrics',
                        default=1)
    parser.add_argument('--save_best_only', type=str2bool, help='Evaluate metrics only for best losses',
                        default=False)
    args = parser.parse_args()

    # subprocess.run(['mkdir',  Path().home().joinpath('lsf').as_posix()])

    leader_boards = ['kitti_4/6', 'discoman_v10', 'tum_debug']

    for d_type in leader_boards:
        cmd = get_lsf_command(d_type, args)
        subprocess.run(' '.join(cmd), shell=True, check=True)
