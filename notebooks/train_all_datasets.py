import argparse
import subprocess
import os
from typing import List

import __init_path__
import env

from odometry.preprocessing.dataset_configs import LEADER_BOARDS
from odometry.utils.utils import str2bool


def get_lsf_command(dataset_type: str, arguments: argparse.Namespace) -> List[str]:

    command = ['bsub',
               '-o ~/lsf/%J',
               '-gpu "num=1:mode:shared"',
               'python',
               f'{os.path.join(env.PROJECT_PATH, "notebooks/train.py")}',
               f'--dataset_root {arguments.dataset_root}',
               f'--dataset_type {dataset_type}',
               f'--run_name {arguments.run_name}'
               f'--prediction_dir {arguments.prediction_dir}',
               f'--visuals_dir {arguments.visuals_dir}',
               f'--period {arguments.period}',
               f'--save_best_only {arguments.save_best_only}'
               ]
    return command


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    pa
    parser.add_argument('--dataset_root', '-r', type=str, help='Directory with trajectories', required=True)
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

    subprocess.run(['mdkir',  '~/lsf'])

    for d_type in LEADER_BOARDS:
        cmd = get_lsf_command(d_type, args)
        subprocess.run(cmd, shell=True, check=True)
