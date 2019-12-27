import os
import time
import datetime
import logging
import argparse
import subprocess as sp
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from typing import Union

import __init_path__
import env

from scripts.leaderboard.average_metrics import MetricAverager


class Leaderboard:

    def __init__(self,
                 script_path,
                 leader_board,
                 load_leader_board,
                 bundle_name,
                 load_bundle_name,
                 bundle_size=1,
                 verbose=False,
                 stride=None,
                 other_args=None):

        if not os.path.exists(script_path):
            raise RuntimeError(f'Could not find trainer script {script_path}')

        self.averager = MetricAverager()

        self.script_path = script_path
        self.leader_board = leader_board
        self.load_leader_board = load_leader_board
        self.bundle_name = bundle_name
        self.load_bundle_name = load_bundle_name
        self.bundle_size = bundle_size

        self.leader_boards = ['kitti_4/6',
                              'kitti_8/3',
                              'discoman_v10',
                              'tum',
                              'euroc',
                              'zju']

        self.verbose = verbose
        self.stride = stride
        self.other_args = other_args or []

    def submit(self):

        if self.leader_board == 'leaderboard':
            self.submit_on_all_leader_boards()
        else:
            self.submit_bundle(self.leader_board)

    def submit_on_all_leader_boards(self):

        pool = Pool(len(self.leader_boards))
        for leader_board in self.leader_boards:
            self.log(f'submitting {self.bundle_name}', leader_board)
            pool.apply_async(self.submit_bundle, (leader_board,))
        pool.close()
        pool.join()

    def submit_bundle(self, leader_board):

        self.setup_logger(leader_board)

        self.log('Submitting jobs', leader_board)

        started_jobs_id = set()
        for b in range(self.bundle_size):
            job_id = self.submit_job(leader_board, b)
            started_jobs_id.add(job_id)

        self.log(f'Started {started_jobs_id}', leader_board)
        self.wait_jobs(leader_board, started_jobs_id)

        self.log('Averaging metrics', leader_board)

        try:
            self.averager.average_bundle(bundle_name=self.bundle_name, leader_board=leader_board)
        except Exception as e:
            self.log(e)

    def submit_job(self, leader_board, bundle_id):

        run_name = f'{self.bundle_name}_b_{bundle_id}'
        load_name = f'{self.load_bundle_name}_b_{bundle_id}' if self.load_bundle_name else None

        seed = np.random.randint(1000000)
        cmd = self.get_lsf_command(leader_board, run_name, load_name, seed)
        self.log(f'Executing command: {cmd}')

        p = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
        outs, errs = p.communicate(timeout=4)

        job_id = str(outs).split(' ')[1][1:-1]
        return job_id

    def get_lsf_command(self,
                        leader_board: str,
                        run_name: str,
                        load_name: Union[str, None],
                        seed: int) -> str:

        command = ['python',
                   f'{self.script_path}',
                   f'--leader_board {leader_board}',
                   f'--run_name {run_name}',
                   f'--bundle_name {self.bundle_name}',
                   f'--seed {seed}']

        if self.stride:
            command.append(f'--stride {self.stride}')

        if load_name:
            command.extend([f'--load_name {load_name}',
                            f'--load_bundle_name {self.load_bundle_name}',
                            f'--load_leader_board {self.load_leader_board}'])

        return ' '.join(command + other_args)

    def wait_jobs(self, leader_board, started_jobs_id):

        finished = False
        while not finished:

            p = sp.Popen(['bjobs'], shell=True, stdout=sp.PIPE)
            outs, errs = p.communicate()
            outs = outs.decode('utf-8').split('\n')
            job_ids = {outs[i].split(' ')[0] for i in range(1, len(outs) - 1)}

            still_running_jobs = started_jobs_id.intersection(job_ids)
            sorted_jobs = list(still_running_jobs)
            sorted_jobs.sort()

            self.log(f'Running {sorted_jobs}', leader_board)

            if still_running_jobs:
                time.sleep(10)
            else:
                finished = True
                self.log('All jobs has been finished', leader_board)

    def setup_logger(self, leader_board):

        logger = logging.getLogger('leaderboard')
        logger.setLevel(logging.DEBUG)

        log_dir = os.path.join(env.PROJECT_PATH, 'logs', leader_board.replace('/', '_'))
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, self.bundle_name + '.txt'), mode='w+')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        if self.verbose:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            logger.addHandler(sh)

    def log(self, info, leader_board=None):

        logger = logging.getLogger('leaderboard')

        timestamp = datetime.datetime.now().isoformat().replace('T', ' ')

        if leader_board:
            logger.info(f'{timestamp} {leader_board}:{self.bundle_name}. {info}')
        else:
            logger.info(f'{timestamp} {info}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_path', type=str, required=True)

    parser.add_argument('--leader_board', '--dataset_type', type=str, required=True,
                        help='You can find available experiment names in slam.preprocessing.dataset_configs.py')
    parser.add_argument('--load_leader_board', '--load_dataset_type', type=str, default=None)

    parser.add_argument('--bundle_name', '-n', type=str, required=True,
                        help='Name of the bundle. Must be unique and specific')
    parser.add_argument('--load_bundle_name', '-ln', type=str, default=None,
                        help='Name of the loaded bundle')

    parser.add_argument('--bundle_size', '-b', type=int, required=True, help='Number runs in evaluate')

    parser.add_argument('--verbose', '-v', action='store_true', help='Print output to console')
    parser.add_argument('--stride', type=int, default=None, help='Stride between frames in dataset')

    args, other_args = parser.parse_known_args()

    leaderboard = Leaderboard(script_path=args.script_path,
                              leader_board=args.leader_board,
                              load_leader_board=args.load_leader_board,
                              bundle_name=args.bundle_name,
                              load_bundle_name=args.load_bundle_name,
                              bundle_size=args.bundle_size,
                              verbose=args.verbose,
                              stride=args.stride,
                              other_args=other_args)

    leaderboard.submit()
