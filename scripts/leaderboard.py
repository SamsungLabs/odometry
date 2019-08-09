import os
import time
import datetime
import logging
import argparse
import subprocess as sp
import numpy as np
from pathlib import Path
from multiprocessing import Pool

import __init_path__
import env

from scripts.average_metrics import average_metrics


class Leaderboard:

    def __init__(self,
                 trainer_path,
                 dataset_type,
                 run_name,
                 machines,
                 bundle_size=1,
                 core=8,
                 verbose=False,
                 debug=False,
                 shared=False,
                 gmem=None,
                 round_robin=0,
                 cache=False):

        if not os.path.exists(trainer_path):
            raise RuntimeError(f'Could not find trainer script {trainer_path}')

        self.trainer_path = trainer_path
        self.dataset_type = dataset_type
        self.run_name = run_name
        self.bundle_size = bundle_size
        self.core = core

        if debug:
            self.leader_boards = ['tum_debug', 'discoman_debug']
        else:
            self.leader_boards = ['kitti_4/6',
                                  'kitti_8/3',
                                  'discoman_v10',
                                  'tum',
                                  'saic_office',
                                  'retail_bot',
                                  'euroc',
                                  'zju']

        self.verbose = verbose
        self.machines = machines.split(' ')
        self.shared = shared
        self.gmem = gmem
        self.round_robin = min(len(self.machines), round_robin or len(self.machines))
        self.cache = cache

    def submit(self):

        if self.dataset_type == 'leaderboard':
            self.submit_on_all_datasets()
        else:
            self.submit_bundle(self.dataset_type)

    def submit_on_all_datasets(self):

        pool = Pool(len(self.leader_boards))
        for d_type in self.leader_boards:
            pool.apply_async(self.submit_bundle, (d_type, ))
        pool.close()
        pool.join()

    def submit_bundle(self, dataset_type):

        self.setup_logger(dataset_type)

        self.log('Started submitting jobs', dataset_type)

        started_jobs_id = set()
        for b in range(self.bundle_size):
            job_id = self.submit_job(dataset_type, b)
            started_jobs_id.add(job_id)

        self.log(f'Started started_jobs_id {started_jobs_id}', dataset_type)
        self.wait_jobs(dataset_type, started_jobs_id)

        self.log('Averaging metrics', dataset_type)
        try:
            average_metrics(self.run_name, dataset_type)
        except Exception as e:
            self.log(e)

    def submit_job(self, dataset_type, bundle_id):

        run_name = self.run_name + f'_b_{bundle_id}'

        machines = np.random.choice(self.machines, self.round_robin, replace=False)

        seed = np.random.randint(1000000)
        cmd = self.get_lsf_command(dataset_type, run_name, ' '.join(machines), seed)
        self.log(f'Running command: {cmd}')

        p = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
        outs, errs = p.communicate(timeout=4)

        job_id = str(outs).split(' ')[1][1:-1]
        return job_id

    def get_lsf_command(self, dataset_type: str, run_name: str, machines: str, seed: int) -> str:

        if dataset_type == 'discoman_v10':
            dataset_root = env.DISCOMAN_V10_PATH
        elif dataset_type == 'discoman_debug':
            dataset_root = env.DISCOMAN_V10_PATH
        elif dataset_type == 'kitti_4/6':
            dataset_root = env.KITTI_PATH
        elif dataset_type == 'kitti_8/3':
            dataset_root = env.KITTI_PATH
        elif dataset_type == 'tum':
            dataset_root = env.TUM_PATH
        elif dataset_type == 'tum_debug':
            dataset_root = env.TUM_PATH
        elif dataset_type == 'saic_office':
            dataset_root = env.SAIC_OFFICE_PATH
        elif dataset_type == 'retail_bot':
            dataset_root = env.RETAIL_BOT_PATH
        elif dataset_type == 'euroc':
            dataset_root = env.EUROC_PATH
        elif dataset_type == 'zju':
            dataset_root = env.ZJU_PATH
        else:
            raise RuntimeError('Unknown dataset_type')

        if self.shared:
            mode = 'shared' + (f':gmem={self.gmem}' if self.gmem else '') + ":gtile='!'"
        else:
            mode = 'exclusive_process'

        command = ['bsub',
                   f'-n 1 -R "span[hosts=1] affinity[core({self.core}):distribute=pack]"',
                   f'-o {Path.home().joinpath("lsf").joinpath("%J").as_posix()}',
                   f'-m "{machines}"',
                   f'-gpu "num=1:mode={mode}"',
                   'python',
                   f'{self.trainer_path}',
                   f'--dataset_root {dataset_root}',
                   f'--dataset_type {dataset_type}',
                   f'--run_name {run_name}',
                   f'--seed {seed}']
        if self.cache:
            command.append('--cache')
        return ' '.join(command)

    def wait_jobs(self, dataset_type, started_jobs_id):

        finished = False
        while not finished:

            p = sp.Popen(['bjobs'], shell=True, stdout=sp.PIPE)
            outs, errs = p.communicate()
            outs = outs.decode('utf-8').split('\n')
            job_ids = {outs[i].split(' ')[0] for i in range(1, len(outs) - 1)}

            still_running_jobs = started_jobs_id.intersection(job_ids)
            sorted_jobs = list(still_running_jobs)
            sorted_jobs.sort()
            self.log(f'Jobs {sorted_jobs} are still running', dataset_type)

            if still_running_jobs:
                time.sleep(10)
            else:
                finished = True
                self.log('All jobs has been finished', dataset_type)

    def setup_logger(self, dataset_type):

        logger = logging.getLogger('leaderboard')
        logger.setLevel(logging.DEBUG)

        dataset_type = dataset_type.replace('/', '_')
        fh = logging.FileHandler(os.path.join(env.PROJECT_PATH, f'log_leaderboard_{dataset_type}.txt'), mode='w+')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        if self.verbose:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            logger.addHandler(sh)

    @staticmethod
    def log(info, dataset_type=None):

        logger = logging.getLogger('leaderboard')

        timestamp = datetime.datetime.now().isoformat().replace('T', ' ')

        if dataset_type:
            logger.info(f'{timestamp} Dataset {dataset_type}. {info}')
        else:
            logger.info(f'{timestamp} {info}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainer_path', type=str, required=True)
    parser.add_argument('--dataset_type', '-t', type=str, required=True,
                        help='You can find availible exp names in slam.preprocessing.dataset_configs.py')

    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Must be unique and specific',
                        required=True)
    parser.add_argument('--bundle_size', '-b', type=int, help='Number runs in evaluate', required=True)
    parser.add_argument('--core', '-c', type=int, help='Number of cpu core', default=8)

    parser.add_argument('--verbose', '-v', action='store_true', help='Print output to console')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--machines', '-m', help='lsf arg. Specify machines on which execute job',
                        default='airugpua01 airugpua02 airugpua03 airugpua04 airugpua05 airugpua06 '
                                'airugpua07 airugpua08 airugpua09 airugpua10 airugpub01 airugpub02')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--gmem', type=str, help='Video memory reserved for training', default=None)
    parser.add_argument('--round_robin', type=int, help='Number of machines available for submitting each job to avoid sending all jobs to a single machine (0 for selecting all machines)', default=0)
    parser.add_argument('--cache', action='store_true', help='Cache images')

    args = parser.parse_args()

    leaderboard = Leaderboard(trainer_path=args.trainer_path,
                              dataset_type=args.dataset_type,
                              run_name=args.run_name,
                              bundle_size=args.bundle_size,
                              core=args.core,
                              verbose=args.verbose,
                              machines=args.machines,
                              debug=args.debug,
                              shared=args.shared,
                              gmem=args.gmem,
                              round_robin=args.round_robin,
                              cache=args.cache)

    leaderboard.submit()
