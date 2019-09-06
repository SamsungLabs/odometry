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
                 experiment_name,
                 load_experiment_name,
                 bundle_name,
                 load_bundle_name,
                 machines,
                 bundle_size=1,
                 core=8,
                 verbose=False,
                 debug=False,
                 shared=False,
                 gmem=None,
                 round_robin=0,
                 other_args=None):

        if not os.path.exists(script_path):
            raise RuntimeError(f'Could not find trainer script {script_path}')

        self.averager = MetricAverager()

        self.script_path = script_path
        self.experiment_name = experiment_name
        self.load_experiment_name = load_experiment_name
        self.bundle_name = bundle_name
        self.load_bundle_name = load_bundle_name
        self.bundle_size = bundle_size
        self.core = core

        if debug:
            self.leader_boards = ['tum_debug', 'discoman_debug']
        else:
            self.leader_boards = ['kitti_4/6',
                                  'kitti_8/3',
                                  'discoman_v10',
                                  'tum',
                                  'euroc',
                                  'zju',
                                  'saic_office',
                                  'retail_bot']

        self.verbose = verbose
        self.machines = machines.split(' ')
        self.shared = shared
        self.gmem = gmem
        self.round_robin = min(len(self.machines), round_robin or len(self.machines))
        self.other_args = other_args or []

    def submit(self):

        if self.experiment_name == 'leaderboard':
            self.submit_on_all_leader_boards()
        else:
            self.submit_bundle(self.experiment_name)

    def submit_on_all_leader_boards(self):

        pool = Pool(len(self.leader_boards))
        for experiment_name in self.leader_boards:
            self.log(f'Submitting {experiment_name}')
            pool.apply_async(self.submit_bundle, (experiment_name,))
        pool.close()
        pool.join()

    def submit_bundle(self, experiment_name):

        self.setup_logger(experiment_name)

        self.log('Submitting jobs', experiment_name)

        started_jobs_id = set()
        for b in range(self.bundle_size):
            job_id = self.submit_job(experiment_name, b)
            started_jobs_id.add(job_id)

        self.log(f'Started {started_jobs_id}', experiment_name)
        self.wait_jobs(experiment_name, started_jobs_id)

        self.log('Averaging metrics', experiment_name)

        try:
            self.averager.average_run(experiment_name, self.bundle_name)
        except Exception as e:
            self.log(e)

    def submit_job(self, experiment_name, bundle_id):

        run_name = f'{self.bundle_name}_b_{bundle_id}'
        load_name = f'{self.load_bundle_name}_b_{bundle_id}' if self.load_bundle_name else None

        machines = np.random.choice(self.machines, self.round_robin, replace=False)
        seed = np.random.randint(1000000)
        cmd = self.get_lsf_command(experiment_name, run_name, load_name, ' '.join(machines), seed)
        self.log(f'Executing command: {cmd}')

        p = sp.Popen(cmd, shell=True, stdout=sp.PIPE)
        outs, errs = p.communicate(timeout=4)

        job_id = str(outs).split(' ')[1][1:-1]
        return job_id

    def get_lsf_command(self, experiment_name: str, run_name: str, load_name: Union[str, None],
                        machines: str, seed: int) -> str:

        if self.shared:
            mode = 'shared' + (f':gmem={self.gmem}' if self.gmem else '') + ":gtile='!'"
        else:
            mode = 'exclusive_process'

        command = ['bsub',
                   f'-n 1 -R "span[hosts=1] affinity[core({self.core}):distribute=pack]"',
                   f'-o {Path.home().joinpath("lsf").joinpath("%J").as_posix()}_{run_name}',
                   f'-m "{machines}"',
                   f'-gpu "num=1:mode={mode}"',
                   'python',
                   f'{self.script_path}',
                   f'--experiment_name {experiment_name}',
                   f'--run_name {run_name}',
                   f'--bundle_name {self.bundle_name}',
                   f'--seed {seed}']

        if load_name:
            command.extend([f'--load_name {load_name}',
                            f'--load_bundle_name {self.load_bundle_name}',
                            f'--load_experiment_name {self.load_experiment_name}'])

        return ' '.join(command + other_args)

    def wait_jobs(self, experiment_name, started_jobs_id):

        finished = False
        while not finished:

            p = sp.Popen(['bjobs'], shell=True, stdout=sp.PIPE)
            outs, errs = p.communicate()
            outs = outs.decode('utf-8').split('\n')
            job_ids = {outs[i].split(' ')[0] for i in range(1, len(outs) - 1)}

            still_running_jobs = started_jobs_id.intersection(job_ids)
            sorted_jobs = list(still_running_jobs)
            sorted_jobs.sort()
            self.log(f'Running {sorted_jobs}', experiment_name)

            if still_running_jobs:
                time.sleep(10)
            else:
                finished = True
                self.log('All jobs has been finished', experiment_name)

    def setup_logger(self, experiment_name):

        logger = logging.getLogger('leaderboard')
        logger.setLevel(logging.DEBUG)

        experiment_name = experiment_name.replace('/', '_')
        fh = logging.FileHandler(os.path.join(env.PROJECT_PATH, f'log_leaderboard_{experiment_name}.txt'), mode='w+')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        if self.verbose:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            logger.addHandler(sh)

    @staticmethod
    def log(info, experiment_name=None):

        logger = logging.getLogger('leaderboard')

        timestamp = datetime.datetime.now().isoformat().replace('T', ' ')

        if experiment_name:
            logger.info(f'{timestamp} Experiment {experiment_name}. {info}')
        else:
            logger.info(f'{timestamp} {info}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_path', type=str, required=True)

    parser.add_argument('--experiment_name', '-exp', type=str, required=True,
                        help='You can find available experiment names in slam.preprocessing.dataset_configs.py')
    parser.add_argument('--load_experiment_name', '-lexp', type=str, default=None)

    parser.add_argument('--bundle_name', '-n', type=str, required=True,
                        help='Name of the bundle. Must be unique and specific')
    parser.add_argument('--load_bundle_name', '-ln', type=str, default=None,
                        help='Name of the loaded bundle')

    parser.add_argument('--bundle_size', '-b', type=int, required=True, help='Number runs in evaluate')

    parser.add_argument('--core', '-c', type=int, default=3, help='Number of cpu core')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print output to console')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--machines', '-m', help='lsf arg. Specify machines on which execute job',
                        default='airugpua01 airugpua02 airugpua03 airugpua04 airugpua05 airugpua06 '
                                'airugpua07 airugpua08 airugpua09 airugpua10 airugpub01 airugpub02')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--gmem', type=str, default=None, help='Video memory reserved for training')
    parser.add_argument('--round_robin', type=int, default=0,
                        help='Number of machines available for submitting each job '
                             'to avoid sending all jobs to a single machine '
                             '(0 for selecting all machines)')

    args, other_args = parser.parse_known_args()

    leaderboard = Leaderboard(script_path=args.script_path,
                              experiment_name=args.experiment_name,
                              load_experiment_name=args.load_experiment_name,
                              bundle_name=args.bundle_name,
                              load_bundle_name=args.load_bundle_name,
                              bundle_size=args.bundle_size,
                              core=args.core,
                              verbose=args.verbose,
                              machines=args.machines,
                              debug=args.debug,
                              shared=args.shared,
                              gmem=args.gmem,
                              round_robin=args.round_robin,
                              other_args=other_args)

    leaderboard.submit()
