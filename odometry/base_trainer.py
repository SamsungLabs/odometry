import os
import mlflow
import datetime
import argparse

import env

from odometry.preprocessing.dataset_configs import get_config, DATASET_TYPES


class BaseTrainer:
    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 lsf=False,
                 batch=1,
                 prediction_dir='predictions',
                 visuals_dir='visuals',
                 period=1,
                 save_best_only=False
                 ):

        self.tracking_uri = env.TRACKING_URI

        self.config = get_config(dataset_root, dataset_type)

        if not self.is_unique_run_name(self.config['exp_name'], run_name):
            raise RuntimeError('run_name must be unique')

        # MLFlow initialization
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.config['exp_name'])
        mlflow.start_run(run_name=run_name)
        print(f'Active run {mlflow.active_run()}')
        mlflow.log_param('run_name', run_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())

        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.run_name = run_name
        self.lsf = lsf
        self.batch = batch
        self.prediction_dir = prediction_dir
        self.visuals_dir = visuals_dir
        self.period = period
        self.save_best_only = save_best_only

    def is_unique_run_name(self, exp_name, run_name):
        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        exp = client.get_experiment_by_name(exp_name)
        exp_name = exp_name.replace('/','_')
        mlflow.create_experiment(exp_name, os.path.join(env.ARTIFACT_PATH, exp_name)) if exp is None else None

        exp_id = exp.experiment_id

        run_names = list()
        for info in client.list_run_infos(exp_id):
            run_names.append(client.get_run(info.run_id).data.params.get('run_name', ''))

        return run_name not in run_names

    def train(self):
        raise NotImplemented('Method of abstract class')

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset_root', '-r', type=str, help='Directory with trajectories', required=True)
        parser.add_argument('--dataset_type', '-t', type=str, choices=DATASET_TYPES, required=True)
        parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Must be unique and specific',
                            required=True)
        parser.add_argument('--prediction_dir', '-p', type=str, help='Name of subdir to store predictions',
                            default=os.path.join(env.PROJECT_PATH, 'predictions'))
        parser.add_argument('--visuals_dir', '-v', type=str, help='Name of subdir to store visualizations',
                            default=os.path.join(env.PROJECT_PATH, 'visuals'))
        parser.add_argument('--period', type=int, help='Period of evaluating train and val metrics',
                            default=1)
        parser.add_argument('--save_best_only', action='store_true', help='Evaluate metrics only for best losses',
                            default=False)

        return parser
