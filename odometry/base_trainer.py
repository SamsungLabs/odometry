import mlflow
import datetime
import argparse

import __init_path__
import env
from odometry.utils import str2bool
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
        if exp is None:
            raise RuntimeError(f'Could not find exp. Got {exp_name}.'
                               f' Available names {DATASET_TYPES}')

        exp_id = exp.experiment_id
        run_names = [client.get_run(i.run_id).data.params.get('run_name', '') for i in client.list_run_infos(exp_id)]

        if run_name in run_names:
            return False
        else:
            return True

    def train(self):
        raise NotImplemented('Method of abstract class')