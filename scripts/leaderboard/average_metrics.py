import mlflow
from mlflow import entities
import datetime
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Set, Union, Iterator

import __init_path__
import env


class MetricAverager:
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
        mlflow.set_tracking_uri(env.TRACKING_URI)
        self.ignore = ['successfully_finished']
        self.save_once = ['num_of_parameters', 'Number of parameters']
        self._run_infos = None

    def average_db(self):
        experiments = self.client.list_experiments()
        for experiment in experiments:
            print(f'Averaging {experiment.name} experiment.')
            self.average_experiment(experiment.name)

    def average_experiment(self, leader_board):
        experiment = self.client.get_experiment_by_name(leader_board)

        bundle_names = set()

        for run_info in self.client.list_run_infos(experiment.experiment_id):
            run = self.client.get_run(run_info.run_id)
            bundle_name = run.data.params['bundle_name']
            bundle_names.add(bundle_name)

        for bundle_name in bundle_names:
            print(f'    Averaging {bundle_name} run.')
            self.average_bundle(leader_board, bundle_name)

        print(f'    Averaged {len(bundle_names)} runs in {leader_board} leaderboard.')

    def average_bundle(self, bundle_name, leader_board):
        mlflow.set_experiment(leader_board)

        metrics, model_name = self.load_metrics(bundle_name, leader_board)
        if metrics is None:
            raise ValueError(f'    No successfully finished runs were found for {bundle_name}')

        metrics_mean = self.calculate_stat(metrics, np.mean)
        metrics_std = self.calculate_stat(metrics, np.std, ignore=self.save_once, suffix='std')

        run_name = bundle_name + '_avg'
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('run_name', run_name)
            mlflow.log_param('bundle_name', bundle_name)
            mlflow.log_param('starting_time', datetime.datetime.now().isoformat())
            mlflow.log_param('model.name', model_name)
            mlflow.log_param('num_of_runs', len(metrics))
            mlflow.log_param('avg', True)

            mlflow.log_metrics(metrics_mean)
            mlflow.log_metrics(metrics_std)
            mlflow.log_metric('successfully_finished', 1)

    def load_metrics(self, bundle_name, leader_board):
        experiment = self.client.get_experiment_by_name(leader_board)

        filter_string = f'params.bundle_name = "{bundle_name}"'
        df = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=filter_string)
        if len(df) == 0:
            return None, None

        model_name = df['params.model.name'].unique()[0] or 'Unknown'

        print(f'Found {len(df)} runs with bundle_name="{bundle_name}"')
        success_col = 'metrics.successfully_finished'
        if not success_col in df.columns or not df[success_col].values.any():
            return None, model_name

        df = df[df[success_col] == 1]
        print(f'Successfully finished: {len(df)} runs')

        average_col = 'params.avg'
        if average_col in df.columns:
            is_average = df[average_col].isin([True, 'True'])
            if is_average.values.any():
                print('Average run already exists')

            df = df[~is_average]

        print(f'Averaging {len(df)} runs')

        if len(df) == 0:
            return None, model_name

        metrics_columns = [col for col in df.columns if col.startswith('metrics.')]
        metrics_columns = [col for col in metrics_columns if not col.endswith('std')]
        metrics = df[metrics_columns]
        mapping = {col: col.replace('metrics.', '') for col in metrics_columns}
        metrics.rename(columns=mapping, inplace=True)
        return metrics, model_name

    def calculate_stat(self, metrics, stat_fn, ignore=None, suffix=None):
        ignore = self.ignore + (ignore or [])
        selected_columns = [col for col in metrics.columns if not col in ignore]
        metrics = metrics[selected_columns]
        if suffix:
            mapping = {col: col + '_' + suffix for col in selected_columns}
            metrics.rename(columns=mapping, inplace=True)
        stat = metrics.apply(stat_fn, axis=0)
        return stat.to_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--leader_board', '--dataset_type', type=str, default=None,
                        help='You can find available experiment names in slam.preprocessing.dataset_configs.py')

    parser.add_argument('--bundle_name', '-n', type=str, default=None,
                        help='Name of the bundle. Must be unique and specific')

    args = parser.parse_args()

    averager = MetricAverager()
    if args.leader_board is None:
        averager.average_db()
    elif args.leader_board is not None and args.bundle_name is None:
        averager.average_experiment(args.leader_board)
    else:
        averager.average_bundle(bundle_name=args.bundle_name, leader_board=args.leader_board)
