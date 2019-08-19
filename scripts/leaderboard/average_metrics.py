import mlflow
from mlflow import entities
import datetime
import numpy as np
import argparse
from collections import defaultdict

import __init_path__
import env

from typing import List, Set, Union


class MetricAverager:
    def __init__(self):
        self.client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
        mlflow.set_tracking_uri(env.TRACKING_URI)
        self.save_once = ['successfully_finished', 'num_of_parameters', 'Number of parameters']
        self._run_infos = None

    def average_db(self):
        experiments = self.client.list_experiments()
        for experiment in experiments:
            print(f'Averaging {experiment.name} experiment.')
            self.average_experiment(experiment.name)

    def average_experiment(self, experiment_name):
        self._run_infos = None
        run_infos = self.get_run_infos(experiment_name)
        run_names = self.get_run_names(run_infos)
        base_names = self.get_base_names(run_names)
        base_names = self.filter_already_averaged(run_names, base_names)

        counter = 0
        for base_name in base_names:
            print(f'    Averaging {base_name} run.')
            self.average_run(experiment_name, base_name)
            counter += 1
        print(f'    Averaged {counter} runs in {experiment_name} experiment.')

    def get_run_names(self, run_infos: List[entities.RunInfo]) -> List[str]:
        run_names = list()
        for run_info in run_infos:
            run = self.client.get_run(run_info.run_id)
            run_names.append(run.data.params['run_name'])
        return run_names

    def get_base_names(self, run_names: List[str]) -> Set[str]:
        base_names = set()
        for run_name in run_names:
            base_name = self.get_base_name(run_name)
            if base_name is None:
                print(f'    It seems like given run name {run_name} is not belongs to any bundle')
            else:
                base_names.add(self.get_base_name(run_name))
        return base_names

    @staticmethod
    def filter_already_averaged(run_names: List[str], base_names: Set[str]) -> Set[str]:
        filtered_base_names = set()
        for base_name in base_names:
            if (base_name + '_avg') not in run_names:
                filtered_base_names.add(base_name)
        return filtered_base_names

    @staticmethod
    def get_base_name(run_name: str) -> Union[str, None]:
        run_name_split = run_name.split('_')

        if len(run_name) < 2 or (run_name_split[-2] != 'b'):
            return None
        else:
            return '_'.join(run_name_split[:-2])

    def get_run_infos(self, experiment_name):
        if self._run_infos is None:
            experiment = self.client.get_experiment_by_name(experiment_name)
            run_infos = self.client.list_run_infos(experiment.experiment_id)
            return run_infos
        else:
            return self._run_infos

    def calculate_mean(self, metrics):
        metrics_mean = dict()
        for k, v in metrics.items():
            if k in self.save_once:
                metrics_mean[k] = v[0]
                continue
            metrics_mean[k] = np.mean(v)
        return metrics_mean

    def calculate_std(self, metrics):
        metrics_std = dict()
        for k, v in metrics.items():
            if k in self.save_once:
                continue
            metrics_std[k + '_std'] = np.std(v)
        return metrics_std

    def average_run(self, experiment_name, run_name):
        mlflow.set_experiment(experiment_name)

        metrics, model_name = self.load_metrics(experiment_name, run_name)
        aggregated_metrics = self.aggregate_metrics(metrics)
        metrics_mean = self.calculate_mean(aggregated_metrics)
        metrics_std = self.calculate_std(aggregated_metrics)

        num_of_runs = len(next(iter(aggregated_metrics.values())))
        run_name = run_name + '_avg'
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('run_name', run_name)
            mlflow.log_param('starting_time', datetime.datetime.now().isoformat())

            mlflow.log_param('model.name', model_name)
            mlflow.log_param('num_of_runs', num_of_runs)
            mlflow.log_param('avg', True)

            mlflow.log_metrics(metrics_mean)
            mlflow.log_metrics(metrics_std)
            mlflow.log_metric('successfully_finished', 1)

    def load_metrics(self, experiment_name, base_name):
        metrics = list()
        model_name = None
        run_infos = self.get_run_infos(experiment_name)
        for run_info in run_infos:
            data = self.client.get_run(run_info.run_id).data
            current_base_name = self.get_base_name(data.params['run_name'])
            if base_name == current_base_name:
                metrics.append(data.metrics)
                model_name = model_name or data.params.get('model.name', 'Unknown')

        return metrics, model_name

    @staticmethod
    def aggregate_metrics(metrics):
        aggregated_metrics = defaultdict(list)

        for metric in metrics:
            for k, v in metric.items():
                aggregated_metrics[k].append(v)

        return aggregated_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', '-t', type=str, default=None,
                        help='You can find available exp names in slam.preprocessing.dataset_configs.py')

    parser.add_argument('--run_name', '-n', type=str, default=None,
                        help='Name of the run. Must be unique and specific')

    args = parser.parse_args()

    averager = MetricAverager()
    if args.experiment_name is None:
        averager.average_db()
    elif args.experiment_name is not None and args.run_name is None:
        averager.average_experiment(args.experiment_name)
    else:
        averager.average_run(run_name=args.run_name, experiment_name=args.experiment_name)
