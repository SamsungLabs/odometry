import mlflow
import datetime
import numpy as np
import argparse
from collections import defaultdict

import __init_path__
import env

from typing import List, Set


def average_db():
    client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
    experiments = client.list_experiments()
    for experiment in experiments:
        print(f'Averaging {experiment} experiment.')
        average_experiment(experiment.name)


def average_experiment(experiment_name, client=None):
    client = client or mlflow.tracking.MlflowClient(env.TRACKING_URI)
    experiment_id = client.get_experiment_by_name(experiment_name)
    run_infos = client.list_run_infos(experiment_id)
    run_names = [run_info.run_name for run_info in run_infos]
    base_names = get_base_names(run_names)
    base_names = filter_already_averaged(run_names, base_names)

    counter = 0
    for base_name in base_names:
        print(f'Averaging {base_name} run.')
        average_run(experiment_name, base_name)
        counter += 1
    print(f'Averaged {counter} runs in {experiment_name} experiment.')


def get_base_names(run_names: List[str]) -> Set[str]:
    base_names = set()
    for run_name in run_names:
        try:
            base_names.add(get_base_name(run_name))
        except Exception as e:
            print(e)
    return base_names


def filter_already_averaged(run_names: List[str], base_names: Set[str]) -> Set[str]: 
    filtered_base_names = set()
    for base_name in base_names:
        if (base_name + '_avg') not in run_names:
            filtered_base_names.add(base_name)
    return filtered_base_names


def get_base_name(run_name: str) -> str:
    run_name_split = run_name.split('_')
    if run_name_split[-2] != 'b':
        raise RuntimeError(f'It seems like given run name {run_name} is not belongs to any bundle')
    else:
        return '_'.join(run_name_split[:-2])


def average_run(experiment_name, run_name):

    metrics, model_name = load_metrics(experiment_name, experiment_name)

    aggregated_metrics = aggregate_metrics(metrics)

    metrics_mean = {k + '_mean': np.mean(v) for k, v in aggregated_metrics.items()}
    metrics_std = {k + '_std': np.std(v) for k, v in aggregated_metrics.items()}

    num_of_runs = len(next(iter(aggregated_metrics.values())))

    mlflow.set_tracking_uri(env.TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    run_name = run_name + '_avg'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('run_name', run_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())

        mlflow.log_param('model.name', model_name)
        mlflow.log_param('num_of_runs', num_of_runs)
        mlflow.log_param('avg', True)

        mlflow.log_metrics(metrics_mean)
        mlflow.log_metrics(metrics_std)


def load_metrics(experiment_name, run_name):
    client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id

    metrics = list()
    model_name = None
    for run_info in client.list_run_infos(exp_id):
        data = client.get_run(run_info.run_id).data
        base_run_name = get_base_name(data.params['run_name'])
        if run_name == base_run_name:
            metrics.append(data.metrics)
            model_name = model_name or data.params.get('model.name', 'Unknown')

    return metrics, model_name


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

    if args.experiment_name is None:
        average_db()
    elif args.experiment_name is not None and args.run_name is None:
        average_experiment(args.experiment_name)
    else:
        average_run(run_name=args.run_name,  experiment_name=args.experiment_name)
