import mlflow
import datetime
import numpy as np
import argparse
from collections import defaultdict

import __init_path__
import env


def average_metrics(run_name, dataset_type):
    metrics, model_name = load_metrics(run_name, dataset_type)

    aggregated_metrics = aggregate_metrics(metrics)

    metrics_mean = {k + '_mean': np.mean(v) for k, v in aggregated_metrics.items()}
    metrics_std = {k + '_std': np.std(v) for k, v in aggregated_metrics.items()}

    num_of_runs = len(next(iter(aggregated_metrics.values())))

    mlflow.set_tracking_uri(env.TRACKING_URI)
    mlflow.set_experiment(dataset_type)

    run_name = run_name + '_avg'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('run_name', run_name)
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())

        mlflow.log_param('model.name', model_name)
        mlflow.log_param('num_of_runs', num_of_runs)
        mlflow.log_param('avg', True)

        mlflow.log_metrics(metrics_mean)
        mlflow.log_metrics(metrics_std)


def load_metrics(run_name, dataset_type):
    client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
    exp = client.get_experiment_by_name(dataset_type)
    exp_id = exp.experiment_id

    metrics = list()
    model_name = None
    for run_info in client.list_run_infos(exp_id):
        base_run_name = client.get_run(run_info.run_id).data.params['run_name'].split('_')
        base_run_name = '_'.join(base_run_name[:-2])
        if run_name == base_run_name:
            metrics.append(client.get_run(run_info.run_id).data.metrics)
            if not model_name:
                model_name = client.get_run(run_info.run_id).data.params.get('model_name', 'Unknown')

    return metrics, model_name


def aggregate_metrics(metrics):
    aggregated_metrics = defaultdict(list)

    for metric in metrics:
        for k, v in metric.items():
                aggregated_metrics[k].append(v)

    return aggregated_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', '-t', type=str, required=True,
                        help='You can find available exp names in odometry.preprocessing.dataset_configs.py')

    parser.add_argument('--run_name', '-n', type=str, help='Name of the run. Must be unique and specific',
                        required=True)

    args = parser.parse_args()

    average_metrics(args.run_name, args.dataset_type)
