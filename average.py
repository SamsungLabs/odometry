import mlflow
import env
import numpy as np
import datetime


def average_metrics(dataset_type):
    metrics, model_name = load_metrics(dataset_type)

    aggregated_metrics = aggregate_metrics(metrics)

    metrics_mean = {k + '_mean': np.mean(v) for k, v in aggregated_metrics.items() if 'test' in k}
    metrics_var = {k + '_var': np.var(v) for k, v in aggregated_metrics.items() if 'test' in k}

    mlflow.set_tracking_uri(env.TRACKING_URI)
    mlflow.set_experiment(dataset_type)

    with mlflow.start_run(run_name=(self.run_name + "_av")):
        mlflow.log_param('run_name', self.run_name + "_av")
        mlflow.log_param('starting_time', datetime.datetime.now().isoformat())

        mlflow.log_param('model.name', model_name)
        mlflow.log_param('num_of_runs_to_average', len(metrics))

        mlflow.log_metrics(metrics_mean)
        mlflow.log_metrics(metrics_var)


def load_metrics(self, dataset_type):
    client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
    exp = client.get_experiment_by_name(dataset_type)
    exp_id = exp.experiment_id

    metrics = list()
    model_name = None
    for run_info in client.list_run_infos(exp_id):
        base_run_name = client.get_run(run_info.run_id).data.params['run_name'].split('_')
        base_run_name = '_'.join(base_run_name[:-2])
        if self.run_name == base_run_name:
            metrics.append(client.get_run(run_info.run_id).data.metrics)
            if not model_name:
                model_name = client.get_run(run_info.run_id).data.params.get('model.name', 'Unknown')

    return metrics, model_name


def aggregate_metrics(metrics):
    aggregated_metrics = {k: [] for k in metrics[0].keys()}

    for metric in metrics:
        for k, v in metric.items():
            aggregated_metrics.get(k, list()).append(v)
    return aggregated_metrics