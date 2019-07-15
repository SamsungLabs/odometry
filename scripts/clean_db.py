import __init_path__
import env

import mlflow


if __name__ == '__main__':
    client = mlflow.tracking.MlflowClient(env.TRACKING_URI)

    for exp in client.list_experiments():
        exp_id = exp.experiment_id
        for info in client.list_run_infos(exp_id):
            run = client.get_run(info.run_id)
            if not run.data.metric['successfully_finished']:
                print(f'Run from experiment {exp.name} run {run.data.params["run_name"]} has been deleted')
                # client.delete_run(run_id)
