import argparse
import mlflow

import __init_path__
import env


class Cleaner:

    def __init__(self):
        self.client = mlflow.tracking.MlflowClient(env.TRACKING_URI)
        mlflow.set_tracking_uri(env.TRACKING_URI)

    def clean_db(self):
        experiments = self.client.list_experiments()
        for experiment in experiments:
            print(f'Leaderboard {experiment.name}')
            self.clean_experiment(experiment.name)

    def delete_experiment(self, experiment):
        self.client.delete_experiment(experiment.experiment_id)
        print(f'Deleted {experiment.name}')

    def delete_run(self, run_id, run_data):
        run_name = run_data.params.get('run_name', 'Unknown')
        self.client.delete_run(run_id)
        print(f'Deleted {run_name}')

    def clean_experiment(self, leader_board):
        experiment = self.client.get_experiment_by_name(leader_board)

        experiment_id = experiment.experiment_id

        for info in self.client.list_run_infos(experiment_id):
            run_data = self.client.get_run(info.run_id).data
            success = run_data.metrics.get('successfully_finished', False)
            finished = info.status != 'RUNNING'
            if finished and not success:
                try:
                    self.delete_run(info.run_id, run_data)
                except e:
                    print(e)

        if len(self.client.list_run_infos(experiment_id)) == 0:
            self.delete_experiment(experiment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--leader_board', '--dataset_type', type=str, default=None,
                        help='You can find available experiment names in slam.preprocessing.dataset_configs.py')

    args = parser.parse_args()

    cleaner = Cleaner()
    if args.leader_board is None:
        cleaner.clean_db()
    else:
        cleaner.clean_experiment(args.leader_board)
