import mlflow
import shutil
from pathlib import Path
from tqdm import tqdm

import __init_path__
import env

aliases = {'kitti_8_3': 'kitti_8/3', 'kitti_4_6': 'kitti_4/6'}
artifacts_root = Path(env.ARTIFACT_PATH)
client = mlflow.tracking.MlflowClient(env.TRACKING_URI)


def replace_exp_name(exp_name):
    for k, v in aliases.items():
        exp_name = exp_name.replace(k, v)
    return exp_name


def main():
    cleared_dirs_counter = 0
    failed_dirs_counter = 0

    for exp_dir in tqdm(artifacts_root.iterdir(), desc='Experiments dirs'):
        exp_name = replace_exp_name(exp_dir.name)
        experiment = client.get_experiment_by_name(exp_name)
        runs_in_db = client.list_run_infos(experiment.experiment_id)
        runs_id_in_db = [run.run_id for run in runs_in_db]

        for run_dir in tqdm(exp_dir.iterdir(), desc='Runs dirs'):

            if run_dir.name not in runs_id_in_db:
                cleared_dirs_counter += 1
                try:
                    shutil.rmtree(run_dir)
                except Exception as e:
                    print(e)
                    failed_dirs_counter += 1

    print(f'Cleared {cleared_dirs_counter} dirs')
    print(f'Failed to clear {failed_dirs_counter} dirs')


if __name__ == '__main__':
    main()
