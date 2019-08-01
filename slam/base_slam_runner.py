import os
import mlflow

from slam.base_trainer import BaseTrainer

from slam.evaluation.evaluate import (calculate_metrics,
                                      average_metrics,
                                      normalize_metrics)

from slam.linalg import RelativeTrajectory
from slam.utils import visualize_trajectory_with_gt


class BaseSlamRunner(BaseTrainer):

    def __init__(self, knn=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reloc_weights = kwargs['reloc_weights']
        self.optflow_weights = kwargs['optflow_weights']
        self.odometry_model = kwargs['odometry_model']
        self.knn = knn

    def get_slam(self):
        raise RuntimeError('Not implemented')

    def set_dataset_args(self):
        self.x_col = ['path_to_rgb']
        self.y_col = []
        self.image_col = ['path_to_rgb']
        self.load_mode = 'rgb'
        self.preprocess_mode = 'rgb'
        self.batch_size = 1

    def create_file_path(self, trajectory_id, subset):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.run_dir,
                                 subset,
                                 f'{trajectory_name}.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def evaluate(self, predicted_trajectories, gt, subset):

        records = list()

        for trajectory_id, predicted_trajectory in predicted_trajectories.items():

            file_path = self.create_file_path(trajectory_id, subset)

            gt_trajectory = RelativeTrajectory.from_dataframe(gt[gt.trajectory_id == trajectory_id]).to_global()
            record = calculate_metrics(gt_trajectory,
                                       predicted_trajectory,
                                       rpe_indices=self.config['rpe_indices'])

            records.append(record)

            record = normalize_metrics(record)
            trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                                   for key, value in record.items()])
            title = f'{trajectory_id.upper()}: {trajectory_metrics_as_str}'
            visualize_trajectory_with_gt(gt_trajectory,
                                         predicted_trajectory,
                                         title=title,
                                         file_path=file_path)

        total_metrics = {f'{(subset + "_")}{subset}_{key}': float(value)
                         for key, value in average_metrics(records).items()}

        if mlflow.active_run():
            mlflow.log_metrics(total_metrics)
            mlflow.log_artifacts(self.run_dir, subset)

    def run(self):

        dataset = self.get_dataset()

        slam = self.get_slam()
        slam.construct()

        predicted_trajectories = slam.predict_generators(dataset.get_train_generator(as_list=True, append_last=True))
        self.evaluate(predicted_trajectories, dataset.df_train, 'train')

        predicted_trajectories = slam.predict_generators(dataset.get_val_generator(as_list=True, append_last=True))
        self.evaluate(predicted_trajectories, dataset.df_val, 'val')

        predicted_trajectories = slam.predict_generators(dataset.get_test_generator(as_list=True, append_last=True))
        self.evaluate(predicted_trajectories, dataset.df_test, 'test')

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--reloc_weights', type=str)
        parser.add_argument('--optflow_weights', type=str)
        parser.add_argument('--odometry_model', type=str)

        return parser
