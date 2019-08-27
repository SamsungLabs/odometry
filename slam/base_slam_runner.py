import os
import mlflow

from slam.base_trainer import BaseTrainer

from slam.evaluation import (calculate_metrics,
                             average_metrics,
                             normalize_metrics)

from slam.linalg import RelativeTrajectory
from slam.utils import visualize_trajectory_with_gt


class BaseSlamRunner(BaseTrainer):

    def __init__(self, reloc_weights, optflow_weights, odometry_model, knn=20, *args, **kwargs):
        super().__init__(per_process_gpu_memory_fraction=1, *args, **kwargs)
        self.reloc_weights = reloc_weights
        self.optflow_weights = optflow_weights
        self.odometry_model = odometry_model
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
        self.target_size = -1

    def create_visualization_path(self, trajectory_id, subset, trajectory_type):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.run_dir, 'visuals', trajectory_type, subset, trajectory_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def create_prediction_path(self, trajectory_id):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.run_dir, 'predictions', trajectory_name, 'df.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def evaluate_trajectory(self, prediction, gt, subset):

        trajectory_id = prediction['id']

        prediction_path = self.create_prediction_path(trajectory_id)
        prediction['frame_history'].to_csv(prediction_path, index=False)

        gt_trajectory = RelativeTrajectory.from_dataframe(gt[gt.trajectory_id == trajectory_id]).to_global()

        trajectory_types = ['slam_trajectory', 'odometry_trajectory', 'odometry_trajectory_with_loop_closures']
        slam_record = None
        for trajectory_type in trajectory_types:
            predicted_trajectory = prediction[trajectory_type]
            record = calculate_metrics(gt_trajectory, predicted_trajectory, rpe_indices=self.config['rpe_indices'])

            if trajectory_type == 'slam_trajectory':
                slam_record = record

            normalized_record = normalize_metrics(record)
            trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}' for key, value in normalized_record.items()])
            title = f'{trajectory_id.upper()}: {trajectory_metrics_as_str}'
            visualization_path = self.create_visualization_path(trajectory_id, subset, trajectory_type)
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, file_path=visualization_path)

            mlflow.log_artifacts(self.run_dir) if mlflow.active_run() else None

        return slam_record

    def evaluate_subset(self, slam, generators, df, subset):

        if generators is None or len(generators) == 0:
            return

        records = list()
        for generator in generators:
            print(f'Predicting {generator.trajectory_id}')
            prediction = slam.predict_generator(generator)
            record = self.evaluate_trajectory(prediction, df, subset)
            records.append(record)

        if mlflow.active_run():
            total_metrics = {f'{subset}_{key}': float(value) for key, value in average_metrics(records).items()}
            mlflow.log_metrics(total_metrics)

    def run(self):

        dataset = self.get_dataset()

        slam = self.get_slam()
        slam.construct()

        generators = dataset.get_train_generator(as_is=True, as_list=True, include_last=True)
        self.evaluate_subset(slam, generators, dataset.df_train, 'train')

        generators = dataset.get_val_generator(as_list=True, include_last=True)
        self.evaluate_subset(slam, generators, dataset.df_val, 'val')

        generators = dataset.get_test_generator(as_list=True, include_last=True)
        self.evaluate_subset(slam, generators, dataset.df_test, 'test')

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--reloc_weights', type=str)
        parser.add_argument('--optflow_weights', type=str)
        parser.add_argument('--odometry_model', type=str)
        parser.add_argument('--knn', type=int, default=20, help='Number of nearest neighbors')

        return parser
