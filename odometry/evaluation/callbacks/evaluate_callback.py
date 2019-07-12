import os
import mlflow
import tqdm
import numpy as np
import pandas as pd

import keras
from keras import backend as K

from odometry.evaluation.evaluate import calculate_metrics, average_metrics, normalize_metrics
from odometry.linalg import RelativeTrajectory
from odometry.utils import visualize_trajectory, visualize_trajectory_with_gt


class Evaluate(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 run_dir=None,
                 artifact_dir=None,
                 period=10,
                 save_best_only=True,
                 rpe_indices='full',
                 max_to_visualize=5):
        super(Evaluate, self).__init__()
        self.model = model
        self.run_dir = run_dir
        self.artifact_dir = artifact_dir

        self.period = period
        self.epoch_counter = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only
        self.rpe_indices = rpe_indices
        self.max_to_visualize = max_to_visualize

        self.train_generator = dataset.get_train_generator(trajectory=True)
        self.val_generator = dataset.get_val_generator()
        self.test_generator = dataset.get_test_generator()

        self.df_train = dataset.df_train
        self.df_val = dataset.df_val
        self.df_test = dataset.df_test

        self.y_cols = self.train_generator.y_cols[:]

        if self.run_dir:
            self.visuals_dir = os.path.join(self.run_dir, 'visuals')
            os.makedirs(self.visuals_dir, exist_ok=True)

            self.predictions_dir = os.path.join(self.run_dir, 'predictions')
            os.makedirs(self.predictions_dir, exist_ok=True)

        self.save_artifacts = self.run_dir and self.artifact_dir

    def _create_visualization_file_path(self, prediction_id, subset, trajectory_id):
        file_path = os.path.join(self.visuals_dir, prediction_id, subset, f'{trajectory_id}.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_prediction_file_path(self, prediction_id, subset, trajectory_id):
        file_path = os.path.join(self.predictions_dir, prediction_id, subset, f'{trajectory_id}.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_trajectory(self, df):
        return RelativeTrajectory.from_dataframe(df[self.y_cols]).to_global()

    def _save_predictions(self, prediction_id, subset, trajectory_id, predictions):
        file_path = self._create_prediction_file_path(prediction_id, subset, trajectory_id)
        predictions.to_csv(file_path)

    def _visualize_trajectory(self, prediction_id, subset, trajectory_id, predicted_trajectory,
                              gt_trajectory=None, trajectory_metrics=None):
        file_path = self._create_visualization_file_path(prediction_id, subset, trajectory_id)
        if gt_trajectory is None:
            title = trajectory_id.upper()
            visualize_trajectory(predicted_trajectory, title=title, file_path=file_path)
        else:
            normalized_metrics = normalize_metrics(trajectory_metrics)
            normalized_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                                   for key, value in normalized_metrics.items()])
            title = f'{trajectory_id.upper()}: {normalized_metrics_as_str}'
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory,
                                         title=title, file_path=file_path)

    def _evaluate(self, generator, gt, subset, prediction_id, max_to_visualize=None):
        if generator is None:
            return dict()

        predictions = self._predict(generator, gt)

        records = []

        max_to_visualize = max_to_visualize or gt.trajectory_id.nunique()

        gt_by_trajectory = gt.groupby(by='trajectory_id').indices.items()
        for i, (trajectory_id, indices) in enumerate(tqdm.tqdm(gt_by_trajectory,
                                                               total=gt.trajectory_id.nunique(),
                                                               desc=f'Evaluate on {subset}')):
            gt_trajectory = self._create_trajectory(gt.iloc[indices])
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._save_predictions(prediction_id, subset, trajectory_id, predictions.iloc[indices])

            trajectory_metrics = calculate_metrics(gt_trajectory, predicted_trajectory,
                                                   rpe_indices=self.rpe_indices)
            records.append(trajectory_metrics)

            if i < max_to_visualize:
                self._visualize_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory,
                                           gt_trajectory, trajectory_metrics)

        total_metrics = {f'{subset}_{key}': value for key, value in average_metrics(records).items()}
        return total_metrics

    def _predict(self, generator, gt):
        generator.reset()
        generator.y_cols = self.y_cols[:]
        model_output = self.model.predict_generator(generator, steps=len(generator))
        data = np.stack(model_output).transpose(1, 2, 0)
        data = data.reshape((len(data), -1))
        columns = self.y_cols[:]
        assert data.shape[1] in (len(columns), 2 * len(columns))
        if data.shape[1] == 2 * len(columns):
            columns.extend([col + '_confidence' for col in columns])
        predictions = pd.DataFrame(data=data, index=gt.index, columns=columns)
        return predictions

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_counter += 1
        if self.period and self.epoch_counter < self.period:
            return

        self.epoch_counter = 0
        train_loss = logs['loss']
        val_loss = logs.get('val_loss', np.inf)
        if self.save_best_only and self.best_loss < val_loss:
            return

        self.best_loss = min(val_loss, self.best_loss)

        prediction_id = f'{(epoch + 1):03d}_train:{train_loss:.6f}_val:{val_loss:.6f}'

        train_metrics = self._evaluate(self.train_generator, self.df_train, 'train', prediction_id,
                                       self.max_to_visualize)
        val_metrics = self._evaluate(self.val_generator, self.df_val, 'val', prediction_id,
                                     self.max_to_visualize)

        if mlflow.active_run():
            [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in train_metrics.items()]
            [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in val_metrics.items()]

            if self.save_artifacts:
                mlflow.log_artifacts(self.run_dir, self.artifact_dir)

    def on_train_end(self, logs={}):
        prediction_id = 'test'
        test_metrics = self._evaluate(self.test_generator, self.df_test, 'test', prediction_id)
        mlflow.log_metrics(test_metrics)

        self._visualize(self.test_generator, self.df_test, 'test', 'test')
        if mlflow.active_run() and self.save_artifacts:
            mlflow.log_artifacts(self.run_dir, self.artifact_dir)
