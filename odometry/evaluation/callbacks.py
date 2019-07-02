import os
import keras
import mlflow
import numpy as np
import pandas as pd

from odometry.evaluation.evaluate import calculate_metrics, average_metrics, normalize_metrics
from odometry.linalg import RelativeTrajectory
from odometry.utils import visualize_trajectory, visualize_trajectory_with_gt


class PredictCallback(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 predictions_dir=None,
                 visuals_dir=None,
                 period=1,
                 save_best_only=True):
        super(PredictCallback, self).__init__()
        self.model = model
        self.dataset = dataset
        self.predictions_dir = predictions_dir
        self.visuals_dir = visuals_dir

        self.period = period
        self.epoch_counter = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only

        self.save_visuals = self.visuals_dir is not None
        if self.save_visuals:
            os.makedirs(self.visuals_dir, exist_ok=True)

        self.save_predictions = self.predictions_dir is not None
        if self.save_predictions:
            os.makedirs(self.predictions_dir, exist_ok=True)

    def _create_visualization_file_path(self, prediction_id, subset, trajectory_id):
        file_path = os.path.join(self.visuals_dir, prediction_id, subset, f'{trajectory_id}.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_prediction_file_path(self, prediction_id, subset, trajectory_id):
        file_path = os.path.join(self.predictions_dir, prediction_id, subset, f'{trajectory_id}.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    @staticmethod
    def _create_trajectory(df):
        return RelativeTrajectory.from_dataframe(df).to_global()

    def _maybe_save_trajectory(self, prediction_id, subset, trajectory_id, trajectory):
        if not self.save_predictions:
            return
        file_path = self._create_prediction_file_path(prediction_id, subset, trajectory_id)
        trajectory.to_dataframe().to_csv(file_path)

    def _maybe_visualize_trajectory(self, prediction_id, subset, trajectory_id, predicted_trajectory,
                                    gt_trajectory=None, trajectory_metrics=None):
        if not self.save_visuals:
            return

        file_path = self._create_visualization_file_path(prediction_id, subset, trajectory_id)
        if gt_trajectory is None:
            title = trajectory_id.upper()
            visualize_trajectory(predicted_trajectory, title=title, file_path=file_path)
        else:
            normalized_metrics = normalize_metrics(trajectory_metrics)
            normalized_metrics_as_str = ', '.join(['{}: {:.6f}'.format(key, value)
                                                   for key, value in normalized_metrics.items()])
            title = f'{trajectory_id.upper()}: {normalized_metrics_as_str}'
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, file_path=file_path)

    def _evaluate(self, generator, gt, subset, prediction_id):

        if generator is None:
            return dict()

        predictions = self._predict(generator, gt)

        records = []
        for trajectory_id, indices in gt.groupby(by='trajectory_id').indices.items():
            gt_trajectory = self._create_trajectory(gt.iloc[indices])
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._maybe_save_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory)

            trajectory_metrics = calculate_metrics(gt_trajectory, predicted_trajectory)
            records.append(trajectory_metrics)

            self._maybe_visualize_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory,
                                             gt_trajectory, trajectory_metrics)

        total_metrics = {f'{subset}_{key}': value for key, value in average_metrics(records).items()}
        return total_metrics

    def _visualize(self, generator, gt, subset, prediction_id):

        if generator is None:
            return

        predictions = self._predict(generator, gt)

        for trajectory_id, indices in gt.groupby(by='trajectory_id').indices.items():
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._maybe_save_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory)
            self._maybe_visualize_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory)

    def _predict(self, generator, gt):
        model_output = self.model.predict_generator(generator, steps=len(generator))
        predictions = pd.DataFrame(data=np.stack(model_output)[..., 0].transpose(),
                                   index=gt.index,
                                   columns=self.dataset.y_col)
        return predictions

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_counter += 1
        if self.epoch_counter < self.period:
            return

        self.epoch_counter = 0
        train_loss = logs['loss']
        val_loss = logs['val_loss']
        if self.save_best_only and self.best_loss < val_loss:
            return

        self.best_loss = min(val_loss, self.best_loss)

        prediction_id = '{:03d}_train:{:.6f}_val:{:.6f}'.format(epoch + 1, train_loss, val_loss)
        train_metrics = self._evaluate(self.dataset.get_train_generator(), self.dataset.df_train, 'train', prediction_id)
        val_metrics = self._evaluate(self.dataset.get_val_generator(), self.dataset.df_val, 'val', prediction_id)

        [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in train_metrics.items()]
        [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in val_metrics.items()]

    def on_train_end(self, logs={}):

        prediction_id = 'test'
        test_metrics = self._evaluate(self.dataset.get_test_generator(), self.dataset.df_test, 'test', prediction_id)
        mlflow.log_metrics(test_metrics)
        self._visualize(self.dataset.get_test_generator(), self.dataset.df_test, 'test', 'test')
