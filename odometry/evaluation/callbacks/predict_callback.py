import os
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

import keras
from keras import backend as K

from odometry.evaluation.evaluate import (calculate_metrics,
                                          average_metrics,
                                          normalize_metrics)
from odometry.linalg import RelativeTrajectory
from odometry.utils import visualize_trajectory_with_gt, visualize_trajectory


class Predict(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 run_dir=None,
                 save_dir=None,
                 artifact_dir=None,
                 period=10,
                 save_best_only=True,
                 max_to_visualize=5,
                 evaluate=False,
                 rpe_indices='full'):
        super(Predict, self).__init__()
        self.model = model
        self.run_dir = run_dir
        self.save_dir = save_dir
        self.artifact_dir = artifact_dir

        self.period = period
        self.epoch = 0
        self.last_evaluated_epoch = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only
        self.max_to_visualize = max_to_visualize
        self.evaluate = evaluate
        self.rpe_indices = rpe_indices

        self.train_generator = dataset.get_train_generator(trajectory=True)
        self.val_generator = dataset.get_val_generator()
        self.test_generator = dataset.get_test_generator()

        self.df_train = dataset.df_train
        self.df_val = dataset.df_val
        self.df_test = dataset.df_test

        self.y_cols = self.train_generator.y_cols[:]

        if self.save_dir:
            self.visuals_dir = os.path.join(self.save_dir, 'visuals')
            os.makedirs(self.visuals_dir, exist_ok=True)

            self.predictions_dir = os.path.join(self.save_dir, 'predictions')
            os.makedirs(self.predictions_dir, exist_ok=True)

        self.save_artifacts = self.run_dir and self.artifact_dir

    def _create_visualization_file_path(self, prediction_id, subset, trajectory_id):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.visuals_dir,
                                 prediction_id,
                                 subset,
                                 f'{trajectory_name}.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_prediction_file_path(self, prediction_id, subset, trajectory_id):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.predictions_dir,
                                 prediction_id,
                                 subset,
                                 f'{trajectory_name}.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_trajectory(self, df):
        return RelativeTrajectory.from_dataframe(df[self.y_cols]).to_global()

    def _save_predictions(self, prediction_id, subset, trajectory_id, predictions):
        file_path = self._create_prediction_file_path(prediction_id,
                                                      subset,
                                                      trajectory_id)
        predictions.to_csv(file_path)

    def _visualize_trajectory(self,
                              prediction_id,
                              subset,
                              trajectory_id,
                              predicted_trajectory,
                              gt_trajectory=None,
                              trajectory_metrics=None):
        file_path = self._create_visualization_file_path(prediction_id,
                                                         subset,
                                                         trajectory_id)
        if gt_trajectory is None:
            title = trajectory_id.upper()
            visualize_trajectory(predicted_trajectory, title=title, file_path=file_path)
        else:
            trajectory_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                                   for key, value in trajectory_metrics.items()])
            title = f'{trajectory_id.upper()}: {trajectory_metrics_as_str}'
            visualize_trajectory_with_gt(gt_trajectory,
                                         predicted_trajectory,
                                         title=title,
                                         file_path=file_path)

    def _predict_generator(self, generator, gt):
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

    def _predict(self,
                 generator,
                 gt,
                 subset,
                 prediction_id,
                 max_to_visualize=None):
        if generator is None:
            return dict()

        predictions = self._predict_generator(generator, gt)

        records = [] if self.evaluate else None

        nunique = gt.trajectory_id.nunique()
        max_to_visualize = max_to_visualize or nunique

        counter = 0
        for trajectory_id, indices in tqdm(gt.groupby(by='trajectory_id').indices.items(),
                                           total=nunique,
                                           desc=f'Evaluate on {subset}'):
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._save_predictions(prediction_id,
                                   subset,
                                   trajectory_id,
                                   predictions.iloc[indices])
            if self.evaluate:
                gt_trajectory = self._create_trajectory(gt.iloc[indices])

                trajectory_metrics = calculate_metrics(gt_trajectory,
                                                       predicted_trajectory,
                                                       rpe_indices=self.rpe_indices)
                records.append(trajectory_metrics)

            if not counter < max_to_visualize:
                continue

            if self.evaluate:
                self._visualize_trajectory(prediction_id,
                                           subset,
                                           trajectory_id,
                                           predicted_trajectory,
                                           gt_trajectory,
                                           normalize_metrics(trajectory_metrics))
            else:
                self._visualize_trajectory(prediction_id,
                                           subset,
                                           trajectory_id,
                                           predicted_trajectory)
            counter += 1

        if self.evaluate:
            total_metrics = {f'{subset}_{key}': value 
                             for key, value in average_metrics(records).items()}
            return total_metrics

    def on_epoch_end(self, epoch, logs={}):

        self.last_evaluated_epoch = epoch
        self.epoch += 1

        if not self.period or self.epoch % self.period:
            return

        train_loss = logs['loss']
        val_loss = logs.get('val_loss', np.inf)
        if self.save_best_only and self.best_loss < val_loss:
            return

        self.best_loss = min(val_loss, self.best_loss)

        prediction_id = f'{(epoch + 1):03d}_train:{train_loss:.6f}_val:{val_loss:.6f}'

        train_metrics = self._predict(self.train_generator,
                                      self.df_train,
                                      'train',
                                      prediction_id,
                                      self.max_to_visualize)
        val_metrics = self._predict(self.val_generator,
                                    self.df_val,
                                    'val',
                                    prediction_id,
                                    self.max_to_visualize)

        if self.evaluate and mlflow.active_run():
            [mlflow.log_metric(key=key, value=value, step=epoch + 1) for key, value in train_metrics.items()]
            [mlflow.log_metric(key=key, value=value, step=epoch + 1) for key, value in val_metrics.items()]

            if self.save_artifacts:
                mlflow.log_artifacts(self.run_dir, self.artifact_dir)

    def on_train_end(self, logs={}):

        # Check to not calculate metrics twice on_train_end
        if self.last_evaluated_epoch != (self.epoch - 1):
            self.period = 1  
            self.on_epoch_end(self.epoch - 1, logs)

        prediction_id = 'test'
        test_metrics = self._predict(self.test_generator,
                                     self.df_test,
                                     'test',
                                     prediction_id)
        if mlflow.active_run():
            if self.evaluate:
                mlflow.log_metrics(test_metrics)

            if self.save_artifacts:
                mlflow.log_artifacts(self.run_dir, self.artifact_dir)
