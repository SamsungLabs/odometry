import os
import mlflow
import numpy as np
import pandas as pd
from itertools import chain
from multiprocessing import Pool

import keras
from keras import backend as K

from slam.evaluation.evaluate import (calculate_metrics,
                                      average_metrics,
                                      normalize_metrics)
from slam.linalg import RelativeTrajectory
from slam.utils import visualize_trajectory_with_gt, visualize_trajectory


def _process_single_task(args):
    predicted_trajectory, gt_trajectory, trajectory_id, rpe_indices = args
    trajectory_metrics = calculate_metrics(gt_trajectory,
                                           predicted_trajectory,
                                           rpe_indices=rpe_indices)
    return trajectory_metrics


class Predict(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 run_dir=None,
                 save_dir=None,
                 artifact_dir=None,
                 prefix=None,
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
        self.prefix = prefix

        self.period = period
        self.epoch = 0
        self.last_evaluated_epoch = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only
        self.max_to_visualize = max_to_visualize
        self.evaluate = evaluate
        self.rpe_indices = rpe_indices
        self.workers = 8

        self.train_generator = dataset.get_train_generator(trajectory=self.evaluate)
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

    def _create_visualization_file_path(self, trajectory_id, subset, prediction_id):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.visuals_dir,
                                 prediction_id,
                                 subset,
                                 f'{trajectory_name}.html')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_prediction_file_path(self, trajectory_id, subset, prediction_id):
        trajectory_name = trajectory_id.replace('/', '_')
        file_path = os.path.join(self.predictions_dir,
                                 prediction_id,
                                 subset,
                                 f'{trajectory_name}.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return file_path

    def _create_trajectory(self, df):
        return RelativeTrajectory.from_dataframe(df[self.y_cols]).to_global()

    def _save_predictions(self, predictions, trajectory_id, subset, prediction_id):
        file_path = self._create_prediction_file_path(trajectory_id,
                                                      subset,
                                                      prediction_id)
        predictions.to_csv(file_path)

    def _visualize_trajectory(self,
                              predicted_trajectory,
                              gt_trajectory,
                              trajectory_id,
                              subset,
                              prediction_id,
                              trajectory_metrics=None):
        file_path = self._create_visualization_file_path(trajectory_id,
                                                         subset,
                                                         prediction_id)
        if gt_trajectory is None:
            title = trajectory_id.upper()
            visualize_trajectory(predicted_trajectory, title=title, file_path=file_path)
        else:
            trajectory_metrics = normalize_metrics(trajectory_metrics)
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
        predictions = pd.DataFrame(data=data, index=gt.index, columns=columns).astype(float)
        predictions['path_to_rgb'] = gt.path_to_rgb
        predictions['path_to_rgb_next'] = gt.path_to_rgb_next
        return predictions

    def _create_tasks(self,
                      predictions,
                      gt,
                      subset,
                      prediction_id):
        tasks = []

        for trajectory_id, indices in gt.groupby(by='trajectory_id').indices.items():
            self._save_predictions(predictions.iloc[indices],
                                   trajectory_id,
                                   subset,
                                   prediction_id)

            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            gt_trajectory = self._create_trajectory(gt.iloc[indices]) if self.evaluate else None

            tasks.append([predicted_trajectory,
                          gt_trajectory,
                          trajectory_id,
                          self.rpe_indices])

        return tasks

    def _predict(self,
                 generator,
                 gt,
                 subset,
                 prediction_id,
                 max_to_visualize=None):

        if generator is None:
            return dict()

        predictions = self._predict_generator(generator, gt)

        tasks = self._create_tasks(predictions, gt, subset, prediction_id)

        if self.evaluate:
            pool = Pool(self.workers)
            records = [res for res in pool.imap(_process_single_task, tasks)]
            pool.close()
            pool.join()

        max_to_visualize = (max_to_visualize if max_to_visualize is not None
                            else gt.trajectory_id.nunique())

        counter = 0

        for index, task in enumerate(tasks):
            predicted_trajectory, gt_trajectory, trajectory_id, _ = task
            if counter < max_to_visualize:
                trajectory_metrics = records[index] if self.evaluate else None
                self._visualize_trajectory(predicted_trajectory,
                                           gt_trajectory,
                                           trajectory_id,
                                           subset,
                                           prediction_id,
                                           trajectory_metrics)
            counter += 1

        if self.evaluate:
            total_metrics = {f'{self.prefix}_{subset}_{key}': value if self.prefix else f'{subset}_{key}'
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

        if mlflow.active_run():
            if self.evaluate:
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
