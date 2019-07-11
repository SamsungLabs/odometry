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


def on_batch_end(cls, batch, logs=None):
    cls.params['metrics'] = ['loss', 'val_loss']

    logs = logs or {}
    batch_size = logs.get('size', 0)
    if cls.use_steps:
        cls.seen += 1
    else:
        cls.seen += batch_size

    for k in cls.params['metrics']:
        if k in logs:
            cls.log_values.append((k, logs[k]))

    if cls.verbose and cls.seen < cls.target:
        cls.progbar.update(cls.seen, cls.log_values)


keras.callbacks.ProgbarLogger.on_batch_end = on_batch_end


class TerminateOnLR(keras.callbacks.Callback):
    '''Stop training when a lr has become less than passed min_lr.
    # Arguments
        monitor: quantity to be monitored.
        verbose: verbosity mode.
        min_lr: Threshold value for the monitored quantity to reach.
            Training will stop if the monitored value is less than min_lr.
    '''
    def __init__(self,
                 min_lr=None,
                 verbose=0):
        super(TerminateOnLR, self).__init__()
        self.verbose = verbose
        self.min_lr = min_lr
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        mlflow.log_metric('lr', lr, step=epoch) if mlflow.active_run() else None
        if lr < self.min_lr:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            lr = K.get_value(self.model.optimizer.lr)
            print(f'Epoch {self.stopped_epoch}: terminated on lr = {lr} < {self.min_lr}')


class Evaluate(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 predictions_dir=None,
                 visuals_dir=None,
                 period=1,
                 save_best_only=True,
                 max_to_visualize=5):
        super(Evaluate, self).__init__()
        self.model = model
        self.predictions_dir = predictions_dir
        self.visuals_dir = visuals_dir

        self.period = period
        self.epoch_counter = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only
        self.max_to_visualize = max_to_visualize

        self.train_generator = dataset.get_train_generator(trajectory=True)
        self.val_generator = dataset.get_val_generator()
        self.test_generator = dataset.get_test_generator()

        self.df_train = dataset.df_train
        self.df_val = dataset.df_val
        self.df_test = dataset.df_test

        self.y_cols = self.train_generator.y_cols[:]

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

    def _create_trajectory(self, df):
        return RelativeTrajectory.from_dataframe(df[self.y_cols]).to_global()

    def _maybe_save_predictions(self, prediction_id, subset, trajectory_id, predictions):
        if not self.save_predictions:
            return
        file_path = self._create_prediction_file_path(prediction_id, subset, trajectory_id)
        predictions.to_csv(file_path)

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
            normalized_metrics_as_str = ', '.join([f'{key}: {value:.6f}'
                                                   for key, value in normalized_metrics.items()])
            title = f'{trajectory_id.upper()}: {normalized_metrics_as_str}'
            visualize_trajectory_with_gt(gt_trajectory, predicted_trajectory, title=title, file_path=file_path)

    def _evaluate(self, generator, gt, subset, prediction_id):
        if generator is None:
            return dict()

        predictions = self._predict(generator, gt)

        records = []
        gt_by_trajectory = gt.groupby(by='trajectory_id').indices.items()
        for i, (trajectory_id, indices) in enumerate(tqdm.tqdm(gt_by_trajectory,
                                                               total=gt.trajectory_id.nunique(),
                                                               desc=f'Evaluate on {subset}')):
            gt_trajectory = self._create_trajectory(gt.iloc[indices])
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._maybe_save_predictions(prediction_id, subset, trajectory_id, predictions.iloc[indices])

            trajectory_metrics = calculate_metrics(gt_trajectory, predicted_trajectory)
            records.append(trajectory_metrics)

            if i < self.max_to_visualize:
                self._maybe_visualize_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory,
                                                 gt_trajectory, trajectory_metrics)

        total_metrics = {f'{subset}_{key}': value for key, value in average_metrics(records).items()}
        return total_metrics

    def _visualize(self, generator, gt, subset, prediction_id):
        if generator is None:
            return

        predictions = self._predict(generator, gt)

        gt_by_trajectory = gt.groupby(by='trajectory_id').indices.items()
        for trajectory_id, indices in tqdm.tqdm(gt_by_trajectory,
                                                total=gt.trajectory_id.nunique(),
                                                desc=f'Visualize {subset}'):
            predicted_trajectory = self._create_trajectory(predictions.iloc[indices])
            self._maybe_save_predictions(prediction_id, subset, trajectory_id, predictions.iloc[indices])
            self._maybe_visualize_trajectory(prediction_id, subset, trajectory_id, predicted_trajectory)

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
        if self.epoch_counter < self.period:
            return

        self.epoch_counter = 0
        train_loss = logs['loss']
        val_loss = logs.get('val_loss', np.inf)
        if self.save_best_only and self.best_loss < val_loss:
            return

        self.best_loss = min(val_loss, self.best_loss)

        prediction_id = f'{(epoch + 1):03d}_train:{train_loss:.6f}_val:{val_loss:.6f}'

        train_metrics = self._evaluate(self.train_generator, self.df_train, 'train', prediction_id)
        val_metrics = self._evaluate(self.val_generator, self.df_val, 'val', prediction_id)

        if mlflow.active_run():
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in train_metrics.items()]
            [mlflow.log_metric(key=key, value=value, step=epoch) for key, value in val_metrics.items()]

    def on_train_end(self, logs={}):
        prediction_id = 'test'
        test_metrics = self._evaluate(self.test_generator, self.df_test, 'test', prediction_id)
        mlflow.log_metrics(test_metrics) if mlflow.active_run() else None
        self._visualize(self.test_generator, self.df_test, 'test', 'test')
