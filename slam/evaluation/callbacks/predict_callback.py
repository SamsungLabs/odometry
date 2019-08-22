import os
import stat
import mlflow
import numpy as np
import pandas as pd
from multiprocessing import Pool

import keras
from keras import backend as K

from slam.evaluation import calculate_metrics, average_metrics, normalize_metrics
from slam.linalg import RelativeTrajectory
from slam.utils import (visualize_trajectory_with_gt,
                        visualize_trajectory,
                        create_vis_file_path,
                        create_prediction_file_path,
                        partial_format)


def process_single_task(args):
    gt_trajectory = args['gt']
    predicted_trajectory = args['predicted']
    rpe_indices = args['rpe_indices']
    backend = args['backend']
    cuda = args['cuda']
    trajectory_metrics = calculate_metrics(gt_trajectory,
                                           predicted_trajectory,
                                           rpe_indices=rpe_indices,
                                           backend=backend,
                                           cuda=cuda)
    return trajectory_metrics


class Predict(keras.callbacks.Callback):
    def __init__(self,
                 model,
                 dataset,
                 run_dir=None,
                 save_dir=None,
                 artifact_dir=None,
                 prefix=None,
                 monitor='val_loss',
                 period=10,
                 save_best_only=True,
                 max_to_visualize=5,
                 evaluate=False,
                 rpe_indices='full',
                 backend='numpy',
                 cuda=False,
                 workers=8):
        super(Predict, self).__init__()
        self.model = model
        self.run_dir = run_dir
        self.save_dir = save_dir
        self.artifact_dir = artifact_dir
        self.prefix = prefix
        self.add_prefix = lambda k: (self.prefix + '_' + k) if self.prefix else k

        self.monitor = monitor
        self.template = ''.join(['{epoch:03d}', '_', self.monitor, '_', '{', self.monitor, ':.6f', '}'])
        self.period = period
        self.epoch = 0
        self.epochs_since_last_predict = 0
        self.best_loss = np.inf
        self.save_best_only = save_best_only
        self.max_to_visualize = max_to_visualize
        self.evaluate = evaluate
        self.rpe_indices = rpe_indices
        self.backend = backend
        self.cuda = cuda
        self.workers = workers if backend == 'numpy' else 0

        self.train_generator = dataset.get_train_generator(as_is=self.evaluate)
        self.val_generator = dataset.get_val_generator()
        self.test_generator = dataset.get_test_generator()

        self.df_train = dataset.df_train
        self.df_val = dataset.df_val
        self.df_test = dataset.df_test

        self.y_cols = self.train_generator.y_cols[:]

        self.save_artifacts = self.run_dir and self.artifact_dir

    def create_trajectory(self, df):
        return RelativeTrajectory.from_dataframe(df[self.y_cols]).to_global()

    def save_predictions(self, predictions, trajectory_id, subset, prediction_id):
        file_path = create_prediction_file_path(trajectory_id, subset, prediction_id, self.save_dir)
        predictions.to_csv(file_path)
        os.chmod(file_path, 0o777)

    def visualize_trajectory(self,
                             predicted_trajectory,
                             gt_trajectory,
                             trajectory_id,
                             subset,
                             prediction_id,
                             record=None):
        file_path = create_vis_file_path(trajectory_id, subset, prediction_id, self.save_dir)
        if gt_trajectory is None:
            title = trajectory_id.upper()
            visualize_trajectory(predicted_trajectory, title=title, file_path=file_path)
        else:
            record_as_str = ', '.join([f'{k}: {v:.6f}' for k, v in normalize_metrics(record).items()])
            title = f'{trajectory_id.upper()}: {record_as_str}'
            visualize_trajectory_with_gt(gt_trajectory,
                                         predicted_trajectory,
                                         title=title,
                                         file_path=file_path)
        os.chmod(file_path, 0o777)

    def predict_generator(self, generator):
        generator.reset()
        generator.y_cols = self.y_cols[:]
        model_output = self.model.predict_generator(generator, steps=len(generator))
        data = np.stack(model_output).transpose(1, 2, 0)
        data = data.reshape((len(data), -1))
        columns = self.y_cols[:]
        assert data.shape[1] in (len(columns), 2 * len(columns))
        if data.shape[1] == 2 * len(columns):
            columns.extend([col + '_confidence' for col in columns])

        predictions = pd.DataFrame(data=data, index=generator.df.index, columns=columns).astype(float)
        predictions['path_to_rgb'] = generator.df.path_to_rgb
        predictions['path_to_rgb_next'] = generator.df.path_to_rgb_next
        return predictions

    def create_tasks(self, generator, subset):
        if generator is None:
            return dict()

        gt = generator.df
        predictions = self.predict_generator(generator)

        tasks = []

        for trajectory_id, indices in gt.groupby(by='trajectory_id').indices.items():
            predicted_df = predictions.iloc[indices]
            predicted_trajectory = self.create_trajectory(predicted_df)
            gt_trajectory = self.create_trajectory(gt.iloc[indices]) if self.evaluate else None

            tasks.append({'df': predicted_df,
                          'predicted': predicted_trajectory,
                          'gt': gt_trajectory,
                          'id': trajectory_id,
                          'subset': subset,
                          'rpe_indices': self.rpe_indices,
                          'backend': self.backend,
                          'cuda': self.cuda})

        return tasks

    def save_tasks(self, tasks, prediction_id, max_to_visualize=None):
        max_to_visualize = max_to_visualize or len(tasks)

        for counter, task in enumerate(tasks):
            predicted_df = task['df']
            trajectory_id = task['id']
            subset = task['subset']

            self.save_predictions(predicted_df,
                                  trajectory_id,
                                  subset,
                                  prediction_id)

            if counter < max_to_visualize:
                gt_trajectory = task['gt']
                predicted_trajectory = task['predicted']
                record = task.get('record', None)

                self.visualize_trajectory(predicted_trajectory,
                                          gt_trajectory,
                                          trajectory_id,
                                          subset,
                                          prediction_id,
                                          record)

    def process_tasks(self, tasks):
        if self.workers:
            with Pool(self.workers) as pool:
                records = [res for res in pool.imap(process_single_task, tasks)]
        else:
            records = [process_single_task(task) for task in tasks]
        return records

    def evaluate_tasks(self, tasks):
        records = self.process_tasks(tasks)
        assert len(records) == len(tasks)

        subset = None
        for index, record in enumerate(records):
            tasks[index]['record'] = record
            subset = subset or tasks[index]['subset']
            assert subset == tasks[index]['subset']

        total_metrics = average_metrics(records)
        total_metrics = {self.add_prefix(subset + '_' + k): float(v) for k, v in total_metrics.items()}
        return tasks, total_metrics

    def is_best(self, logs):
        loss = logs.get(self.monitor, np.inf)
        if self.save_best_only and self.best_loss < loss:
            return False
        else:
            self.best_loss = min(loss, self.best_loss)
            return True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.epoch += 1
        self.epochs_since_last_predict += 1

        if self.period and self.epochs_since_last_predict % self.period == 0:

            if not self.is_best(logs):
                return logs

            prediction_id = partial_format(self.template, epoch=epoch + 1, **logs)

            val_tasks = self.create_tasks(self.val_generator, 'val')
            if self.evaluate:
                val_tasks, val_metrics = self.evaluate_tasks(val_tasks)
                prediction_id = partial_format(prediction_id, **val_metrics)

                if not self.is_best(val_metrics):
                    return logs

            train_tasks = self.create_tasks(self.train_generator, 'train')
            if self.evaluate:
                train_tasks, train_metrics = self.evaluate_tasks(train_tasks)
                prediction_id = partial_format(prediction_id, **train_metrics)

                logs = dict(**logs, **train_metrics, **val_metrics)

            self.save_tasks(train_tasks + val_tasks, prediction_id, self.max_to_visualize)

            if mlflow.active_run():
                if self.evaluate:
                    [mlflow.log_metric(key=k, value=v, step=epoch + 1) for k, v in train_metrics.items()]
                    [mlflow.log_metric(key=k, value=v, step=epoch + 1) for k, v in val_metrics.items()]

                if self.save_artifacts:
                    mlflow.log_artifacts(self.run_dir, self.artifact_dir)

            self.epochs_since_last_predict = 0

        return logs

    def on_train_end(self, logs=None):
        # Check to not calculate metrics twice on_train_end
        if self.epochs_since_last_predict:
            self.period = 1
            self.on_epoch_end(self.epoch - 1, logs)

        test_tasks, test_metrics = self.create_tasks(self.test_generator, 'test')
        if self.evaluate:
            test_tasks, train_metrics = self.evaluate_tasks(test_tasks)

        self.save_tasks(test_tasks, prediction_id='test')

        if mlflow.active_run():
            if self.evaluate:
                mlflow.log_metrics(test_metrics)

            if self.save_artifacts:
                mlflow.log_artifacts(self.run_dir, self.artifact_dir)
