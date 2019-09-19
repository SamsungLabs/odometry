import os
import keras

from .cyclic_lr_callback import CyclicLR

from .mlflow_logger_callback import MlflowLogger

from .model_checkpoint_callback import ModelCheckpoint

from .predict_callback import Predict

from .terminate_on_lr_callback import TerminateOnLR


__all__ = [
    'CyclicLR',
    'MlflowLogger',
    'ModelCheckpoint',
    'Predict',
    'TerminateOnLR'
]


def reset_params_on_batch_end(cls, batch, logs=None):
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


keras.callbacks.ProgbarLogger.on_batch_end = reset_params_on_batch_end


def update_logs_on_epoch_end(cls, epoch, logs=None):
    logs = logs or {}
    for callback in cls.callbacks:
        logs = callback.on_epoch_end(epoch, logs) or logs


def update_logs_on_train_end(cls, logs=None):
    logs = logs or {}
    for callback in cls.callbacks:
        logs = callback.on_train_end(logs) or logs


def update_logs_on_train_end(cls, logs=None):
    logs = logs or {}
    for callback in cls.callbacks:
        logs = callback.on_train_end(logs) or logs


keras.callbacks.CallbackList.on_epoch_end = update_logs_on_epoch_end
keras.callbacks.CallbackList.on_train_end = update_logs_on_train_end
