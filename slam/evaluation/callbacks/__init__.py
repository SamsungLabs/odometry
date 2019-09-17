import os
import keras
import keras.backend as K
import keras_contrib

from .mlflow_logger_callback import MlflowLogger

from .predict_callback import Predict

from .terminate_on_lr_callback import TerminateOnLR


__all__ = [
    'MlflowLogger',
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


def save_weights_on_train_end(cls, logs=None):
    logs = logs or {}
    file_path = cls.filepath.format(epoch='final', **logs)
    if cls.save_weights_only:
        cls.model.save_weights(file_path, overwrite=True)
    else:
        cls.model.save(file_path, overwrite=True)


keras.callbacks.ModelCheckpoint.on_train_end = save_weights_on_train_end


def update_logs_on_epoch_end(cls, epoch, logs=None):
    logs = logs or {}
    for callback in cls.callbacks:
        logs = callback.on_epoch_end(epoch, logs) or logs


def update_logs_on_train_end(cls, logs=None):
    logs = logs or {}
    for callback in cls.callbacks:
        logs = callback.on_train_end(logs) or logs


keras.callbacks.CallbackList.on_epoch_end = update_logs_on_epoch_end
keras.callbacks.CallbackList.on_train_end = update_logs_on_train_end


def adjust_lr_on_batch_end(cls, epoch, logs=None):
    logs = logs or {}

    cyclic_lr = cls.clr()
    cls.trn_iterations += 1
    cls.clr_iterations += 1

    actual_lr = K.get_value(cls.model.optimizer.lr)

    k = actual_lr / cyclic_lr

    if abs(1 - k) > K.epsilon():
        cls.base_lr *= k
        cls.max_lr *= k

    K.set_value(cls.model.optimizer.lr, cls.clr())

    cls.history.setdefault('lr', []).append(K.get_value(cls.model.optimizer.lr))
    cls.history.setdefault('iterations', []).append(cls.trn_iterations)

    for k, v in logs.items():
        cls.history.setdefault(k, []).append(v)


keras_contrib.callbacks.CyclicLR.on_batch_end = adjust_lr_on_batch_end
