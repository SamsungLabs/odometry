import os
import keras

from .mlflow_logger_callback import MlflowLogger

from .predict_callback import Predict

from .terminate_on_lr_callback import TerminateOnLR


__all__ = [
    'MlflowLogger',
    'Predict',
    'TerminateOnLR'
]


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


def on_train_end(self, logs=None):
    logs = logs or {}
    file_ext = os.path.splitext(self.filepath)[-1]
    file_path = os.path.join(os.path.dirname(self.filepath), 'final' + file_ext)
    if self.save_weights_only:
        self.model.save_weights(file_path, overwrite=True)
    else:
        self.model.save(file_path, overwrite=True)


keras.callbacks.ModelCheckpoint.on_train_end = on_train_end
