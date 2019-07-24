import mlflow
import numpy as np
import keras


class MlflowLogger(keras.callbacks.Callback):

    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs['loss']
        val_loss = logs.get('val_loss', np.inf)
        if mlflow.active_run():
            mlflow.log_metric(f'{self.prefix}_train_loss' if self.prefix else 'train_loss', train_loss, step=epoch)
            mlflow.log_metric(f'{self.prefix}_val_loss' if self.prefix else 'val_loss', val_loss, step=epoch)
