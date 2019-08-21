import mlflow
import numpy as np
import keras


class MlflowLogger(keras.callbacks.Callback):

    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if mlflow.active_run():
            train_loss = logs['loss']
            val_loss = logs.get('val_loss', np.inf)
            mlflow.log_metric(((self.prefix + '_') if self.prefix else '') + 'train_loss', float(train_loss), step=epoch)
            mlflow.log_metric(((self.prefix + '_') if self.prefix else '') + 'val_loss', float(val_loss), step=epoch)

        return logs
