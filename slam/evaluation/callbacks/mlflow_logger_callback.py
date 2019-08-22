import mlflow
import numpy as np
import keras


class MlflowLogger(keras.callbacks.Callback):

    def __init__(self, prefix=None):
        super().__init__()
        self.prefix = prefix
        self.add_prefix = lambda k: (self.prefix + '_' + k) if self.prefix else k

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if mlflow.active_run():
            train_loss = logs['loss']
            val_loss = logs.get('val_loss', np.inf)

            mlflow.log_metric(self.add_prefix('train_loss'), float(train_loss), step=epoch)
            mlflow.log_metric(self.add_prefix('val_loss'), float(val_loss), step=epoch)

        return logs
