import os
import keras

from slam.utils import symlink


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    
    def __init__(self, filepath, **kwargs):

        super().__init__(filepath, **kwargs)

        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        super().on_epoch_end(epoch, logs)

        return logs

    def on_train_end(self, logs=None):
        logs = logs or {}

        if self.epochs_since_last_save:
            self.save_best_only = False
            self.period = 1
            self.on_epoch_end(self.epoch - 1, logs)

        return logs
