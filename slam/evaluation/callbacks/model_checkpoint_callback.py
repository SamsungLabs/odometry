import os
import keras

from slam.utils import symlink


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    
    def __init__(self, filepath, **kwargs):

        super().__init__(filepath, **kwargs)

        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.epoch += 1
        if self.epochs_since_last_save + 1 >= self.period:
            self.last_file_path = self.filepath.format(epoch=epoch + 1, **logs)

        super().on_epoch_end(epoch, logs)
        return logs

    def on_train_end(self, logs=None):
        logs = logs or {}

        if self.epochs_since_last_save:
            self.save_best_only = False
            self.period = 1
            self.on_epoch_end(self.epoch - 1, logs)

        file_path = self.filepath.format(epoch=0, **logs)
        ext = os.path.splitext(file_path)[-1]
        file_path = os.path.join(os.path.dirname(file_path), 'final' + ext)

        symlink(self.last_file_path, file_path)

        return logs
