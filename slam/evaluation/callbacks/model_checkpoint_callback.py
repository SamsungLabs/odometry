import os
import shutil
import keras


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):

    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)

        self.epoch = 0
        self.last_file_path = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch

        if self.epochs_since_last_save + 1 >= self.period:
            self.last_file_path = self.filepath.format(epoch=epoch + 1, **logs)
            os.makedirs(os.path.dirname(self.last_file_path), exist_ok=True)

        super().on_epoch_end(epoch, logs)

        return logs

    def on_train_end(self, logs=None):
        logs = logs or {}

        if self.save_best_only:
            file_dir, file_name = os.path.split(self.filepath)
            file_name, ext = os.path.splitext(file_name)
            self.filepath = os.path.join(file_dir, 'final' + ext)

        reuse = self.epochs_since_last_save == 0 and self.last_file_path is not None
        if reuse:
            file_path = self.filepath.format(epoch=self.epoch + 1, **logs)
            if self.last_file_path != file_path:
                shutil.copyfile(self.last_file_path, file_path)
        else:
            self.save_best_only = False
            self.period = 1
            self.on_epoch_end(self.epoch - 1, logs)

        return logs
