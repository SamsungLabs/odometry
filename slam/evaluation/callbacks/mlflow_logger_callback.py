import os
import keras
import mlflow
import numpy as np


class MlflowLogger(keras.callbacks.Callback):

    def __init__(self,
                 ignore=None,
                 alias=None,
                 prefix=None,
                 run_dir=None,
                 artifact_dir=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.ignore = ignore or []
        self.alias = alias or {}
        self.prefix = prefix
        self.run_dir = run_dir
        self.artifact_dir = artifact_dir

        os.makedirs(self.run_dir, exist_ok=True)
        self.save_artifacts = self.run_dir and self.artifact_dir

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if mlflow.active_run():
            for key, value in dict({'epoch': epoch + 1, **logs}).items():
                if key in self.ignore:
                    continue

                name = self.alias.get(key, None) or key
                if self.prefix:
                    name = self.prefix + '_' + name

                mlflow.log_metric(name, float(value), step=epoch)

            if self.save_artifacts:
                mlflow.log_artifacts(self.run_dir, self.artifact_dir)

        return logs

    def on_train_end(self, logs=None):
        if mlflow.active_run() and self.save_artifacts:
            mlflow.log_artifacts(self.run_dir, self.artifact_dir)

        return logs
