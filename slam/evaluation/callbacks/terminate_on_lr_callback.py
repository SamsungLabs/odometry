import mlflow
import keras
from keras import backend as K


class TerminateOnLR(keras.callbacks.Callback):
    '''Stop training when a lr has become less than passed min_lr.
    # Arguments
        monitor: quantity to be monitored.
        verbose: verbosity mode.
        min_lr: Threshold value for the monitored quantity to reach.
            Training will stop if the monitored value is less than min_lr.
    '''
    def __init__(self, min_lr=None, verbose=0, **kwargs):

        super().__init__(**kwargs)

        self.min_lr = min_lr
        self.verbose = verbose
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)

        if logs['lr'] < self.min_lr:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        return logs

    def on_train_end(self, logs=None):
        logs = logs or {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)

        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch}: terminated on lr = {logs["lr"]} < {self.min_lr}')

        return logs
