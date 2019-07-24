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
    def __init__(self,
                 min_lr=None,
                 verbose=0):
        super(TerminateOnLR, self).__init__()
        self.verbose = verbose
        self.min_lr = min_lr
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        mlflow.log_metric('lr', float(lr), step=epoch) if mlflow.active_run() else None
        if lr < self.min_lr:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            lr = K.get_value(self.model.optimizer.lr)
            print(f'Epoch {self.stopped_epoch}: terminated on lr = {lr} < {self.min_lr}')
