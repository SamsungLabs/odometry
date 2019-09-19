import os
import keras.backend as K
import keras_contrib


class CyclicLR(keras_contrib.callbacks.CyclicLR):

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}

        cyclic_lr = self.clr()
        self.trn_iterations += 1
        self.clr_iterations += 1

        actual_lr = K.get_value(self.model.optimizer.lr)

        k = actual_lr / cyclic_lr

        if abs(1 - k) > K.epsilon():
            self.base_lr *= k
            self.max_lr *= k

        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        return logs

    def on_train_end(self, logs=None):
        return logs
