import os
import keras.backend as K
import keras_contrib


class CyclicLR(keras_contrib.callbacks.CyclicLR):

    def __init__(self,
                 base_lr,
                 max_lr,
                 step_size=1000,
                 mode='triangular',
                 gamma=1.,
                 scale=None,
                 freeze_epoch=None,
                 **kwargs):

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.freeze_epoch = freeze_epoch

        if self.mode == 'triangular':
            if scale is None:
                self.scale_fn = lambda x: 1.
            else:
                self.scale_fn = lambda x: 1 / (scale ** (x - 1))

            self.scale_mode = 'cycle'

        elif self.mode == 'exp_range':
            self.scale_fn = lambda x: self.gamma ** x
            self.scale_mode = 'iterations'
        else:
            raise ValueError(f'Unknown mode: "{self.mode}"')

        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.epoch = 1
        self.history = {}

        self._reset()

    def on_batch_end(self, batch_index, logs=None):
        logs = logs or {}

        prev_lr = K.get_value(self.model.optimizer.lr)
        new_lr = self.clr()

        self.trn_iterations += 1
        self.clr_iterations += 1

        k = prev_lr / new_lr
        if abs(1 - k) > K.epsilon():
            self.base_lr *= k
            self.max_lr *= k

        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1

        if self.epoch == self.freeze_epoch:
            self.max_lr = self.base_lr

        logs['lr'] = K.get_value(self.model.optimizer.lr)
        return logs

    def on_train_end(self, logs=None):
        return logs

    def __repr__(self):
        s = ['Cyclic LR',
             f'lr range=[{self.base_lr}, {self.max_lr}]',
             f'step_size={self.step_size}',
             f'mode="{self.mode}"',
             f'gamma={self.gamma}',
             f'freeze after {self.freeze_epoch} epochs' if self.freeze_epoch else 'no freeze']
        return ', '.join(s)
