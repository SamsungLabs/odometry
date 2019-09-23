import tensorflow as tf

from keras import backend as K
from keras.layers import Layer


class BasicOp(Layer):

    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis
        self.op = None

    def call(self, inputs):
        axis = self.axis or list(range(1, K.ndim(inputs)))
        outputs = self.op(inputs, axis, keepdims=True)
        return outputs

    def compute_output_shape(self, input_shape):
        axis = self.axis or list(range(1, len(input_shape)))
        output_shape = list(input_shape)
        if isinstance(axis, int):
            output_shape[axis] = 1
        else:
            for ax in axis:
                output_shape[ax] = 1

        return tuple(output_shape)

    def get_config(self):
        config = super(BasicOp, self).get_config()
        config['axis'] = self.axis
        return config


class Min(BasicOp):

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.op = K.min


class Max(BasicOp):

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.op = K.max


class Mean(BasicOp):

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.op = K.mean


class Std(BasicOp):

    def __init__(self, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.op = K.std


class Percentile(BasicOp):

    def __init__(self, q=50, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)

        self.q = q

        self.op = lambda inputs, axis, keepdims: \
            tf.contrib.distributions.percentile(inputs,
                                                q=self.q,
                                                axis=axis,
                                                keep_dims=keepdims)

    def get_config(self):
        config = super(BasicOp, self).get_config()
        config['q'] = self.q
        return config


class Abs(Layer):

    def call(self, inputs):
        outputs = K.abs(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
