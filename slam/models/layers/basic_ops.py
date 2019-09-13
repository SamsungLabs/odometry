import tensorflow as tf

from keras import backend as K
from keras.layers import Layer


class BasicOp(Layer):

    def __init__(self, axis=None, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis
        self.op = None

    def call(self, inputs):
        outputs = self.op(inputs, self.axis, keepdims=True)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.axis:
            if isinstance(self.axis, int):
                output_shape[self.axis] = 1
            else:
                for axis in self.axis:
                    output_shape[axis] = 1

        return tuple(output_shape)


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

    def __init__(self, q=90, axis=None, **kwargs):
        super().__init__(axis=axis, **kwargs)
        
        self.q = q

        self.op = lambda inputs, axis, keepdims: \
            tf.contrib.distributions.percentile(inputs,
                                                q=self.q,
                                                axis=axis,
                                                keep_dims=keepdims)


class Abs(Layer):

    def call(self, inputs):
        outputs = K.abs(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape
