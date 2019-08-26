import tensorflow as tf
from functools import partial
from keras import backend as K
from keras.layers import Layer, Lambda, subtract, concatenate

from .basic_ops import Abs, Min, Max, Mean, Std, Percentile
from .functions import affine, inverse, divide, repeat, chunk, concat


def identity(inputs, scale=None, axis=None):
    return inputs, scale


def percentile_scale(inputs, axis=None, q=90):
    axis = axis or tuple(range(1, K.ndim(inputs)))

    num_channels = K.int_shape(inputs)[-1]
    if num_channels == 2:
        p = Percentile(q=q, axis=axis)(Abs()(inputs))
        outputs = divide(inputs, p)
        scale = Lambda(lambda x: x[:, 0, 0])(p)
    else:
        if num_channels == 4:
            size = [2, 2]
        elif num_channels == 7:
            size = [4, 3]
        else:
            raise ValueError

        inputs_translation, inputs_rotation = chunk(inputs, size=size)

        p_translation = Percentile(q=q, axis=axis)(Abs()(inputs_translation))
        p_rotation = Percentile(q=q, axis=axis)(Abs()(inputs_rotation))
        p = concat([repeat(p_rotation, rep=3, axis=-1),
                    repeat(p_translation, rep=3, axis=-1)])

        outputs = concat([divide(inputs_translation, p_translation),
                          divide(inputs_rotation, p_rotation)])

        scale = Lambda(lambda x: x[:, 0, 0])(p)
        scale = chunk(scale, n=6)

    return outputs, scale


def apply_scale(inputs, scale, bias=None):
    if bias is None:
        outputs = divide(inputs, scale)
    else:
        outputs = affine(inputs, weight=inverse(scale), bias=bias)

    scale = Lambda(lambda x: x[:, 0, 0])(scale)
    return outputs, scale


def project(inputs, axis=None, q=0, use_bias=True):
    axis = axis or tuple(range(1, K.ndim(inputs)))

    assert 0 <= q <= 100

    if q == 0:
        min_value = Min(axis=axis)(inputs)
        max_value = Max(axis=axis)(inputs)
    else:
        min_value = Percentile(q=q, axis=axis)(inputs)
        max_value = Percentile(q=100 - q, axis=axis)(inputs)

    value_range = subtract([max_value, min_value])

    return apply_scale(inputs,
                       scale=value_range,
                       bias=min_value if use_bias else None)


def normalize(inputs, axis=None, use_bias=True):
    axis = axis or tuple(range(1, K.ndim(inputs)))

    mean = Mean(axis=axis)(inputs)
    std = Std(axis=axis)(inputs)

    return apply_scale(inputs,
                       scale=std,
                       bias=mean if use_bias else None)


class Transform:

    def __init__(self, transform=None, agnostic=False, channel_wise=False):

        self.transform = transform
        self.agnostic = agnostic
        self.channel_wise = channel_wise

        self.transform_fn = self.get_transform_fn(transform)
        self.axis = (1, 2) if self.channel_wise else (1, 2, 3)

    @staticmethod
    def get_transform_fn(transform):
        if transform is None:
            return identity
        if transform == 'percentile_scale':
            return percentile_scale
        elif transform == 'range_scale':
            return partial(project, use_bias=False)
        elif transform == 'project':
            return project
        elif transform == 'standard_scale':
            return partial(normalize, use_bias=False)
        elif transform == 'normalize':
            return normalize
        elif transform == 'divide':
            def _divide(inputs, scale, axis):
                return divide(inputs, scale), scale
            return _divide
        else:
            raise ValueError(f'Unknown transform option: "{transform}"')

    def __call__(self, inputs):
        if (not self.transform) or self.agnostic:
            inputs, scale = concat(inputs), None
        else:
            inputs, scale = concat(inputs[:-1]), inputs[-1]

        if self.agnostic:
            inputs, scale = self.transform_fn(inputs, axis=self.axis)
        else:
            inputs, _ = self.transform_fn(inputs, scale=scale, axis=self.axis)

        return inputs, scale


def transform_inputs(inputs, transform=None, agnostic=False, channel_wise=False):
    return Transform(transform=transform,
                     agnostic=agnostic,
                     channel_wise=channel_wise)(inputs)
