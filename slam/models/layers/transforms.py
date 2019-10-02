import tensorflow as tf
from functools import partial
from keras import backend as K
from keras.layers import Layer, Lambda, subtract, concatenate

from .basic_ops import Abs, Min, Max, Mean, Std, Percentile
from .functions import affine, inverse, divide, repeat, chunk, concat


def identity(inputs, scale=None, axis=None):
    return inputs, scale


def percentile_scale(inputs, axis=None, q=50):
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

    def __init__(self, transform=None, agnostic=True, channel_wise=False):

        self.transform = transform
        self.agnostic = agnostic
        self.channel_wise = channel_wise
        self.axis = (1, 2) if self.channel_wise else (1, 2, 3)

    def __call__(self, inputs):
        if self.transform is None or self.agnostic:
            inputs, scale = concat(inputs), None
        else:
            inputs, scale = concat(inputs[:-1]), inputs[-1]

        if self.transform == 'percentile_scale':
            transformed_inputs, scale_from_inputs = percentile_scale(inputs, axis=self.axis, q=90)
        elif self.transform == 'absmean_scale':
            transformed_inputs, scale_from_inputs = percentile_scale(inputs, axis=self.axis, q=50)
        elif self.transform == 'range_scale':
            transformed_inputs, scale_from_inputs = project(inputs, axis=self.axis, q=10, use_bias=False)
        elif self.transform == 'project':
            transformed_inputs, scale_from_inputs = project(inputs, axis=self.axis, q=10)
        elif self.transform == 'standard_scale':
            transformed_inputs, scale_from_inputs = normalize(inputs, axis=self.axis, use_bias=False)
        elif self.transform == 'normalize':
            transformed_inputs, scale_from_inputs = normalize(inputs, axis=self.axis)
        elif self.transform == 'divide':
            transformed_inputs = divide(inputs, scale)
            scale_from_inputs = scale
        elif self.transform is None:
            transformed_inputs = inputs
            scale_from_inputs = scale
        else:
            raise ValueError(f'Unknown transform option: "{self.transform}"')

        if self.agnostic:
            return transformed_inputs, scale_from_inputs
        else:
            return transformed_inputs, scale


def transform_inputs(inputs, transform=None, agnostic=True, channel_wise=False):
    return Transform(transform=transform,
                     agnostic=agnostic,
                     channel_wise=channel_wise)(inputs)
