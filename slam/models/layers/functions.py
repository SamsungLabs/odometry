import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer, Lambda, concatenate


def concat(inputs, **kwargs):
    if len(inputs) == 1:
        return inputs[0]
    return concatenate(inputs, **kwargs)


class Clip(Layer):

    def call(self, inputs, **kwargs):
        outputs = K.clip(K.abs(inputs), K.epsilon(), None) * K.sign(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def clip(inputs, **kwargs):
    return Clip(**kwargs)(inputs)


class Inverse(Layer):

    def call(self, inputs, **kwargs):
        outputs = 1. / clip(inputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def inverse(inputs, **kwargs):
    return Inverse(**kwargs)(inputs)


class Repeat(Layer):

    def __init__(self, rep=1, axis=-1, **kwargs):
        super().__init__(**kwargs)

        self.rep = rep
        self.axis = axis

    def call(self, inputs, **kwargs):
        outputs = K.repeat_elements(inputs, rep=self.rep, axis=self.axis)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] *= self.rep
        return tuple(output_shape)


def repeat(inputs, rep=1, axis=-1, **kwargs):
    return Repeat(rep=rep, axis=axis, **kwargs)(inputs)


def expand_as(inputs, target):
    for _ in range(K.ndim(target) - K.ndim(inputs)):
        inputs = K.expand_dims(inputs, -1)
    return inputs


class Affine(Layer):

    def call(self, inputs, **kwargs):
        x, weight, bias = inputs
        outputs = x * expand_as(weight, x) + bias
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def affine(inputs, weight, bias, **kwargs):
    return Affine(**kwargs)([inputs, weight, bias])


class Divide(Layer):

    def call(self, inputs, **kwargs):
        x, divider = inputs
        outputs = x / expand_as(clip(divider), x)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


def divide(inputs, scale, **kwargs):
    return Divide(**kwargs)([inputs, scale])


def chunk(inputs, n=None, size=None, names=None):
    assert n or size

    dim = K.int_shape(inputs)[-1]
    if n:
        indices = np.linspace(0, dim, num=n + 1).astype(int)[1:]
    elif isinstance(size, int):
        assert dim % size == 0
        indices = list(range(size, dim + 1, size))
    else:
        indices = np.cumsum(size)

    names = names or [None] * len(indices)

    assert len(names) == len(indices)

    chunks = []
    first_index = 0
    for last_index, name in zip(indices, names):
        print(f'chunk [{first_index}, {last_index}]')
        chunks.append(Lambda(lambda x: x[..., first_index:last_index], name=name)(inputs))
        first_index = last_index

    return chunks


def grid_sample(x, shifted_grid):
    input_shape = K.shape(x)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]

    # Find interpolation sides
    i, j = shifted_grid[..., 0], shifted_grid[..., 1]

    i = tf.cast(height - 1, shifted_grid.dtype) * (i + 1) / 2
    j = tf.cast(width - 1, shifted_grid.dtype) * (j + 1) / 2

    i_floor = tf.cast(tf.floor(i), tf.int32)
    j_floor = tf.cast(tf.floor(j), tf.int32)

    i_ceil = tf.clip_by_value(i_floor + 1, 0, height - 1)
    i_floor = tf.clip_by_value(i_floor, 0, height - 1)

    j_ceil = tf.clip_by_value(j_floor + 1, 0, width - 1)
    j_floor = tf.clip_by_value(j_floor, 0, width - 1)

    # Gather pixel values
    num_repeats = tf.range(batch_size)[:, tf.newaxis, tf.newaxis]
    shape = tf.concat([[1], tf.shape(i)[1:]], axis=0)
    n_idx = tf.tile(num_repeats, shape)

    q_11 = tf.gather_nd(x, tf.stack([n_idx, i_floor, j_floor, n_idx], axis=-1))
    q_12 = tf.gather_nd(x, tf.stack([n_idx, i_floor, j_ceil, n_idx], axis=-1))
    q_21 = tf.gather_nd(x, tf.stack([n_idx, i_ceil, j_floor, n_idx], axis=-1))
    q_22 = tf.gather_nd(x, tf.stack([n_idx, i_ceil, j_ceil, n_idx], axis=-1))

    # Interpolation coefficients
    dtype = x.dtype
    distance_i_floor = tf.cast(i, dtype) - tf.cast(i_floor, dtype)
    weight_i_floor = 1 - distance_i_floor
    distance_j_floor = tf.cast(j, dtype) - tf.cast(j_floor, dtype)
    weight_j_floor = 1 - distance_j_floor

    # Compute interpolations
    q_i1 = q_11 * weight_i_floor + q_21 * (1 - weight_i_floor)
    q_i2 = q_12 * weight_i_floor + q_22 * (1 - weight_i_floor)
    q_ij = q_i2 * weight_j_floor + q_i2 * (1 - weight_j_floor)
    q_ij = tf.expand_dims(q_ij, -1)
    return q_ij
