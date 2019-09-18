from functools import partial

import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense, concatenate, multiply
from keras.regularizers import l2

from slam.linalg import convert_euler_angles_to_rotation_matrix
from .functions import grid_sample
from .functions import concat


def construct_outputs(inputs,
                      regularization=0,
                      scale=None,
                      return_confidence=False):
    outputs = []

    for i, output_name in enumerate(('euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z')):
        x = inputs[i]

        output = Dense(1, kernel_regularizer=l2(regularization), name=output_name)(x)

        if scale is not None:
            s = scale[i] if isinstance(scale, list) else scale
            output = multiply([output, s])

        returned_values = [output]

        if scale is not None:
            returned_values.append(s)

        if return_confidence:
            confidence = Dense(1,
                               kernel_regularizer=l2(regularization),
                               kernel_initializer='glorot_normal',
                               trainable=False)(x)
            returned_values.append(confidence)

        output = concat(returned_values)
        outputs.append(output)

    return outputs


class DepthFlow(Layer):

    def __init__(self, **kwargs):
        self.batch_grid_x = None
        self.batch_grid_y = None

        super().__init__(*kwargs)

    def build(self, input_shape):
        assert input_shape[-1] >= 4

        height, width = input_shape[1], input_shape[2]

        xx = tf.expand_dims(tf.linspace(-1., 1., width), 1)
        yy = tf.expand_dims(tf.linspace(-1., 1., height), 1)

        grid_x, grid_y = tf.meshgrid(xx, yy)

        self.batch_grid_x = tf.expand_dims(grid_x, axis=0)
        self.batch_grid_y = tf.expand_dims(grid_y, axis=0)

        super().build(input_shape)

    def call(self, x):
        flow_x, flow_y = x[..., 0], x[..., 1]
        depth_first, depth_second = x[..., -2:-1], x[..., -1:]

        shifted_grid_x = self.batch_grid_x + flow_x
        shifted_grid_y = self.batch_grid_y + flow_y
        shifted_grid = tf.stack([shifted_grid_y, shifted_grid_x], axis=3)
        shifted_depth_second = grid_sample(depth_second, shifted_grid)

        depth_flow = shifted_depth_second - depth_first
        return depth_flow

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)


def depth_flow(inputs, **kwargs):
    return DepthFlow(**kwargs)(inputs)


class AddGrid(Layer):

    def __init__(self, f_x=1, f_y=1, c_x=0.5, c_y=0.5, **kwargs):

        """
        Args:
            f_x: focal distance along x-axis
            f_y: focal distance along y-axis
            c_x: coordinate of center pixel along x-axis
            c_y: coordinate of center pixel along y-axis
        """

        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

        self.grid = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        height, width = input_shape[1], input_shape[2]

        xx = tf.cast(tf.range(width), tf.float32) / width
        yy = tf.cast(tf.range(height), tf.float32) / height

        xx = (xx - self.c_x) / self.f_x
        yy = (yy - self.c_y) / self.f_y

        grid = tf.stack(tf.meshgrid(tf.expand_dims(xx, 1),
                                    tf.expand_dims(yy, 1)), axis=2)
        self.grid = tf.expand_dims(grid, axis=0)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
            stack input with [grid along x-axis, grid along y-axis] along channel axis
        """
        batch_size = tf.shape(inputs)[0]
        batched_grid = tf.tile(self.grid, [batch_size, 1, 1, 1])
        return tf.concat([inputs, batched_grid], axis=3)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] += 2
        return tuple(output_shape)


def add_grid(inputs, f_x=1, f_y=1, c_x=0.5, c_y=0.5, **kwargs):
    return AddGrid(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y, **kwargs)(inputs)


class FlowComposer(Layer):

    def __init__(self, intrinsics, **kwargs):
        self.translation = np.array([0., 0., 0.])
        self.depth = np.ones((intrinsics.height, intrinsics.width))
        self.intrinsics = intrinsics

        self.pixels_grid = np.c_[np.meshgrid(np.arange(0, self.intrinsics.width),
                                 np.arange(0, self.intrinsics.height))]
        self.pixels_grid = self.pixels_grid.astype(np.float64)

        self.pixels_normalized = self.intrinsics.forward(self.pixels_grid)
        ones = np.ones((1, ) + self.pixels_normalized.shape[1:])
        self.pixels_normalized = np.concatenate([self.pixels_normalized, ones], 0)
        super().__init__(**kwargs)

    def _create_gt_optical_flow_pair(depth, rotation_vector, gt_translation):
        R = convert_euler_angles_to_rotation_matrix(rotation_vector)
        t = gt_translation.reshape(3, -1)
        points1 = depth * self.pixels_normalized
        points2 = R.T @ (points1.reshape(3, -1) - t)
        points2 = points2.reshape((3, self.intrinsics.height, self.intrinsics.width))
        xy_pixels_from_rt = self.intrinsics.backward(points2[:2] / points2[2])
        gt_flow = (xy_pixels_from_rt - self.pixels_grid)
        gt_flow = np.transpose(gt_flow, (1, 2, 0))
        gt_flow[:,:,0] /= gt_flow.shape[1]
        gt_flow[:,:,1] /= gt_flow.shape[0]
        return gt_flow

    def _generate(self, rotation_vector):
        out = tf.py_func(self._create_gt_optical_flow_pair,
                          [self.depth, rotation_vector, self.translation],
                          tf.float32)
        return tf.reshape(out, [self.intrinsics.height, self.intrinsics.width, 2])

    def call(self, batch):
        return tf.map_fn(self._generate, batch, parallel_iterations=1, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (None, self.intrinsics.height, self.intrinsics.width, 2)
