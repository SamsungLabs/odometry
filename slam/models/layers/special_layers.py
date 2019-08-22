import tensorflow as tf
from keras.layers import Layer, Dense, concatenate
from keras.regularizers import l2

from .functions import grid_sample


def construct_outputs(fc_rotation,
                      fc_translation,
                      regularization=0,
                      return_confidence=False):
    outputs = []
    for x, output_names in ((fc_rotation, ['euler_x', 'euler_y', 'euler_z']),
                            (fc_translation, ['t_x', 't_y', 't_z'])):
        for output_name in output_names:
            output = Dense(1, kernel_regularizer=l2(regularization), name=output_name)(x)

            if return_confidence:
                confidence = Dense(1,
                                   kernel_regularizer=l2(regularization),
                                   kernel_initializer='glorot_normal',
                                   trainable=False,
                                   name=f'{output_name}_confidence')(x)
                output = concatenate([output, confidence], name=f'{output_name}_with_confidence')

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
