import tensorflow as tf

from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Layer, Dense, Activation, Multiply, Concatenate
from keras.layers.merge import concatenate
from keras.regularizers import l2


def activ(inputs, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU()(inputs)
    elif activation == 'p_relu':
        activation = PReLU()(inputs)
    else:
        activation = Activation(activation)(inputs)
    return activation


def concat(inputs):
    if len(inputs) == 1:
        return inputs[0]
    return concatenate(inputs)


def conv2d(inputs, filters, kernel_size, activation='linear', batchnorm=False, **kwargs):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)(inputs)
    if batchnorm:
        conv = BatchNormalization()(conv)
    activation = activ(conv, activation)
    return activation


def conv2d_transpose(inputs, filters, kernel_size, activation='linear', batchnorm=False, **kwargs):
    conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, **kwargs)(inputs)
    if batchnorm:
        conv = BatchNormalization()(conv)
    activation = activ(conv, activation)
    return activation


def gated_conv2d(inputs, filters, kernel_size, activation='linear', name=None, batchnorm=False, **kwargs):
    if name is None:
        f_name, g_name = None, None
    else:
        f_name, g_name = '{}_feature'.format(name), '{}_gate'.format(name)

    f = conv2d(inputs, filters, kernel_size, activation=activation, name=f_name, batchnorm=batchnorm, **kwargs)
    g = conv2d(inputs, filters, kernel_size, activation='sigmoid', name=g_name, batchnorm=batchnorm, **kwargs)
    return Multiply()([f, g])


def gated_conv2d_transpose(inputs, filters, kernel_size, activation='linear', name=None, batchnorm=False, **kwargs):
    if name is None:
        f_name, g_name = None, None
    else:
        f_name, g_name = '{}_feature'.format(name), '{}_gate'.format(name)

    f = conv2d_transpose(inputs, filters, kernel_size, activation=activation, name=f_name, batchnorm=batchnorm, **kwargs)
    g = conv2d_transpose(inputs, filters, kernel_size, activation='sigmoid', name=g_name, batchnorm=batchnorm, **kwargs)
    return Multiply()([f, g])


def construct_fc(inputs,
                 hidden_size=1000,
                 regularization=0,
                 activation='relu',
                 kernel_initializer='glorot_normal',
                 name=None):
    fc = Dense(hidden_size, kernel_initializer=kernel_initializer,
               kernel_regularizer=l2(regularization),
               bias_regularizer=l2(regularization), name=name)(inputs)
    activation = activ(fc, activation)
    return activation


def construct_double_fc(inputs,
                        hidden_size,
                        regularization=0,
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        name=None):
    names = ['fc1', 'fc2']
    if name is not None:
        names = [fc_name + '_' + name for fc_name in names]

    fc1 = construct_fc(inputs, hidden_size=hidden_size,
                       regularization=regularization, activation=activation,
                       kernel_initializer=kernel_initializer, name=names[0])
    fc2 = construct_fc(fc1, hidden_size=hidden_size,
                       regularization=regularization, activation=activation,
                       kernel_initializer=kernel_initializer, name=names[1])
    return fc2


def construct_outputs(fc_rotation, fc_translation, regularization=0):
    outputs = []
    for x, output_names in ((fc_rotation, ['euler_x', 'euler_y', 'euler_z']),
                            (fc_translation, ['t_x', 't_y', 't_z'])):
        for output_name in output_names:
            output = Dense(1, kernel_regularizer=l2(regularization), name=output_name)(x)
            outputs.append(output)

    return outputs


def construct_outputs_with_confidences(outputs,
                                       fc_rotation,
                                       fc_translation,
                                       regularization=0,
                                       kernel_initializer='glorot_normal'):
    confidences = []
    names = []
    for x, output_names in ((fc_rotation, ['euler_x', 'euler_y', 'euler_z']),
                            (fc_translation, ['t_x', 't_y', 't_z'])):
        for output_name in output_names:
            confidence = Dense(1,
                               activation='relu',
                               kernel_regularizer=l2(regularization),
                               kernel_initializer=kernel_initializer,
                               trainable=False,
                               name=f'{output_name}_confidence')(x)
            confidences.append(confidence)
            names.append(f'{output_name}_with_confidence')

    outputs_with_confidences = [Concatenate(name=name)([output, confidence])
                                for output, confidence, name in zip(outputs, confidences, names)]
    return outputs_with_confidences


class ConstLayer(Layer):
    def __init__(self, value, **kwargs):
        self.value = value
        super(ConstLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConstLayer, self).build(input_shape)

    def call(self, x):
        return K.expand_dims(
            K.max(K.ones_like(x) * self.value, axis=(1,2,3)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


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


class AssociationLayer(Layer):
    def __init__(self, **kwargs):
        super(AssociationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        height, width = input_shape[1], input_shape[2]
        xx = tf.expand_dims(tf.linspace(-1., 1., width), 1)
        yy = tf.expand_dims(tf.linspace(-1., 1., height), 1)
        grid_x, grid_y = tf.meshgrid(xx, yy)
        self.batched_grid_x = tf.expand_dims(grid_x, axis=0)
        self.batched_grid_y = tf.expand_dims(grid_y, axis=0)
        
        super(AssociationLayer, self).build(input_shape)

    def call(self, x):
        flow_x, flow_y = x[..., 0], x[..., 1]
        depth_first, depth_second = x[..., -2:-1], x[..., -1:]
        
        shifted_grid_x = self.batched_grid_x + flow_x
        shifted_grid_y = self.batched_grid_y + flow_y
        shifted_grid = tf.stack([shifted_grid_y, shifted_grid_x], axis=3)
        shifted_depth_second = grid_sample(depth_second, shifted_grid)

        flow_depth = shifted_depth_second - depth_first
        return flow_depth

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)


class AddGridLayer(Layer):

    def __init__(self, f_x=1, f_y=1, c_x=0.5, c_y=0.5, **kwargs):

        '''
        Args:
            f_x: focal distance along x-axis
            f_y: focal distance along y-axis
            c_x: coordinate of center pixel along x-axis
            c_y: coordinate of center pixel along y-axis
        '''

        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

        super(AddGridLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]

        dtype = tf.float32
        xx = tf.cast(tf.range(width), dtype) / width
        yy = tf.cast(tf.range(height), dtype) / height

        xx = (xx - self.c_x) / self.f_x
        yy = (yy - self.c_y) / self.f_y

        grid = tf.stack(tf.meshgrid(tf.expand_dims(xx, 1),
                                    tf.expand_dims(yy, 1)), axis=2)
        self.grid = tf.expand_dims(grid, axis=0)

        super(AddGridLayer, self).build(input_shape)

    def call(self, x):
        '''
            stack input with [grid along x-axis, grid along y-axis] along channel axis
        '''
        batch_size = tf.shape(x)[0]
        batched_grid = tf.tile(self.grid, [batch_size, 1, 1, 1])
        return tf.concat([x, batched_grid], axis=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] + 2)
