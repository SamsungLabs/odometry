from keras.layers import (Conv2D,
                          Conv2DTranspose,
                          Dense,
                          BatchNormalization,
                          Activation,
                          PReLU,
                          LeakyReLU,
                          multiply)
from keras.regularizers import l2


def activ(inputs, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU()(inputs)
    elif activation == 'p_relu':
        activation = PReLU()(inputs)
    else:
        activation = Activation(activation)(inputs)
    return activation


def _conv2d(conv_layer,
            inputs,
            filters,
            kernel_size,
            activation='linear',
            batch_norm=False,
            **kwargs):
    conv = conv_layer(filters=filters, kernel_size=kernel_size, **kwargs)(inputs)
    if batch_norm:
        conv = BatchNormalization()(conv)
    activation = activ(conv, activation)
    return activation


def conv2d(inputs,
           filters,
           kernel_size,
           activation='linear',
           batch_norm=False,
           **kwargs):
    return _conv2d(Conv2D,
                   inputs=inputs,
                   filters=filters,
                   kernel_size=kernel_size,
                   activation=activation,
                   batch_norm=batch_norm,
                   **kwargs)


def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     activation='linear',
                     batch_norm=False,
                     **kwargs):
    return _conv2d(Conv2DTranspose,
                   inputs=inputs,
                   filters=filters,
                   kernel_size=kernel_size,
                   activation=activation,
                   batch_norm=batch_norm,
                   **kwargs)


def _gated_conv2d(conv_layer,
                  inputs,
                  filters,
                  kernel_size,
                  activation='linear',
                  batch_norm=False,
                  name=None,
                  **kwargs):
    feature = conv_layer(inputs,
                         filters=filters,
                         kernel_size=kernel_size,
                         activation=activation,
                         name=(name + '_feature') if name else None,
                         batch_norm=batch_norm,
                         **kwargs)
    gate = conv_layer(inputs,
                      filters=filters,
                      kernel_size=kernel_size,
                      activation='sigmoid',
                      name=(name + '_gate') if name else None,
                      batch_norm=batch_norm,
                      **kwargs)
    return multiply([feature, gate])


def gated_conv2d(inputs,
                 filters,
                 kernel_size,
                 activation='linear',
                 batch_norm=False,
                 name=None,
                 **kwargs):
    return _gated_conv2d(conv2d,
                         inputs=inputs,
                         filters=filters,
                         kernel_size=kernel_size,
                         activation=activation,
                         batch_norm=batch_norm,
                         name=name,
                         **kwargs)


def gated_conv2d_transpose(inputs,
                           filters,
                           kernel_size,
                           activation='linear',
                           batch_norm=False,
                           name=None,
                           **kwargs):
    return _gated_conv2d(conv2d_transpose,
                         inputs=inputs,
                         filters=filters,
                         kernel_size=kernel_size,
                         activation=activation,
                         batch_norm=batch_norm,
                         name=name,
                         **kwargs)


def dense(inputs,
          output_size,
          layers_num=1,
          regularization=0,
          activation='relu',
          kernel_initializer='glorot_normal',
          name=None):

    if isinstance(output_size, int):
        output_size = [output_size] * layers_num

    assert len(output_size) == layers_num

    x = inputs
    for i in range(layers_num):
        suffix = f'_{i}' if layers_num > 1 else ''
        x = Dense(output_size[i],
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(regularization),
                  bias_regularizer=l2(regularization),
                  name=(name + suffix) if name else None)(x)
        x = activ(x, activation)

    outputs = x
    return outputs
