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


def conv2d(inputs, filters, kernel_size, activation='linear', batch_norm=False, **kwargs):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, **kwargs)(inputs)
    if batch_norm:
        conv = BatchNormalization()(conv)
    activation = activ(conv, activation)
    return activation


def conv2d_transpose(inputs, filters, kernel_size, activation='linear', batch_norm=False, **kwargs):
    conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, **kwargs)(inputs)
    if batch_norm:
        conv = BatchNormalization()(conv)
    activation = activ(conv, activation)
    return activation


def gated_conv2d(inputs, filters, kernel_size, activation='linear', name=None, batch_norm=False, **kwargs):
    if name is None:
        f_name, g_name = None, None
    else:
        f_name, g_name = '{}_feature'.format(name), '{}_gate'.format(name)

    f = conv2d(inputs, filters, kernel_size, activation=activation, name=f_name, batch_norm=batch_norm, **kwargs)
    g = conv2d(inputs, filters, kernel_size, activation='sigmoid', name=g_name, batch_norm=batch_norm, **kwargs)
    return multiply([f, g])


def gated_conv2d_transpose(inputs, filters, kernel_size, activation='linear', name=None, batch_norm=False, **kwargs):
    if name is None:
        f_name, g_name = None, None
    else:
        f_name, g_name = '{}_feature'.format(name), '{}_gate'.format(name)

    f = conv2d_transpose(inputs, filters, kernel_size, activation=activation, name=f_name, batch_norm=batch_norm, **kwargs)
    g = conv2d_transpose(inputs, filters, kernel_size, activation='sigmoid', name=g_name, batch_norm=batch_norm, **kwargs)
    return multiply([f, g])


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
