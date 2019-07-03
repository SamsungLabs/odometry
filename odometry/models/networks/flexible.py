from keras.layers.merge import concatenate
from keras.layers import Flatten
import mlflow

from odometry.models.layers import (concat,
                                    conv2d,
                                    gated_conv2d,
                                    construct_double_fc,
                                    construct_outputs)


def construct_encoder(inputs,
                      use_gated_convolutions=False,
                      use_batchnorm=False,
                      strides=[2, 1, 4, 1],
                      dilation_rates=None,
                      kernel_sizes = [7, 5, 3, 3],
                      kernel_initializer='glorot_normal'):
    if dilation_rates is None:
        dilation_rates = [1] * len(strides)

    conv = gated_conv2d if use_gated_convolutions else conv2d

    for i, (stride, dilation_rate, kernel_size) in enumerate(zip(strides, dilation_rates, kernel_sizes)):
        inputs = conv(inputs,
                      64,
                      kernel_size=kernel_size,
                      strides=stride,
                      dilation_rate=dilation_rate,
                      padding='same',
                      batchnorm=True if use_batchnorm and i == 0 else False,
                      activation='relu',
                      kernel_initializer=kernel_initializer)
   
    flatten1 = Flatten()(inputs)
    return flatten1


def construct_flexible_model(inputs,
                             cropping=((0, 0), (0, 0)),
                             hidden_size=500,
                             regularization=0,
                             activation='relu',
                             kernel_initializer='glorot_normal',
                             use_gated_convolutions=True,
                             use_batchnorm=False,
                             strides=[2, 1, 4, 1],
                             dilation_rates=None,
                             kernel_sizes=[7, 5, 3, 3]):

    mlflow.log_param('model.name', 'Flexible')
    mlflow.log_params({'model.' + k: repr(v) for k, v in locals().items() if 'inputs' not in k})

    inputs = concat(inputs)
    features = construct_encoder(inputs,
                                 use_gated_convolutions=use_gated_convolutions,
                                 use_batchnorm=use_batchnorm,
                                 strides=strides,
                                 dilation_rates=dilation_rates,
                                 kernel_sizes = kernel_sizes,
                                 kernel_initializer=kernel_initializer)
    fc2_rotation = construct_double_fc(features,
                                       hidden_size=hidden_size,
                                       regularization=regularization,
                                       activation=activation,
                                       kernel_initializer=kernel_initializer,
                                       name='rotation')
    fc2_translation = construct_double_fc(features,
                                          hidden_size=hidden_size,
                                          regularization=regularization,
                                          activation=activation,
                                          kernel_initializer=kernel_initializer,
                                          name='translation')
    outputs = construct_outputs(fc2_rotation, fc2_translation, regularization=regularization)
    return outputs
