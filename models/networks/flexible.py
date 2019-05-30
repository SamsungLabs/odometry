import keras
from keras import backend as K

from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers import (Input, BatchNormalization, Flatten,
                          Dense, Layer, Cropping2D, Activation,
                          Concatenate)

from models.layers import (conv2d,
                           conv2d_transpose,
                           gated_conv2d,
                           gated_conv2d_transpose,
                           construct_fc,
                           construct_double_fc,
                           construct_outputs)



def construct_encoder(frames_concatenated,
                      use_gated_convolutions=False,
                      use_batchnorm=False,
                      strides=[2, 1, 4, 1],
                      dilation_rates=None,
                      kernel_sizes = [7, 5, 3, 3],
                      kernel_initializer='glorot_normal'):
    if dilation_rates is None:
        dilation_rates = [1] * len(strides)

    conv = gated_conv2d if use_gated_convolutions else conv2d

    inputs = frames_concatenated
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


def construct_flexible_model(imgs, 
                             frames_concatenated,
                             cropping=((0, 0), (0, 0)),
                             hidden_size=500,
                             regularization=0,
                             activation='relu',
                             kernel_initializer='glorot_normal',
                             use_gated_convolutions=True,
                             use_batchnorm=False,
                             strides=[2, 1, 4, 1],
                             dilation_rates=None,
                             kernel_sizes = [7, 5, 3, 3]):
    features = construct_encoder(frames_concatenated,
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

    model = Model(inputs=imgs, outputs=outputs)
    return model
