from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Cropping2D, Activation

from odometry.models.layers import (concat,
                                    conv2d,
                                    conv2d_transpose,
                                    construct_fc,
                                    construct_double_fc,
                                    construct_outputs)


def construct_encoder(inputs,
                      strides=[2, 1, 4, 1],
                      dilation_rates=[1, 1, 1, 1],
                      kernel_initializer='glorot_normal'):
    conv1 = conv2d(inputs, 64, kernel_size=7, strides=strides[0], dilation_rate=dilation_rates[0],
                   padding='same', activation='relu',
                   kernel_initializer=kernel_initializer)

    conv2 = conv2d(conv1, 64, kernel_size=5, strides=strides[1], dilation_rate=dilation_rates[1],
                   padding='same', activation='relu',
                   kernel_initializer=kernel_initializer)

    conv3 = conv2d(conv2, 64, kernel_size=3, strides=strides[2], dilation_rate=dilation_rates[2],
                   padding='same', activation='relu',
                   kernel_initializer=kernel_initializer)

    conv4 = conv2d(conv3, 64, kernel_size=3, strides=strides[3], dilation_rate=dilation_rates[3],
                   padding='same', activation='relu',
                   kernel_initializer=kernel_initializer)

    pool = MaxPooling2D(pool_size=2, strides=2)(conv4)
    flatten1 = Flatten()(conv3)
    flatten2 = Flatten()(pool)
    merged = concatenate([flatten1, flatten2], axis=1)
    return merged, conv4


def construct_flow_decoder(conv4,
                           cropping=((0, 0), (0, 0)),
                           output_channels=2):
    upsampling1 = conv2d_transpose(conv4, 6, kernel_size=3, strides=4, padding='same', activation='relu')
    cropping = Cropping2D(cropping=cropping)(upsampling1)
    upsampling2 = conv2d_transpose(cropping, 2, kernel_size=1, strides=2, padding='same', name='flow')
    return upsampling2


def construct_st_vo_model(inputs,
                          kernel_initializer='glorot_normal'):
    inputs = concat(inputs)
    conv1 = Conv2D(64, kernel_size=3, strides=2,
                   kernel_initializer=kernel_initializer, name='conv1')(inputs)
    pool1 = MaxPooling2D(pool_size=4, strides=4, name='pool1')(conv1)
    conv2 = Conv2D(20, kernel_size=3,
                   kernel_initializer=kernel_initializer, name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2, name='pool2')(conv2)

    flatten1 = Flatten(name='flatten1')(pool1)
    flatten2 = Flatten(name='flatten2')(pool2)
    merged = concatenate([flatten1, flatten2], axis=1)
    activation = Activation('relu')(merged)
    fc = construct_fc(activation, kernel_initializer=kernel_initializer, name='fc')
    outputs = construct_outputs(fc, fc)
    return outputs


def construct_ls_vo_model(inputs,
                          cropping=((0, 0), (0, 0)),
                          hidden_size=1000,
                          regularization=0,
                          kernel_initializer='glorot_normal'):
    inputs = concat(inputs)
    features, bottleneck = construct_encoder(inputs,
                                             kernel_initializer=kernel_initializer)
    reconstructed_flow = construct_flow_decoder(bottleneck,
                                                cropping=cropping,
                                                output_channels=inputs_concatenated.shape[-1].value)
    fc2 = construct_double_fc(features,
                              hidden_size=hidden_size,
                              regularization=regularization,
                              kernel_initializer=kernel_initializer)
    outputs = construct_outputs(fc2, fc2, regularization=regularization) + [reconstructed_flow]
    model = Model(inputs=inputs, outputs=outputs)
    return model


def construct_ls_vo_rt_model(inputs,
                             cropping=((0, 0), (0, 0)),
                             hidden_size=500,
                             regularization=0,
                             kernel_initializer='glorot_normal'):
    inputs = concat(inputs)
    features, bottleneck = construct_encoder(inputs,
                                             kernel_initializer=kernel_initializer)
    reconstructed_flow = construct_flow_decoder(bottleneck,
                                                cropping=cropping,
                                                output_channels=inputs_concatenated.shape[-1].value)
    fc2_rotation = construct_double_fc(features,
                                       hidden_size=hidden_size,
                                       regularization=regularization,
                                       kernel_initializer=kernel_initializer,
                                       name='rotation')
    fc2_translation = construct_double_fc(features, 
                                          hidden_size=hidden_size,
                                          regularization=regularization,
                                          kernel_initializer=kernel_initializer,
                                          name='translation')
    outputs = construct_outputs(fc2_rotation, fc2_translation, regularization=regularization) + [reconstructed_flow]
    return outputs


def construct_ls_vo_rt_no_decoder_model(inputs,
                                        hidden_size=500,
                                        regularization=0,
                                        kernel_initializer='glorot_normal'):
    inputs = concat(inputs)
    features, _ = construct_encoder(inputs,
                                    kernel_initializer=kernel_initializer)
    fc2_rotation = construct_double_fc(features,
                                       hidden_size=hidden_size,
                                       regularization=regularization,
                                       kernel_initializer=kernel_initializer,
                                       name='rotation')
    fc2_translation = construct_double_fc(features,
                                          hidden_size=hidden_size,
                                          regularization=regularization,
                                          kernel_initializer=kernel_initializer,
                                          name='translation')
    outputs = construct_outputs(fc2_rotation, fc2_translation, regularization=regularization)
    return outputs
