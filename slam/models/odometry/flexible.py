from keras.layers.merge import concatenate
from keras.layers import Flatten

from slam.models.layers import (concat,
                                conv2d,
                                gated_conv2d,
                                construct_double_fc,
                                construct_outputs,
                                construct_outputs_with_confidences)
from slam.utils import mlflow_logging


def construct_encoder(inputs,
                      kernel_sizes=[7, 5, 3, 3, 3, 3],
                      strides=[2, 1, 4, 1, 2, 1],
                      dilation_rates=None,
                      kernel_initializer='glorot_normal',
                      use_gated_convolutions=False,
                      use_batchnorm=False):
    conv = gated_conv2d if use_gated_convolutions else conv2d

    layers = len(strides)
    if dilation_rates is None:
        dilation_rates = [1] * layers

    assert layers == len(dilation_rates) and layers == len(kernel_sizes)
    for i in range(layers):
        inputs = conv(inputs,
                      64,
                      kernel_size=kernel_sizes[i],
                      strides=strides[i],
                      dilation_rate=dilation_rates[i],
                      padding='same',
                      batchnorm=True if use_batchnorm and i == 0 else False,
                      activation='relu',
                      kernel_initializer=kernel_initializer)

    flatten = Flatten()(inputs)
    return flatten


@mlflow_logging(ignore=('inputs',), prefix='model.', name='Flexible')
def construct_flexible_model(inputs,
                             kernel_sizes=[7, 5, 3, 3, 3, 3],
                             strides=[2, 1, 4, 1 ,2, 1],
                             dilation_rates=None,
                             hidden_size=500,
                             regularization=0,
                             activation='relu',
                             kernel_initializer='glorot_normal',
                             use_gated_convolutions=False,
                             use_batchnorm=False,
                             return_confidence=False):

    inputs = concat(inputs)
    features = construct_encoder(inputs,
                                 kernel_sizes=kernel_sizes,
                                 strides=strides,
                                 dilation_rates=dilation_rates,
                                 kernel_initializer=kernel_initializer,
                                 use_gated_convolutions=use_gated_convolutions,
                                 use_batchnorm=use_batchnorm)
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
    if not return_confidence:
        return outputs

    outputs_with_confidences = construct_outputs_with_confidences(outputs, fc2_rotation, fc2_translation,
                                                                  regularization=regularization)
    return outputs_with_confidences
