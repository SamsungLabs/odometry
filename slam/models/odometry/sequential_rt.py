from keras.layers import Lambda, Flatten

from slam.models.layers import (concat,
                                conv2d,
                                dense,
                                gated_conv2d,
                                FlowComposer)
from slam.utils import mlflow_logging

from keras.layers import Layer, Dense, concatenate, Subtract
from keras.regularizers import l2


def construct_output(input_fc, name, regularization=0):
    outputs = []

    if name == 'rotation':
        output_names = ['euler_x', 'euler_y', 'euler_z']
    elif name == 'translation':
        output_names = ['t_x', 't_y', 't_z']
    else:
        raise 'Invalid name'

    for output_name in output_names:
        output = Dense(1, kernel_regularizer=l2(regularization), name=output_name)(input_fc)
        outputs.append(output)

    return outputs


def construct_encoder(inputs,
                      kernel_sizes=[7, 5, 3, 3, 3, 3],
                      strides=[2, 1, 4, 1, 2, 1],
                      dilation_rates=None,
                      kernel_initializer='glorot_normal',
                      use_gated_convolutions=False,
                      use_batch_norm=False):
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
                      batch_norm=use_batch_norm and i == 0,
                      activation='relu',
                      kernel_initializer=kernel_initializer)

    flatten = Flatten()(inputs)
    return flatten


@mlflow_logging(ignore=('inputs',), prefix='model.', name='SequentialRT')
def construct_sequential_rt_model(inputs,
                                  intrinsics,
                                  use_input_flow_for_translation=True,
                                  use_cleaned_flow_for_translation=True,
                                  use_rotation_flow_for_translation=False,
                                  kernel_sizes=[7, 5, 3, 3, 3, 3],
                                  strides=[2, 1, 4, 1 ,2, 1],
                                  dilation_rates=None,
                                  hidden_size=500,
                                  regularization=0,
                                  activation='relu',
                                  kernel_initializer='glorot_normal',
                                  use_gated_convolutions=False,
                                  use_batch_norm=False,
                                  return_confidence=False):

    inputs = concat(inputs)
    features_rotation = construct_encoder(inputs,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          dilation_rates=dilation_rates,
                                          kernel_initializer=kernel_initializer,
                                          use_gated_convolutions=use_gated_convolutions,
                                          use_batch_norm=use_batch_norm)

    fc_rotation = dense(features_rotation,
                        output_size=hidden_size,
                        regularization=regularization,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        layers_num=2,
                        name='rotation')

    output_rotation = construct_output(fc_rotation,
                                       name='rotation',
                                       regularization=regularization)

    rotation_flow = FlowComposer(intrinsics)(output_rotation)
    inputs_for_translation = []
    if use_input_flow_for_translation:
        inputs_for_translation.append(inputs)
    if use_cleaned_flow_for_translation:
        inputs_for_translation.append(Subtract()([inputs, rotation_flow]))
    if use_rotation_flow_for_translation:
        inputs_for_translation.append(rotation_flow)

    features_translation = construct_encoder(concat(inputs_for_translation),
                                             kernel_sizes=kernel_sizes,
                                             strides=strides,
                                             dilation_rates=dilation_rates,
                                             kernel_initializer=kernel_initializer,
                                             use_gated_convolutions=use_gated_convolutions,
                                             use_batch_norm=use_batch_norm)

    fc_translation = dense(features_translation,
                           output_size=hidden_size,
                           regularization=regularization,
                           activation=activation,
                           kernel_initializer=kernel_initializer,
                           layers_num=2,
                           name='translation')

    output_translation = construct_output(fc_rotation,
                                           name='translation',
                                           regularization=regularization)

    return output_rotation + output_translation
