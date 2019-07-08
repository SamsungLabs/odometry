import mlflow
from keras.layers.merge import concatenate
from keras.layers import Flatten

from odometry.models.layers import (concat,
                                    conv2d,
                                    gated_conv2d,
                                    construct_double_fc,
                                    construct_outputs,
                                    construct_outputs_with_confidences)


def construct_encoder(inputs,
                      layers=4,
                      filters=[[16, 16, 32]] * 4,
                      kernel_sizes=[[7, 5, 3]] * 4,
                      strides=2,
                      dilation_rates=None,
                      kernel_initializer='glorot_normal',
                      use_gated_convolutions=False):
    conv = gated_conv2d if use_gated_convolutions else conv2d

    makelist = lambda x: [x] if isinstance(x, int) else x

    if isinstance(filters, int):
        filters = [filters] * layers

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * layers

    if isinstance(strides, int):
        strides = [strides] * layers

    if dilation_rates is None:
        dilation_rates = [1] * layers

    for i in range(layers):
        layer_filters = makelist(filters[i])
        layer_kernel_sizes = makelist(kernel_sizes[i])
        layer_dilation_rates = makelist(dilation_rates[i])
        layer_stride = strides[i]
        convs = max(len(layer_filters), max(len(layer_kernel_sizes), len(layer_dilation_rates)))

        assert len(layer_filters) in (1, convs)
        assert len(layer_kernel_sizes) in (1, convs)
        assert len(layer_dilation_rates) in (1, convs)

        if len(layer_filters) == 1:
            layer_filters *= convs

        if len(layer_kernel_sizes) == 1:
            layer_kernel_sizes *= convs

        if len(layer_dilation_rates) == 1:
            layer_dilation_rates *= convs

        print(f'Layer {i + 1}: {convs} convolutions')

        outputs = []
        for flt, kernel_size, dilation_rate in zip(layer_filters, layer_kernel_sizes, layer_dilation_rates):
            print(f'\tfilters={flt}, kernel size={kernel_size}, stride={layer_stride}, dilation rate={dilation_rate}')
            outputs.append(
                conv(inputs,
                     flt,
                     kernel_size=kernel_size,
                     strides=layer_stride,
                     dilation_rate=dilation_rate,
                     padding='same',
                     activation='relu',
                     kernel_initializer=kernel_initializer)
            )
        inputs = concat(outputs)

    merged = conv(inputs,
                  64,
                  kernel_size=1,
                  padding='same',
                  activation='relu',
                  kernel_initializer=kernel_initializer)
    flatten = Flatten()(merged)
    return flatten


def construct_multiscale_model(inputs,
                               layers=4,
                               filters=[[16, 16, 32]] * 4,
                               kernel_sizes=[[7, 5, 3]] * 4,
                               strides=2,
                               dilation_rates=None,
                               hidden_size=500,
                               regularization=0,
                               activation='relu',
                               kernel_initializer='glorot_normal',
                               use_gated_convolutions=False,
                               return_confidence=False
                              ):
    mlflow.log_param('model.name', 'Multiscale')
    mlflow.log_params({'model.' + k: repr(v) for k, v in locals().items() if 'inputs' not in k})

    inputs = concat(inputs)
    features = construct_encoder(inputs,
                                 layers=layers,
                                 filters=filters,
                                 kernel_sizes=kernel_sizes,
                                 strides=strides,
                                 dilation_rates=dilation_rates,
                                 kernel_initializer=kernel_initializer,
                                 use_gated_convolutions=use_gated_convolutions)
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
