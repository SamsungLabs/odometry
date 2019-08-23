from keras.layers import Lambda, Flatten

from slam.models.layers import (concat,
                                conv2d,
                                gated_conv2d,
                                construct_double_fc,
                                FlowGenerator)
from slam.utils import mlflow_logging

from keras.layers import Layer, Dense, concatenate
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


class Intrinsics:
    def __init__(self, f_x, f_y, c_x, c_y, width, height):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

        self.f_x_scaled = f_x * width
        self.f_y_scaled = f_y * height
        self.c_x_scaled = c_x * width
        self.c_y_scaled = c_y * height

        self.width = width
        self.height = height

    def forward(self, xy):
        xy_processed = xy.copy()
        xy_processed[0] = (xy[0] - self.c_x_scaled) / self.f_x_scaled
        xy_processed[1] = (xy[1] - self.c_y_scaled) / self.f_y_scaled
        return xy_processed

    def backward(self, xy):
        xy_processed = xy.copy()
        xy_processed[0] = xy[0] * self.f_x_scaled + self.c_x_scaled
        xy_processed[1] = xy[1] * self.f_y_scaled + self.c_y_scaled
        return xy_processed

    def create_frustrum(self, x_pixels, y_pixels, depth):
        xy_pixels = np.c_[[x_pixels, y_pixels]]
        return self.forward(xy_pixels) * depth


@mlflow_logging(ignore=('inputs',), prefix='model.', name='SequentialRT')
def construct_sequential_rt_model(inputs,
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

    intrinsics = Intrinsics(f_x=0.5792554391619662,
                            f_y=1.91185106382978721,
                            c_x=0.48927703464947625,
                            c_y=0.4925949468085106,
                            width=320,
                            height=96)

    inputs = concat(inputs)
    features_rotation = construct_encoder(inputs,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          dilation_rates=dilation_rates,
                                          kernel_initializer=kernel_initializer,
                                          use_gated_convolutions=use_gated_convolutions,
                                          use_batch_norm=use_batch_norm)

    fc_rotation = construct_double_fc(features_rotation,
                                      hidden_size=hidden_size,
                                      regularization=regularization,
                                      activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      name='rotation')

    output_rotation = construct_output(fc_rotation,
                                        name='rotation',
                                        regularization=regularization)

    rotation_flow = FlowGenerator(intrinsics)(output_rotation)

    inputs = concat((inputs, rotation_flow))

    features_translation = construct_encoder(inputs,
                                          kernel_sizes=kernel_sizes,
                                          strides=strides,
                                          dilation_rates=dilation_rates,
                                          kernel_initializer=kernel_initializer,
                                          use_gated_convolutions=use_gated_convolutions,
                                          use_batch_norm=use_batch_norm)

    fc_translation = construct_double_fc(features_translation,
                                         hidden_size=hidden_size,
                                         regularization=regularization,
                                         activation=activation,
                                         kernel_initializer=kernel_initializer,
                                         name='translation')

    output_translation = construct_output(fc_rotation,
                                           name='translation',
                                           regularization=regularization)


    return output_rotation + output_translation
