from keras.layers import Flatten
from keras.regularizers import l2

from slam.models.layers import (concat,
                                conv2d,
                                dense,
                                construct_outputs,
                                depth_flow,
                                add_grid)
from slam.utils import mlflow_logging


def construct_encoder(inputs,
                      use_depth=True,
                      use_flow=True,
                      use_association_layer=True,
                      use_grid=False,
                      concat_axis=3,
                      filters=256,
                      stride=2,
                      f_x=1,
                      f_y=1,
                      c_x=0.5,
                      c_y=0.5,
                      kernel_initializer='glorot_normal'):
    # flow convolutional branch
    if use_flow:
        flow = concat(inputs[:2])

        if use_grid:
            flow = add_grid(flow, f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

        for i in range(1, 5):
            flow = conv2d(flow,
                          2 ** (i + 5),
                          kernel_size=3,
                          strides=2,
                          kernel_initializer=kernel_initializer,
                          name=f'conv{i}_flow')

    # depth convolutional branch
    if use_depth:
        if use_association_layer: # pass flow_z as input
            depth = depth_flow(concat(inputs))
        else:
            depth = concat(inputs[2:])

        if use_grid:
            depth = add_grid(depth, f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

        for i in range(1, 5):
            depth = conv2d(depth,
                           2 ** (i + 5),
                           kernel_size=3,
                           strides=2,
                           kernel_initializer=kernel_initializer,
                           name=f'conv{i}_depth')

    if use_flow and use_depth:
        concatenated = concat([flow, depth])
    elif use_flow:
        concatenated = flow
    elif use_depth:
        concatenated = depth

    merged = conv2d(concatenated,
                    filters,
                    kernel_size=1,
                    strides=stride,
                    kernel_initializer=kernel_initializer,
                    name='merge')

    flatten = Flatten()(merged)
    return flatten


@mlflow_logging(ignore=('inputs',), prefix='model.', name='DepthFlow')
def construct_depth_flow_model(inputs,
                               use_depth=True,
                               use_flow=True,
                               use_association_layer=True,
                               use_grid=False,
                               filters=256,
                               stride=1,
                               layers=3,
                               regularization=0,
                               f_x=1,
                               f_y=1,
                               c_x=0.5,
                               c_y=0.5,
                               kernel_initializer='glorot_normal'):

    features = construct_encoder(inputs,
                                 use_depth=use_depth,
                                 use_flow=use_flow,
                                 use_association_layer=use_association_layer,
                                 use_grid=use_grid,
                                 filters=filters,
                                 stride=stride,
                                 f_x=f_x,
                                 f_y=f_y,
                                 c_x=c_x,
                                 c_y=c_y,
                                 kernel_initializer=kernel_initializer)

    fc_rotation = dense(features,
                        output_size=128,
                        layers_num=layers,
                        kernel_initializer=kernel_initializer,
                        name='rotation')

    fc_translation = dense(features,
                           output_size=128,
                           layers_num=layers,
                           kernel_initializer=kernel_initializer,
                           name='translation')

    outputs = construct_outputs([fc_rotation] * 3 + [fc_translation] * 3,
                                regularization=regularization)
    return outputs
