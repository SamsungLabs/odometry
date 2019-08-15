from keras.layers import Flatten
from keras.regularizers import l2

from slam.models.layers import (concat,
                                conv2d,
                                construct_fc,
                                construct_outputs,
                                DepthFlow,
                                AddGrid)
from slam.utils import mlflow_logging


def construct_encoder(inputs,
                      use_depth=True,
                      use_flow=True,
                      use_association_layer=True,
                      concat_axis=3,
                      filters=256,
                      stride=2,
                      add_grid=False,
                      f_x=1,
                      f_y=1,
                      c_x=0.5,
                      c_y=0.5,
                      kernel_initializer='glorot_normal'):
    # flow convolutional branch
    if use_flow:
        flow = concat(inputs[:2])

        if add_grid:
            flow = AddGrid(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)(flow)

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
            depth = DepthFlow()(concat(inputs))
        else:
            depth = concat(inputs[2:])

        if add_grid:
            depth = AddGrid(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)(depth)

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
                               filters=256,
                               stride=1,
                               layers=3,
                               regularization=0,
                               add_grid=False,
                               f_x=1,
                               f_y=1,
                               c_x=0.5,
                               c_y=0.5,
                               kernel_initializer='glorot_normal'):
    features = construct_encoder(inputs,
                                 use_depth=use_depth,
                                 use_flow=use_flow,
                                 use_association_layer=use_association_layer,
                                 filters=filters,
                                 stride=stride,
                                 add_grid=add_grid,
                                 f_x=f_x,
                                 f_y=f_y,
                                 c_x=c_x,
                                 c_y=c_y,
                                 kernel_initializer=kernel_initializer)

    fc_rotation = features
    fc_translation = features
    for i in range(1, layers + 1):
        fc_rotation = construct_fc(fc_rotation,
                                   hidden_size=128,
                                   kernel_initializer=kernel_initializer,
                                   name=f'fc{i}_rotation')
        fc_translation = construct_fc(fc_translation,
                                      hidden_size=128,
                                      kernel_initializer=kernel_initializer,
                                      name=f'fc{i}_translation')

    outputs = construct_outputs(fc_rotation, fc_translation, regularization=regularization)
    return outputs
