from keras.models import Model
from keras.layers.core import Lambda
from keras.layers import BatchNormalization, Flatten, Dense, Concatenate
from keras import regularizers

from odometry.models.layers import (conv2d,
                                    construct_fc,
                                    construct_outputs,
                                    AssociationLayer,
                                    AddGridLayer)


def construct_encoder(frames_concatenated,
                       use_depth=True,
                       use_flow=True,
                       use_association_layer=True,
                       concat_axis=3,
                       merge_filters=256,
                       merge_stride=2,
                       regularization_depth=0,
                       depth_multiplicator=None,
                       use_batchnorm_depth=False,
                       use_batchnorm_flow=False,
                       add_grid_layer=False,
                       f_x=1,
                       f_y=1,
                       c_x=0.5,
                       c_y=0.5,
                       kernel_initializer='glorot_normal'):
    # flow convolutional branch
    if use_flow:
        flow_xy = Lambda(lambda x: x[:, :, :, :2])(frames_concatenated)
        if use_batchnorm_flow:
            flow_xy = BatchNormalization()(flow_xy)

        if add_grid_layer:
            flow_xy = AddGridLayer(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)(flow_xy)

        conv1_flow = conv2d(flow_xy, 64, kernel_size=3, strides=2,
                            kernel_initializer=kernel_initializer, name='conv1_flow')
        conv2_flow = conv2d(conv1_flow, 128, kernel_size=3, strides=2,
                            kernel_initializer=kernel_initializer, name='conv2_flow')
        conv3_flow = conv2d(conv2_flow, 256, kernel_size=3, strides=2,
                            kernel_initializer=kernel_initializer, name='conv3_flow')
        conv4_flow = conv2d(conv3_flow, 512, kernel_size=3, strides=2,
                            kernel_initializer=kernel_initializer, name='conv4_flow')

    # depth convolutional branch
    if use_depth:
        if use_association_layer: # pass flow_z as input
            depths = AssociationLayer()(frames_concatenated)
            if use_batchnorm_depth:
                depths = BatchNormalization()(depths)
        else:
            depths = Lambda(lambda x: x[:, :, :, 2:])(frames_concatenated)

        if add_grid_layer:
            depths = AddGridLayer(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)(depths)

        conv1_depth = conv2d(depths, 64, kernel_size=3, strides=2,
                             kernel_initializer=kernel_initializer, name='conv1_depth')
        conv2_depth = conv2d(conv1_depth, 128, kernel_size=3, strides=2,
                             kernel_initializer=kernel_initializer, name='conv2_depth')
        conv3_depth = conv2d(conv2_depth, 256, kernel_size=3, strides=2,
                             kernel_initializer=kernel_initializer, name='conv3_depth')
        conv4_depth = conv2d(conv3_depth, 512, kernel_size=3, strides=2,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=regularizers.l2(regularization_depth),
                             bias_regularizer=regularizers.l2(regularization_depth),
                             name='conv4_depth')

    if depth_multiplicator is not None:
        conv4_depth = Lambda(lambda x: x * depth_multiplicator)(conv4_depth)

    if use_flow and use_depth:
        concatenated = Concatenate(axis=concat_axis)([conv4_flow, conv4_depth])
    elif use_flow:
        concatenated = conv4_flow
    elif use_depth:
        concatenated = conv4_depth

    merged = conv2d(concatenated, merge_filters, kernel_size=1, strides=merge_stride,
                    kernel_initializer=kernel_initializer, name='merge')

    flatten = Flatten(name='flatten')(merged)
    return flatten


def construct_depth_flow_model(imgs,
                               frames_concatenated,
                               use_depth=True,
                               use_flow=True,
                               use_association_layer=True,
                               concat_axis=3,
                               merge_filters=256,
                               merge_stride=1,
                               fc_layers=2,
                               regularization_depth=0,
                               regularization_fc=0,
                               depth_multiplicator=None,
                               use_batchnorm_depth=False,
                               use_batchnorm_flow=False,
                               add_grid_layer=False,
                               f_x=1,
                               f_y=1,
                               c_x=0.5,
                               c_y=0.5,
                               kernel_initializer='glorot_normal'):
    flatten = construct_encoder(frames_concatenated,
                                use_depth=use_depth,
                                use_flow=use_flow,
                                use_association_layer=use_association_layer,
                                concat_axis=concat_axis,
                                merge_filters=merge_filters,
                                merge_stride=merge_stride,
                                regularization_depth=regularization_depth,
                                depth_multiplicator=depth_multiplicator,
                                use_batchnorm_depth=use_batchnorm_depth,
                                use_batchnorm_flow=use_batchnorm_flow,
                                add_grid_layer=add_grid_layer,
                                f_x=f_x,
                                f_y=f_y,
                                c_x=c_x,
                                c_y=c_y,
                                kernel_initializer=kernel_initializer)

    fc_rotation = construct_fc(flatten, hidden_size=128,
                               kernel_initializer=kernel_initializer, name='fc1_rotation')
    fc_translation = construct_fc(flatten, hidden_size=128,
                                  kernel_initializer=kernel_initializer, name='fc1_translation')
     
    for i in range(2, fc_layers+1):
        fc_rotation = construct_fc(fc_rotation, hidden_size=128,
                                   kernel_initializer=kernel_initializer, name='fc{}_rotation'.format(i))
        fc_translation = construct_fc(fc_translation, hidden_size=128,
                                      kernel_initializer=kernel_initializer, name='fc{}_translation'.format(i))

    outputs = construct_outputs(fc_rotation, fc_translation, regularization=regularization_fc)
    model = Model(inputs=imgs, outputs=outputs)
    return model

