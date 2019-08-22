from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Conv2D, concatenate
from keras.regularizers import l2

from slam.models.layers import (concat,
                                conv2d,
                                construct_fc,
                                construct_double_fc,
                                construct_outputs)
from slam.utils import mlflow_logging


@mlflow_logging(ignore=('inputs',), prefix='model.', name='ResNet50')
def construct_resnet50_model(inputs,
                             weights='imagenet', 
                             kernel_initializer='glorot_normal'):

    inputs = concat(inputs)
    conv0 = Conv2D(3,
                   kernel_size=7,
                   padding='same',
                   activation='relu',
                   kernel_initializer=kernel_initializer,
                   name='conv0')(inputs)

    features = ResNet50(weights=weights, include_top=False, pooling=None)(conv0)
    flatten = Flatten()(features)

    fc2 = construct_double_fc(flatten,
                              hidden_size=500,
                              activation='relu',
                              kernel_initializer=kernel_initializer)
    outputs = construct_outputs(fc2, fc2)
    return outputs


@mlflow_logging(ignore=('inputs',), prefix='model.', name='Simple')
def construct_simple_model(inputs,
                           conv_layers=3,
                           conv_filters=64,
                           kernel_sizes=3,
                           strides=1,
                           paddings='same',
                           fc_layers=2,
                           hidden_sizes=500,
                           activations='elu',
                           regularizations=0,
                           batch_norms=True):
    if type(conv_filters) != list:
        conv_filters = [conv_filters] * conv_layers
    if type(kernel_sizes) != list:
        kernel_sizes = [kernel_sizes] * conv_layers
    if type(strides) != list:
        strides = [strides] * conv_layers
    if type(paddings) != list:
        paddings = [paddings] * conv_layers
    if type(hidden_sizes) != list:
        hidden_sizes = [hidden_sizes] * fc_layers
    if type(activations) != list:
        activations = [activations] * (conv_layers + fc_layers)
    if type(regularizations) != list:
        regularizations = [regularizations] * (conv_layers + fc_layers)
    if type(batch_norms) != list:
        batch_norms = [batch_norms] * (conv_layers + fc_layers)

    inputs = concat(inputs)

    conv = inputs
    for i in range(conv_layers):
        conv = conv2d(conv,
                      conv_filters[i],
                      kernel_size=kernel_sizes[i],
                      batch_norm=batch_norms[i],
                      padding=paddings[i],
                      kernel_initializer='glorot_normal',
                      strides=strides[i],
                      activation=activations[i],
                      activity_regularizer=l2(regularizations[i]))

    flatten = Flatten()(conv)

    fc = flatten
    for i in range(fc_layers):
        fc = Dense(hidden_sizes[i],
                   kernel_initializer='glorot_normal',
                   activation=activations[i + conv_layers],
                   activity_regularizer=l2(regularizations[i]))(fc)

    outputs = construct_outputs(fc, fc)
    return outputs
