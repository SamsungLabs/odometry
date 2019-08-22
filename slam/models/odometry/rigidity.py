from keras.layers import Conv2D, GlobalAveragePooling2D

from slam.models.layers import concat, conv2d, add_grid, chunk
from slam.utils import mlflow_logging


@mlflow_logging(ignore=('inputs',), prefix='model.', name='Rigidity',)
def construct_rigidity_model(inputs,
                             batch_norm=True,
                             use_grid=True,
                             f_x=1,
                             f_y=1,
                             c_x=0.5, 
                             c_y=0.5, 
                             kernel_initializer='he_uniform'):

    inputs = concat(inputs)
    if use_grid:
        inputs = add_grid(inputs, f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

    conv1 = conv2d(inputs, 32, kernel_size=7, batch_norm=batch_norm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv2 = conv2d(conv1, 64, kernel_size=7, batch_norm=batch_norm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv3 = conv2d(conv2, 128, kernel_size=5, batch_norm=batch_norm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv4 = conv2d(conv3, 256, kernel_size=5, batch_norm=batch_norm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv5 = conv2d(conv4, 512, kernel_size=3, batch_norm=batch_norm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv6 = conv2d(conv5, 1024, kernel_size=3, batch_norm=batch_norm,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)

    conv7 = Conv2D(6, kernel_size=1, kernel_initializer=kernel_initializer)(conv6)

    pool = GlobalAveragePooling2D()(conv7)

    outputs = chunk(pool, n=6, names=['r_x', 'r_y', 'r_z', 't_x', 't_y', 't_z'])
    return outputs
