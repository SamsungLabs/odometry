from keras.layers import Lambda, Conv2D, GlobalAveragePooling2D

from slam.models.layers import concat, conv2d, AddGridLayer
from slam.utils import mlflow_logging


@mlflow_logging(ignore=('inputs',), prefix='model.', name='Rigidity',)
def construct_rigidity_model(inputs,
                             batchnorm=True,
                             add_grid_layer=True, 
                             f_x=1,
                             f_y=1,
                             c_x=0.5, 
                             c_y=0.5, 
                             kernel_initializer='he_uniform'):

    inputs = concat(inputs)
    if add_grid_layer:
        inputs = AddGrid(f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)(inputs)

    conv1 = conv2d(inputs, 32, kernel_size=7, batchnorm=batchnorm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv2 = conv2d(conv1, 64, kernel_size=7, batchnorm=batchnorm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv3 = conv2d(conv2, 128, kernel_size=5, batchnorm=batchnorm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv4 = conv2d(conv3, 256, kernel_size=5, batchnorm=batchnorm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv5 = conv2d(conv4, 512, kernel_size=3, batchnorm=batchnorm, strides=2,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)
    conv6 = conv2d(conv5, 1024, kernel_size=3, batchnorm=batchnorm,
                   padding='same', activation='relu', kernel_initializer=kernel_initializer)

    conv7 = Conv2D(6, kernel_size=1, kernel_initializer=kernel_initializer)(conv6)

    pool = GlobalAveragePooling2D()(conv7)

    r_x = Lambda(lambda x: x[:,0:1], name='r_x')(pool)
    r_y = Lambda(lambda x: x[:,1:2], name='r_y')(pool)
    r_z = Lambda(lambda x: x[:,2:3], name='r_z')(pool)
    t_x = Lambda(lambda x: x[:,3:4], name='t_x')(pool)
    t_y = Lambda(lambda x: x[:,4:5], name='t_y')(pool)
    t_z = Lambda(lambda x: x[:,5:6], name='t_z')(pool)

    outputs = [r_x, r_y, r_z, t_x, t_y, t_z]
    return outputs
