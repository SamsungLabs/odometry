import keras
from keras import backend as K

from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam

import numpy as np

from models.losses import (mean_squared_error,
                           mean_absolute_error,
                           mean_squared_logarithmic_error,
                           mean_squared_signed_logarithmic_error,
                           confidence_error,
                           rmse,
                           smooth_L1)

from models.layers import (activ,
                           conv2d,
                           conv2d_transpose,
                           gated_conv2d,
                           construct_fc,
                           construct_double_fc,
                           construct_outputs,
                           ConstLayer,
                           AssociationLayer,
                           AddGridLayer)

from models.networks.basic import (construct_resnet50_model,
                                   construct_simple_model,
                                   construct_constant_model)


from models.networks.rigidity import construct_rigidity_model

from models.networks.ls_vo import (construct_st_vo_model,
                                   construct_ls_vo_model,
                                   construct_ls_vo_rt_model,
                                   construct_ls_vo_rt_no_decoder_model)

from models.networks.depth_flow import construct_depth_flow_model

from models.networks.flexible import construct_flexible_model


def _get_loss_function(loss):
    if isinstance(loss, str):
        loss = loss.lower()
        if loss in ('mse', 'mean_squared_error'):
            return mean_squared_error
        if loss in ('mae', 'mean_absolute_error'):
            return mean_absolute_error
        if loss in ('msle', 'mean_squared_logarithmic_error'):
            return mean_squared_logarithmic_error
        if loss in ('mssle', 'mean_squared_signed_logarithmic_error'):
            return mean_squared_signed_logarithmic_error
        if loss in ('rmse', 'root_mean_squared_error'):
            return rmse
        if loss in ('huber', 'smoothl1', 'smooth_l1'):
            return smooth_L1
    elif callable(loss):
        return loss
    else:
        raise ValueError


class ModelFabric:
    def __init__(self,
                 img_resized_size=(60, 80),
                 model_name='simple_model',
                 channels_counts=(3, 3),
                 lr=0.001,
                 loss=mean_squared_error,
                 flow_reconstruction_loss=mean_squared_logarithmic_error,
                 scale_rotation=1,
                 scale_translation=1):
        self.model_name = model_name
        self.img_resized_size = img_resized_size
        self.channels_counts = channels_counts
        self.optimizer = Adam(lr=lr, amsgrad=True)
        self.loss_fn = _get_loss_function(loss)
        self.loss = [self.loss_fn] * 6
        self.loss_weights = [scale_rotation] * 3 + [scale_translation] * 3
        self.flow_reconstruction_loss = [flow_reconstruction_loss]

    def _concat_inputs(self):
        imgs = [Input((self.img_resized_size[0], self.img_resized_size[1], count)) for count in self.channels_counts]

        if len(imgs) == 1:
            return imgs, imgs[0]
        else:
            return imgs, concatenate(imgs)

    def construct_pretrained_model(self, pretrained_path):
        print('Load weights from {}'.format(pretrained_path))
        self.model = keras.models.load_model(
            pretrained_path,
            custom_objects={'mean_squared_error': mean_squared_error,
                            'mean_absolute_error': mean_absolute_error,
                            'mean_squared_logarithmic_error': mean_squared_logarithmic_error,
                            'smooth_L1': smooth_L1,
                            'smoothL1': smooth_L1,
                            'flow_loss': mean_squared_logarithmic_error,
                            'rmse': rmse,
                            'activ': activ,
                            'conv2d': conv2d,
                            'conv2d_transpose': conv2d_transpose,
                            'gated_conv2d': gated_conv2d,
                            'construct_fc': construct_fc,
                            'construct_double_fc': construct_double_fc,
                            'construct_outputs': construct_outputs,
                            'ConstLayer': ConstLayer,
                            'AssociationLayer': AssociationLayer,
                            'AddGridLayer': AddGridLayer,
                            })
        return self.model

    def _construct_model(self, 
                         construction_procedure, 
                         freeze=False, 
                         trainable_layer_names=None,
                         pretrained_path=None, 
                         loss=None, 
                         loss_weights=None, 
                         **kwargs):
        imgs, frames_concatenated = self._concat_inputs()
        self.model = construction_procedure(imgs, frames_concatenated, **kwargs)

        if freeze and trainable_layer_names is not None:
            for layer in self.model.layers:
                layer.trainable = False
                for partial_name in trainable_layer_names:
                    if partial_name in layer.name:
                        layer.trainable = True

        if pretrained_path is not None:
            self.model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)

        if loss is None:
            loss = self.loss

        if loss_weights is None:
            loss_weights = self.loss_weights

        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=self.optimizer,
                           metrics={output_name : rmse for output_name in ('r_x', 'r_y', 'r_z', 't_x', 't_y', 't_z')})
        return self.model

    def construct_simple_model(self, **kwargs):
        return self._construct_model(construct_simple_model, **kwargs)

    def construct_resnet50_model(self, **kwargs):
        return self._construct_model(construct_resnet50_model, **kwargs)

    def construct_constant_model(self, **kwargs):
        return self._construct_model(construct_constant_model, **kwargs)

    def construct_rigidity_model(self, **kwargs):
        return self._construct_model(construct_rigidity_model, **kwargs)

    def construct_depth_flow_model(self, **kwargs):
        return self._construct_model(construct_depth_flow_model, **kwargs)

    def construct_st_vo_model(self, **kwargs):
        return self._construct_model(construct_st_vo_model, **kwargs)

    def construct_ls_vo_model(self, flow_loss_weight=1.0, **kwargs):
        return self._construct_model(construct_ls_vo_model,
                                     loss=self.loss + self.flow_reconstruction_loss,
                                     loss_weights=self.loss_weights + [flow_loss_weight],
                                     **kwargs)

    def construct_ls_vo_rt_model(self, flow_loss_weight=1.0, **kwargs):
        return self._construct_model(construct_ls_vo_rt_model, 
                                     loss=self.loss + self.flow_reconstruction_loss, 
                                     loss_weights=self.loss_weights + [flow_loss_weight], 
                                     **kwargs)

    def construct_ls_vo_rt_no_decoder_model(self, **kwargs):
        return self._construct_model(construct_ls_vo_rt_no_decoder_model, **kwargs)
    
    def construct_flexible_model(self, **kwargs):
        return self._construct_model(construct_flexible_model,  **kwargs)
