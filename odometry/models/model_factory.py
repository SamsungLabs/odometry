from functools import partial

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam

from odometry.models.losses import (mean_squared_error,
                                    mean_absolute_error,
                                    mean_squared_logarithmic_error,
                                    mean_squared_signed_logarithmic_error,
                                    confidence_error,
                                    rmse,
                                    smooth_L1)

from odometry.models.layers import (activ,
                                    conv2d,
                                    conv2d_transpose,
                                    gated_conv2d,
                                    construct_fc,
                                    construct_double_fc,
                                    construct_outputs,
                                    ConstLayer,
                                    AssociationLayer,
                                    AddGridLayer)


class BaseModelFactory:
    def construct(self):
        raise NotImplementedError


class PretrainedModelFactory(BaseModelFactory):
    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path

    def construct(self):
        model = load_model(
            self.pretrained_path,
            custom_objects={'mean_squared_error': mean_squared_error,
                            'mean_absolute_error': mean_absolute_error,
                            'mean_squared_logarithmic_error': mean_squared_logarithmic_error,
                            'smooth_L1': smooth_L1,
                            'smoothL1': smooth_L1,
                            'flow_loss': mean_squared_logarithmic_error,
                            'confidence_error': confidence_error,
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
        return model


class ModelFactory:
    def __init__(self,
                 construct_graph_fn,
                 input_size=(60, 80),
                 channels_counts=(3, 3),
                 lr=0.001,
                 loss=mean_squared_error,
                 scale_rotation=1.,
                 scale_translation=1.):
        self.construct_graph_fn = construct_graph_fn
        self.input_size = input_size
        self.channels_counts = channels_counts
        self.optimizer = Adam(lr=lr, amsgrad=True)
        self.loss_fn = self._get_loss_function(loss)
        self.loss = [self.loss_fn] * 6
        self.loss_weights = [scale_rotation] * 3 + [scale_translation] * 3
        self.metrics = dict(zip(('euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z'), [rmse] * 6))

    @staticmethod
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
            if loss in ('confidence', 'confidence_error'):
                return confidence_error
        elif callable(loss):
            return loss
        else:
            raise ValueError

    def _concat_inputs(self):
        inputs = [Input((self.input_size[0], self.input_size[1], count))
                  for count in self.channels_counts]

        if len(inputs) == 1:
            return inputs, inputs[0]
        else:
            return inputs, concatenate(inputs)

    def construct(self):
        inputs, inputs_concatenated = self._concat_inputs()
        outputs = self.construct_graph_fn(inputs_concatenated)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss,
                      loss_weights=self.loss_weights,
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        return model


class ModelWithDecoderFactory(ModelFactory):
    def __init__(self,
                 construct_graph_fn,
                 input_size=(60, 80),
                 channels_counts=(3, 3),
                 lr=0.001,
                 loss=mean_squared_error,
                 scale_rotation=1.,
                 scale_translation=1.,
                 flow_loss_weight=1.,
                 flow_reconstruction_loss=mean_squared_logarithmic_error):
        super().__init__(construct_graph_fn=construct_graph_fn,
                         input_size=input_size,
                         channels_counts=channels_counts,
                         lr=lr,
                         loss=loss,
                         scale_rotation=scale_rotation,
                         scale_translation=scale_translation)
        self.loss.append(flow_reconstruction_loss)
        self.loss_weights.append(flow_loss_weight)


class ConstantModelFactory(ModelFactory):
    def __init__(self,
                 rot_and_trans_array,
                 input_size=(60, 80),
                 channels_counts=(3, 3)):
        from odometry.models.networks.basic import construct_constant_model
        super().__init__(construct_graph_fn=partial(construct_constant_model,
                                                    rot_and_trans_array=rot_and_trans_array),
                         input_size=input_size,
                         channels_counts=channels_counts)
