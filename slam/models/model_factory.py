import mlflow

import tensorflow as tf
import keras
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.layer_utils import count_params

from slam.models.losses import (mean_squared_error,
                                mean_absolute_error,
                                mean_squared_logarithmic_error,
                                mean_squared_signed_logarithmic_error,
                                confidence_error,
                                rmse,
                                smooth_L1)
from slam.models.layers import CUSTOM_LAYERS
from slam.utils import mlflow_logging


class BaseModelFactory:
    def construct(self):
        raise NotImplementedError


class PretrainedModelFactory(BaseModelFactory):

    @mlflow_logging(ignore=(), prefix='model_factory.')
    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path
        self.model = None

    def construct(self):
        sess = tf.Session()
        keras.backend.set_session(sess)
        with sess.as_default():
            custom_objects={'mean_squared_error': mean_squared_error,
                            'mean_absolute_error': mean_absolute_error,
                            'mean_squared_logarithmic_error': mean_squared_logarithmic_error,
                            'smooth_L1': smooth_L1,
                            'flow_loss': mean_squared_logarithmic_error,
                            'confidence_error': confidence_error,
                            'rmse': rmse,
                            **CUSTOM_LAYERS}
            self.model = load_model(self.pretrained_path, custom_objects=custom_objects)
        return self.model


class ModelFactory:

    @mlflow_logging(ignore=('construct_graph_fn',), prefix='model_factory.')
    def __init__(self,
                 construct_graph_fn,
                 input_shapes=((60, 80, 3), (60, 80, 3)),
                 lr=0.001,
                 loss=mean_squared_error,
                 scale_rotation=1.,
                 scale_translation=1.):

        self.model = None
        self.construct_graph_fn = construct_graph_fn
        self.input_shapes = input_shapes
        self.lr = lr
        self.loss_fn = self._get_loss_function(loss)
        self.loss = [self.loss_fn] * 6
        self.loss_weights = [scale_rotation] * 3 + [scale_translation] * 3
        self.metrics = dict(zip(('euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z'), [rmse] * 6))

    def _get_optimizer(self):
        return Adam(lr=self.lr, amsgrad=True)

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

    def _compile(self):
        self.optimizer = self._get_optimizer()
        self.model.compile(loss=self.loss,
                           loss_weights=self.loss_weights,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def construct(self):
        inputs = [Input(input_shape) for input_shape in self.input_shapes]
        outputs = self.construct_graph_fn(inputs)
        self.model = Model(inputs=inputs, outputs=outputs)
        self._compile()

        if mlflow.active_run():
            mlflow.log_metric('Number of parameters', count_params(self.model.trainable_weights))
        return self.model


class ModelWithDecoderFactory(ModelFactory):
    def __init__(self,
                 construct_graph_fn,
                 input_shapes=((60, 80, 3), (60, 80, 3)),
                 lr=0.001,
                 loss=mean_squared_error,
                 scale_rotation=1.,
                 scale_translation=1.,
                 flow_loss_weight=1.,
                 flow_reconstruction_loss=mean_squared_logarithmic_error):
        super().__init__(construct_graph_fn=construct_graph_fn,
                         input_shapes=input_shapes,
                         lr=lr,
                         loss=loss,
                         scale_rotation=scale_rotation,
                         scale_translation=scale_translation)
        self.loss.append(flow_reconstruction_loss)
        self.loss_weights.append(flow_loss_weight)


class ModelWithConfidenceFactory(ModelFactory):

    def freeze(self, lr=0.01):
        for layer in self.model.layers:
            layer.trainable = not layer.trainable

        self.loss = confidence_error
        self.lr = lr
        self._compile()
        for layer in self.model.layers:
            print(f'{layer.name:<30} {layer.trainable}')
        return self.model
