import os
import shutil
import copy
import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

from submodules.tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from submodules.tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

from odometry.preprocessing.estimators.pwcnet_estimator import PWCNetEstimator
from odometry.utils import resize_image


class PWCNetFeatureExtractor(PWCNetEstimator):

    def __init__(self, *args, **kwargs):
        super(PWCNetFeatureExtractor, self).__init__(*args, **kwargs)
        self.name = 'PWCNetExtractor'

    def _load_model(self):
        nn_opts = copy.deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = True
        nn_opts['ckpt_path'] = self.checkpoint
        nn_opts['batch_size'] = 1

        devices = device_lib.list_local_devices()
        gpus = [device for device in devices if device.device_type=='GPU']
        device = (gpus if len(gpus) else devices)[0].name

        nn_opts['gpu_devices'] = [device]
        nn_opts['controller'] = device
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
        nn_opts['ret_feat'] = True
        self.model = pwc_net(mode='test', options=nn_opts)

    def _run_model_inference(self, model_input):
        return self.model.return_features(model_input, batch_size=1, verbose=False)
