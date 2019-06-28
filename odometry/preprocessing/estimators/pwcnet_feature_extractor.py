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

    def get_nn_opts(self):
        nn_opts = super(PWCNetFeatureExtractor, self).get_nn_opts()
        nn_opts['ret_feat'] = True
        return nn_opts

    def _run_model_inference(self, model_input):
        return self.model.return_features(model_input, batch_size=1, verbose=False)
