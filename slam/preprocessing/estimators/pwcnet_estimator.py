import copy
import numpy as np

from tensorflow.python.client import device_lib

from submodules.tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from submodules.tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

from slam.preprocessing.estimators.network_estimator import NetworkEstimator
from slam.utils import resize_image


class PWCNetEstimator(NetworkEstimator):

    def __init__(self, batch_size=1, *args, **kwargs):
        self.batch_size = batch_size
        super(PWCNetEstimator, self).__init__(*args, **kwargs)
        self.name = 'PWCNet'

    def get_nn_opts(self):
        nn_opts = copy.deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
        nn_opts['verbose'] = True
        nn_opts['ckpt_path'] = self.checkpoint
        nn_opts['batch_size'] = self.batch_size

        devices = device_lib.list_local_devices()
        gpus = [device for device in devices if device.device_type=='GPU']
        device = (gpus if len(gpus) else devices)[0].name

        nn_opts['gpu_devices'] = [device]
        nn_opts['controller'] = device
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
        return nn_opts 

    def _load_model(self):
        nn_opts = self.get_nn_opts()
        self.model = pwc_net(mode='test', options=nn_opts)

    def _convert_model_output_to_prediction(self, optical_flow):
        if isinstance(optical_flow, list):
            optical_flow = np.stack(optical_flow)

        size = optical_flow.shape
        small_size = (size[0], size[1] // 4, size[2] // 4, size[3])
        optical_flow_small = np.zeros(small_size)
        for batch in range(optical_flow.shape[0]):
            optical_flow_small[batch] = resize_image(optical_flow[batch], (small_size[2], small_size[1]))
        optical_flow_small[..., 0] /= size[2]
        optical_flow_small[..., 1] /= size[1]
        return optical_flow_small

    def _run_model_inference(self, model_input):
        return self.model.predict_from_img_pairs(model_input, batch_size=1, verbose=False)
