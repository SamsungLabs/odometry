import copy
import numpy as np

from tensorflow.python.client import device_lib

from submodules.tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from submodules.tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

from slam.preprocessing.estimators.network_estimator import NetworkEstimator
from slam.utils import resize_image, resize_image_arr


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
        nn_opts['resize'] = False
        return nn_opts 

    def _load_model(self):
        nn_opts = self.get_nn_opts()
        self.model = pwc_net(mode='test', options=nn_opts)

    def _convert_model_output_to_prediction(self, optical_flow, target_size=None):
        if not isinstance(optical_flow, np.ndarray):
            optical_flow = np.stack(optical_flow)

        batch_size, height, width, channels_num = optical_flow.shape

#         small_height = height // 4
#         small_width = width // 4

#         small_optical_flow = np.zeros((batch_size, small_height, small_width, channels_num))
#         for batch_index in range(batch_size):
#             small_optical_flow[batch_index] = resize_image(optical_flow[batch_index], (small_width, small_height))

        if target_size is not None:
            final_height, final_width = target_size

            final_optical_flow = np.zeros((batch_size, final_height, final_width, channels_num))
            for batch_index in range(batch_size):
                final_optical_flow[batch_index] = resize_image_arr(optical_flow[batch_index],
                                                                   target_size=target_size,
                                                                   data_format='channels_last',
                                                                   mode='nearest')
        else:
            final_optical_flow = optical_flow

        final_optical_flow[..., 0] /= width * width / final_width
        final_optical_flow[..., 1] /= height * height / final_height
        return final_optical_flow

    def _run_model_inference(self, model_input):
        return self.model.predict_from_img_pairs(model_input, batch_size=1, verbose=False)
