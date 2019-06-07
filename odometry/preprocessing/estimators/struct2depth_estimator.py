import os
import numpy as np
import tensorflow as tf

from submodules.tf_models.research.struct2depth.model import Model as struct2depth_net
from submodules.tf_models.research.struct2depth.nets import RESNET
from submodules.tf_models.research.struct2depth.util import get_vars_to_save_and_restore

from utils.io_utils import load_image
from preprocessing.estimators.network_estimator import NetworkEstimator


class Struct2DepthEstimator(NetworkEstimator):

    def _load_model(self):
        self.model = struct2depth_net(is_training=False,
                                      batch_size=1,
                                      img_height=self.image_manager.height,
                                      img_width=self.image_manager.width,
                                      seq_length=3,
                                      architecture=RESNET,
                                      imagenet_norm=True,
                                      use_skip=True,
                                      joint_encoder=True)
        vars_to_restore = get_vars_to_save_and_restore(self.checkpoint)
        self.saver = tf.train.Saver(vars_to_restore)

    def _convert_image_to_model_input(self, image):
        return np.array(image, dtype=np.float32) / 255.

    def run(self, row, dataset_root=None):
        assert dataset_root is not None
        inputs = self._load_inputs(row, dataset_root)
        supervisor = tf.train.Supervisor(logdir='/tmp/', saver=None)
        with supervisor.managed_session() as sess:     
            self.saver.restore(sess, self.checkpoint)
            model_output = self.model.inference_depth(inputs, sess)[0]
        
        self._save_model_output(model_output, row, dataset_root)    
        row[self.output_col] = output_path
        return row
