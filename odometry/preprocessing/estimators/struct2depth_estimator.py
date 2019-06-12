import os
import numpy as np
import tensorflow as tf

from submodules.tf_models.research.struct2depth.model import Model as struct2depth_net
from submodules.tf_models.research.struct2depth.nets import RESNET
from submodules.tf_models.research.struct2depth.util import get_vars_to_save_and_restore

from odometry.utils import resize_image
from odometry.preprocessing.estimators.network_estimator import NetworkEstimator


class Struct2DepthEstimator(NetworkEstimator):

    def _load_model(self):
        assert self.height is not None and self.width is not None
        self.model = struct2depth_net(is_training=False,
                                      batch_size=1,
                                      img_height=self.height,
                                      img_width=self.width,
                                      seq_length=3,
                                      architecture=RESNET,
                                      imagenet_norm=True,
                                      use_skip=True,
                                      joint_encoder=True)
        vars_to_restore = get_vars_to_save_and_restore(self.checkpoint)
        saver = tf.train.Saver(vars_to_restore)
        self.sess = tf.get_default_session() or tf.Session()
        saver.restore(self.sess, self.checkpoint)

    def _convert_image_to_model_input(self, image):
        image = resize_image(image, target_size=(self.width, self.height))
        return np.array(image, dtype=np.float32) / 255.

    def _run_model_inference(self, model_input):
        # tf.train.Supervisor is deprecated!
        # supervisor = tf.train.Supervisor(logdir='/tmp/', saver=None)
        # with supervisor.managed_session() as sess:
        #    self.saver.restore(sess, self.checkpoint)
        #    model_output = self.model.inference_depth(model_input, sess)
        return self.model.inference_depth(model_input, self.sess)
