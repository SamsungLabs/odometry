import mlflow
import numpy as np
import tensorflow as tf

from submodules.tf_models.research.struct2depth.model import Model as struct2depth_net
from submodules.tf_models.research.struct2depth.nets import RESNET
from submodules.tf_models.research.struct2depth.util import get_vars_to_save_and_restore

from slam.utils import resize_image
from slam.preprocessing.estimators.network_estimator import NetworkEstimator


class Struct2DepthEstimator(NetworkEstimator):

    def __init__(self, *args, **kwargs):
        super(Struct2DepthEstimator, self).__init__(*args, **kwargs)
        self.name = 'Struct2Depth'

    def _load_model(self):
        assert self.input_size is not None
        self.model = struct2depth_net(is_training=False,
                                      batch_size=1,
                                      img_height=self.input_size[0],
                                      img_width=self.input_size[1],
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
        image = resize_image(image, target_size=self.input_size)
        return np.array(image, dtype=np.float32) / 255.

    def _run_model_inference(self, model_input):
        return self.model.inference_depth(model_input, self.sess)
