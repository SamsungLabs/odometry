import os
import shutil
import copy
import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

#import torch

from submodules.tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from submodules.tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

#from struct2depth.inference import _run_inference as run_struct2depth
from submodules.tf_models.research.struct2depth.model import Model as struct2depth_net
from submodules.tf_models.research.struct2depth.nets import RESNET
from submodules.tf_models.research.struct2depth.util import get_vars_to_save_and_restore

#from depth_pred import senet_model as se_net

import cv2


class BaseEstimator:
    def __init__(self,
                 directory,
                 image_manager,
                 checkpoint,
                 ext='.npy'):
        self.directory = directory
        self.image_manager = image_manager
        self.checkpoint = checkpoint
        self.ext = ext

        self.mapping = dict()
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def _load_model(self):
        pass

    def _convert_image_to_model_input(self, image):
        return np.array(image, dtype=np.float32)

    def _convert_model_output_to_prediction(self, output):
        return output

    def _construct_filename(self, reference_path, reference_path_next=None):
        filename = os.path.splitext(os.path.basename(reference_path))[0]
        if reference_path_next is not None:
            filename += '_' + os.path.splitext(os.path.basename(reference_path_next))[0]
        return filename + self.ext

    def _create_path_to_save(self, reference_path, reference_path_next=None):
        if reference_path is None:
            return None
        path_to_save = os.path.join(self.directory, self. _construct_filename(reference_path, reference_path_next))
        return path_to_save

    def run(self):
        pass

    def __repr__(self):
        return 'Estimator(dir={}, image_manager={}, checkpoint={})'.format(
            self.directory, self.image_manager, self.checkpoint)


class Struct2DepthEstimator(BaseEstimator):

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

    def run(self):
#         run_struct2depth(output_dir=self.directory,
#                          file_extension='jpg',
#                          depth=True,
#                          egomotion=False,
#                          model_ckpt=self.checkpoint,
#                          input_dir=self.image_manager.directory,
#                          input_list_file=self.image_manager.image_filenames,
#                          img_height=self.image_manager.height,
#                          img_width=self.image_manager.width)

        self._load_model()
        print("debug_1")
        supervisor = tf.train.Supervisor(logdir='/tmp/', saver=None)
        with supervisor.managed_session() as sess:
            self.saver.restore(sess, self.checkpoint)
            for path_to_rgb in self.image_manager.image_filenames:
                image = self.image_manager.load_image(path_to_rgb)
                model_input = self._convert_image_to_model_input(image)
                model_output = self.model.inference_depth(model_input[None], sess)[0]
                depth = self._convert_model_output_to_prediction(model_output)

                path_to_depth = self._create_path_to_save(path_to_rgb)
                np.save(path_to_depth, depth)
                self.mapping[path_to_rgb] = path_to_depth


class SENetDepthEstimator(BaseEstimator):

    def _load_model(self):
        self.model = se_net.senet154()
        checkpoint = torch.load(open(self.checkpoint, 'rb'))
        self.model.load_state_dict(checkpoint["state_dict"])

    def _convert_image_to_model_input(self, image):
        image_pil = PIL.Image.fromarray(image)
        depth_pil = PIL.Image.fromarray(image[:, :, 0])
        model_input = self.transform({'image': frame_pil, 'target': depth_pil, 'lines': None})['image']
        return model_input.unsqueeze(0)

    def _convert_model_output_to_prediction(self, output):
        return output.cpu().detach().numpy()

    def _set_transforms(self):
        imagenet_stats = {'mean': torch.Tensor([0.485, 0.456, 0.406]),
                          'std': torch.Tensor([0.229, 0.224, 0.225])}
        imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])
        }
        final_size = [320, 256]

        self.transform = transforms.Compose([
            Scale(min(final_size)),
            CenterCrop(final_size, final_size),
            ToTensor(is_test=False),
            Normalize(imagenet_stats['mean'],
            imagenet_stats['std'])
        ])

    def run(self):
        self._load_model()
        self._set_transforms()
        for path_to_rgb in self.image_manager.image_filenames:
            image = self.image_manager.load_image(path_to_rgb)
            model_input = self._convert_image_to_model_input(image)
            model_output = self.model(model_input)[-1][0, 0]
            depth = self._convert_model_output_to_prediction(model_output)

            path_to_depth = self._create_path_to_save(path_to_rgb)
            np.save(path_to_depth, depth)
            self.mapping[path_to_rgb] = path_to_depth


class PWCOpticalFlowEstimator(BaseEstimator):

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
        nn_opts['adapt_info'] = (1, self.image_manager.height, self.image_manager.width, 2)
        self.model = pwc_net(mode='test', options=nn_opts)

    def _convert_model_output_to_prediction(self, optical_flow):
        size = optical_flow.shape
        optical_flow_small = cv2.resize(optical_flow, (int(size[1] / 4), int(size[0] / 4)), cv2.INTER_LINEAR)
        optical_flow_small[..., 0] /= size[1]
        optical_flow_small[..., 1] /= size[0]
        return optical_flow_small

    def run(self):
        self._load_model()

        pairs = [image_pair for image_pair \
                 in zip(self.image_manager.image_filenames, self.image_manager.next_image_filenames) \
                 if image_pair[1] is not None]

        with tqdm.tqdm(pairs, total=len(pairs), desc='Optical flow estimation') as tbar:
            for path_to_rgb, path_to_next_rgb in tbar:
                first_image = self.image_manager.load_image(path_to_rgb)
                first_input = self._convert_image_to_model_input(first_image)

                second_image = self.image_manager.load_image(path_to_next_rgb)
                second_input = self._convert_image_to_model_input(second_image)
                inputs = [[first_input, second_input]]

                optical_flow = self.model.predict_from_img_pairs(inputs, batch_size=1, verbose=False)[0]
                optical_flow = self._convert_model_output_to_prediction(optical_flow)

                path_to_optical_flow = self._create_path_to_save(path_to_rgb, path_to_next_rgb)
                np.save(path_to_optical_flow, optical_flow)
                self.mapping[path_to_rgb] = path_to_optical_flow
