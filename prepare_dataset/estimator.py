import os
import shutil
import copy
import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.python.client import device_lib

#import torch

from tfoptflow.tfoptflow.model_pwcnet import _DEFAULT_PWCNET_TEST_OPTIONS
from tfoptflow.tfoptflow.model_pwcnet import ModelPWCNet as pwc_net

#from struct2depth.inference import _run_inference as run_struct2depth
from struct2depth.model import Model as struct2depth_net
from struct2depth.nets import RESNET
from struct2depth.util import get_vars_to_save_and_restore

#from depth_pred import senet_model as se_net


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
        return np.array(image, dtype=np.float32) / 255.

    def _convert_model_output_to_prediction(self, output):
        return output

    def _create_path_to_save(self, reference_path):
        if reference_path is None:
            return None
        filename = ''.join((os.path.splitext(os.path.basename(reference_path))[0], self.ext))
        path_to_save = os.path.join(self.directory, filename)
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
        print(self)

    def _convert_model_output_to_prediction(self, output):
        optical_flow = output * 20.0 / 4
        optical_flow[..., 0] /= optical_flow.shape[1]
        optical_flow[..., 1] /= optical_flow.shape[0]
        return optical_flow

    @staticmethod
    def make_divisible_by_64(number):
        divisor = 64.
        return int(np.ceil(number / divisor) * divisor)

    def _set_input_size(self):
        self.input_size = (self.make_divisible_by_64(self.image_manager.width),
                           self.make_divisible_by_64(self.image_manager.height))

    def run(self):
        self._load_model()

        self._set_input_size()
        first_input = None

        pairs = [image_pair for image_pair \
                 in zip(self.image_manager.image_filenames, self.image_manager.next_image_filenames) \
                 if image_pair[1] is not None]

        with tqdm.tqdm(pairs, total=len(pairs), desc='Optical flow estimation') as tbar:
            for path_to_rgb, path_to_next_rgb in tbar:
                if first_input is None:
                    first_image = self.image_manager.load_image(path_to_rgb, self.input_size)
                    first_input = self._convert_image_to_model_input(first_image)

                second_image = self.image_manager.load_image(path_to_next_rgb, self.input_size)
                second_input = self._convert_image_to_model_input(second_image)
                inputs = np.stack([first_input, second_input])[None]
                optical_flow = self.model.predict_from_img_pairs(inputs, batch_size=1, verbose=False)[0]

                path_to_optical_flow = self._create_path_to_save(path_to_next_rgb)
                np.save(path_to_optical_flow, optical_flow)
                self.mapping[path_to_rgb] = path_to_optical_flow

                first_input = second_input
