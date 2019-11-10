import os
import tqdm
import numpy as np
import torch

from submodules.depth_pred import senet_model as se_net
from .network_estimator import NetworkEstimator


class SENetEstimator(NetworkEstimator):

    def __init__(self, *args, **kwargs):
        super(SENetEstimator, self).__init__(*args, **kwargs)
        self.name = 'SENet'

    def _load_model(self):
        self.model = se_net.senet154()
        checkpoint = torch.load(open(self.checkpoint, 'rb'))
        self.model.load_state_dict(checkpoint['state_dict'])

    def _convert_image_to_model_input(self, image):
        image_pil = PIL.Image.fromarray(image)
        depth_pil = PIL.Image.fromarray(image[:, :, 0])
        model_input = self.transform({'image': frame_pil, 'target': depth_pil, 'lines': None})['image']
        return model_input.unsqueeze(0)

    def _convert_model_output_to_prediction(self, output):
        return output.cpu().detach().numpy()

    @attrsetter
    def transform(self):
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

        return transforms.Compose([
            Scale(min(final_size)),
            CenterCrop(final_size, final_size),
            ToTensor(is_test=False),
            Normalize(imagenet_stats['mean'],
            imagenet_stats['std'])
        ])

    def _run_model_inference(self, model_input):
        return self.model(inputs)[-1][0]
