import os

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_flexible_model

from slam.linalg import Intrinsics, create_optical_flow_from_rt


class FlexibleWithAugmentationTrainer(BaseTrainer):

    def set_model_args(self):
        self.train_generator_args = {
            'intrinsics': Intrinsics(f_x=0.5792554391619662,
                                     f_y=1.91185106382978721,
                                     c_x=0.48927703464947625,
                                     c_y=0.4925949468085106,
                                     width=320,
                                     height=96),
            'generate_flow_by_rt_proba': 0.5}

        self.construct_model_fn = construct_flexible_model
        self.lr = 0.001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow', 'path_to_binocular_depth']
        self.load_mode = ['flow_xy', 'depth']
        self.preprocess_mode = ['flow_xy', 'depth']


if __name__ == '__main__':

    parser = FlexibleWithAugmentationTrainer.get_parser()
    args = parser.parse_args()

    trainer = FlexibleWithAugmentationTrainer(**vars(args))
    trainer.train()
