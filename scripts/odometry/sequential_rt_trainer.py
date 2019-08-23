import os

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_sequential_rt_model


class SequentialRTTrainer(BaseTrainer):

    def set_model_args(self):
        self.construct_model_fn = construct_sequential_rt_model
        self.lr = 0.001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow']
        self.load_mode = ['flow_xy']
        self.preprocess_mode = ['flow_xy']
        self.batch_size = 128


if __name__ == '__main__':

    parser = SequentialRTTrainer.get_parser()
    args = parser.parse_args()

    trainer = SequentialRTTrainer(**vars(args))
    trainer.train()
