import os
from functools import partial

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_sequential_rt_model


class SequentialRTTrainer(BaseTrainer):
    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 intrinsics,
                 use_input_flow_for_translation=True,
                 use_cleaned_flow_for_translation=True,
                 use_rotation_flow_for_translation=False,
                 **kwargs):
        self.intrinsics = intrinsics
        self.use_input_flow_for_translation=use_input_flow_for_translation
        self.use_cleaned_flow_for_translation=use_cleaned_flow_for_translation
        self.use_rotation_flow_for_translation=use_rotation_flow_for_translation
        super().__init__(dataset_root,
                         dataset_type,
                         run_name,
                         **kwargs)

    def set_model_args(self):
        self.construct_model_fn = partial(
            construct_sequential_rt_model,
            intrinsics=self.intrinsics,
            use_input_flow_for_translation=self.use_input_flow_for_translation,
            use_cleaned_flow_for_translation=self.use_cleaned_flow_for_translation,
            use_rotation_flow_for_translation=self.use_rotation_flow_for_translation)
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
