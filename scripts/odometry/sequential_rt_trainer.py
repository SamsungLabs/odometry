import os
from functools import partial

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_sequential_rt_model
from slam.linalg import Intrinsics


class SequentialRTTrainer(BaseTrainer):
    def __init__(self,
                 dataset_root,
                 dataset_type,
                 run_name,
                 use_input_flow_for_translation,
                 use_cleaned_flow_for_translation,
                 use_rotation_flow_for_translation,
                 f_x,
                 f_y,
                 c_x,
                 c_y,
                 width,
                 height,
                 **kwargs):
        self.intrinsics = Intrinsics(f_x=f_x,
                                     f_y=f_y,
                                     c_x=c_x,
                                     c_y=c_y,
                                     width=width,
                                     height=height)
        self.use_input_flow_for_translation = use_input_flow_for_translation
        self.use_cleaned_flow_for_translation = use_cleaned_flow_for_translation
        self.use_rotation_flow_for_translation = use_rotation_flow_for_translation
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

    @staticmethod
    def get_parser():
        parser = super().get_parser()
        parser.add_argument('--use_input_flow_for_translation', type=bool, default=True)
        parser.add_argument('--use_cleaned_flow_for_translation', type=bool, default=True)
        parser.add_argument('--use_rotation_flow_for_translation', type=bool, default=False)
        parser.add_argument('--f_x', type=float, default=0.5792554391619662)
        parser.add_argument('--f_y', type=float, default=1.91185106382978721)
        parser.add_argument('--c_x', type=float, default=0.48927703464947625)
        parser.add_argument('--c_y', type=float, default=0.4925949468085106)
        parser.add_argument('--width', type=int, default=320)
        parser.add_argument('--height', type=int, default=96)
        return parser

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
