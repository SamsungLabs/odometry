import os

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_flexible_model


class FlexibleWithAugmentationTrainer(BaseTrainer):

    def set_model_args(self):
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

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--generate_flow_by_rt_proba', type=float, default=1)
        parser.add_argument('--generate_flow_by_rt_mode', type=str,
                            choices=['constant', 'exp', 'linear', 'r_exp', 'r_linear'], default='constant')
        parser.add_argument('--generate_percentile', '-q', type=int, default=None)
        parser.add_argument('--generate_distribution', '--distr',
                            choices=[None, 'uniform', 'normal', 'student', 'shuffle', 'same'], type=str, default=None)
        parser.add_argument('--augment_with_rectangle_proba', type=float, default=0)
        parser.add_argument('--augment_with_rectangle_mode', type=str,
                            choices=['constant', 'exp', 'linear', 'r_exp', 'r_linear'], default='constant')
        return parser


if __name__ == '__main__':

    parser = FlexibleWithAugmentationTrainer.get_parser()
    args = parser.parse_args()

    args.train_generator_args = {
        'generate_flow_by_rt_proba': args.generate_flow_by_rt_proba,
        'generate_flow_by_rt_mode': args.generate_flow_by_rt_mode,
        'generate_percentile': args.generate_percentile,
        'generate_distribution': args.generate_distribution,
        'augment_with_rectangle_proba': args.augment_with_rectangle_proba,
        'augment_with_rectangle_mode': args.augment_with_rectangle_mode,
        'epochs': args.epochs
    }

    del args.generate_flow_by_rt_proba
    del args.generate_flow_by_rt_mode
    del args.generate_percentile
    del args.generate_distribution
    del args.augment_with_rectangle_proba
    del args.augment_with_rectangle_mode

    trainer = FlexibleWithAugmentationTrainer(**vars(args))
    trainer.train()
