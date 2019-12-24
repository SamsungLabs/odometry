import __init_path__

from scripts.base_trainer import BaseTrainer
from slam.models import PretrainedModelFactory


class PretrainedTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        self.weights = kwargs['weights']
        del kwargs['weights']
        super().__init__(*args, **kwargs)

    def set_model_args(self):
        self.lr = 0.001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow']
        self.load_mode = ['flow_xy']
        self.preprocess_mode = ['flow_xy']

    def get_model_factory(self, _input_shapes):
        return PretrainedModelFactory(self.weights)

    @staticmethod
    def get_parser():
        parser = BaseTrainer.get_parser()
        parser.add_argument('--weights', type=str, required=True)
        return parser


if __name__ == '__main__':
    args = PretrainedTrainer.get_parser().parse_args()

    trainer = PretrainedTrainer(**vars(args))
    trainer.train()
