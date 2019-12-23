from scripts.base_trainer import BaseTrainer
from slam.models import construct_resnet50_model


class ResNet50Trainer(BaseTrainer):

    def set_model_args(self):
        self.construct_model_fn = construct_resnet50_model
        self.lr = 0.001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow']
        self.load_mode = ['flow_xy']
        self.preprocess_mode = ['flow_xy']


if __name__ == '__main__':

    parser = ResNet50Trainer.get_parser()
    args = parser.parse_args()

    trainer = ResNet50Trainer(**vars(args))
    trainer.train()
