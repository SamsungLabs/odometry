import os

import __init_path__
import env

from slam.base_trainer import BaseTrainer
from slam.models import construct_depth_flow_model


class DepthFlowTrainer(BaseTrainer):

    def set_model_args(self):
        self.construct_model_fn = construct_depth_flow_model
        self.lr = 0.0001
        self.loss = 'mae'
        self.scale_rotation = 50

    def set_dataset_args(self):
        self.x_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.y_col = ['euler_x', 'euler_y', 'euler_z', 't_x', 't_y', 't_z']
        self.image_col = ['path_to_optical_flow', 'path_to_depth', 'path_to_depth_next']
        self.load_mode = ['flow_xy', 'depth', 'depth']
        self.preprocess_mode = ['flow_xy', 'disparity', 'disparity']
        self.batch_size = 128
        self.target_size = self.config['target_size']

    def get_dataset(self, train_trajectories=None, val_trajectories=None):
        return super().get_dataset(train_trajectories=train_trajectories,
                                   val_trajectories=val_trajectories)

    def get_model_factory(self, input_shapes):
        return super().get_model_factory(input_shapes)

    def get_callbacks(self, model, dataset, evaluate=True, save_dir=None, prefix=None):
        return super().get_callbacks(model=model,
                                     dataset=dataset,
                                     evaluate=evaluate,
                                     save_dir=save_dir,
                                     prefix=prefix)

    def fit_generator(self, model, dataset, epochs, evaluate=True, save_dir=None, prefix=None):
        return super().fit_generator(model=model,
                                     dataset=dataset,
                                     epochs=epochs,
                                     evaluate=evaluate,
                                     save_dir=save_dir,
                                     prefix=prefix)

    def train(self):
        return super().train()


if __name__ == '__main__':

    parser = DepthFlowTrainer.get_parser()
    args = parser.parse_args()

    trainer = DepthFlowTrainer(**vars(args))
    trainer.train()
